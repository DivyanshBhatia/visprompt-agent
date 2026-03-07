#!/usr/bin/env python3
"""Full cross-product ablation: 3 LLMs × 3 CLIP backbones × 2 datasets.

Efficiently reuses cached image features across LLM runs on same backbone.
Runs both classification and retrieval for each config.

Example:
    python scripts/run_cross_ablation.py \
        --datasets cifar100 flowers102 \
        --anthropic-key sk-ant-... \
        --openai-key sk-...
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────
CLIP_BACKBONES = [
    # OpenAI CLIP family
    ("ViT-B/32", "openai", "CLIP-B/32"),
    ("ViT-B/16", "openai", "CLIP-B/16"),
    ("ViT-L/14", "openai", "CLIP-L/14"),
    # EVA-CLIP family (better training, same arch)
    ("EVA02-B-16", "merged2b_s8b_b131k", "EVA02-B/16"),
    # MetaCLIP family (curated data from Meta)
    ("ViT-B-32-quickgelu", "metaclip_400m", "MetaCLIP-B/32"),
    ("ViT-L-14-quickgelu", "metaclip_400m", "MetaCLIP-L/14"),
    # SigLIP family (Google, sigmoid loss)
    ("ViT-B-16-SigLIP", "webli", "SigLIP-B/16"),
]

LLM_CONFIGS = [
    {"model": "gpt-4o", "provider": "openai", "label": "GPT-4o"},
    {"model": "gpt-5.2", "provider": "openai", "label": "GPT-5.2"},
    {"model": "claude-sonnet-4-20250514", "provider": "anthropic", "label": "Claude-Sonnet-4"},
    {"model": "claude-opus-4-5-20251101", "provider": "anthropic", "label": "Claude-Opus-4.5"},
]

DATASET_CONFIGS = {
    "cifar10": {"val_size": 10000},
    "cifar100": {"val_size": 10000},
    "flowers102": {"val_size": 6149},
    "dtd": {"val_size": 1880},
    "food101": {"val_size": 10000},
    "eurosat": {"val_size": 5000},
    "oxford_pets": {"val_size": 3669},
    "caltech101": {"val_size": 6084},
    "fgvc_aircraft": {"val_size": 3333},
    "country211": {"val_size": 10000},
}

WEIGHT_CONFIGS = [
    (1.0, 0.0, "100/0"),
    (0.85, 0.15, "85/15"),
    (0.7, 0.3, "70/30"),
    (0.55, 0.45, "55/45"),
    (0.4, 0.6, "40/60"),
    (0.2, 0.8, "20/80"),
    (0.0, 1.0, "0/100"),
]


def generate_descriptions(task_spec, llm_model, llm_provider):
    """Generate descriptions using specified LLM."""
    from visprompt.utils.llm import CostTracker, LLMClient

    cost_tracker = CostTracker()
    llm = LLMClient(model=llm_model, provider=llm_provider, cost_tracker=cost_tracker)

    all_descriptions = {}
    batch_size = 10

    for i in range(0, len(task_spec.class_names), batch_size):
        batch = task_spec.class_names[i:i + batch_size]
        prompt = (
            f"Generate 10-15 short visual descriptions for each class below.\n"
            f"Format: \"a {{class_name}}, {{visual description}}\"\n"
            f"Keep under 15 words. Focus on shape, color, size, texture.\n\n"
            f"IMPORTANT: Make each description DIFFERENT — vary the visual angle.\n\n"
            f"Classes: {', '.join(batch)}\n\n"
            f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
        )
        try:
            result = llm.call_json(
                prompt=prompt,
                system="Generate visual descriptions for zero-shot image classification.",
                agent_name="cross_ablation",
            )
            for cls in batch:
                descs = result.get(cls, result.get(cls.replace("_", " "), []))
                if isinstance(descs, list) and descs:
                    cleaned = []
                    for d in descs:
                        cls_display = cls.replace("_", " ")
                        if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                            d = f"a {cls_display}, {d}"
                        cleaned.append(d)
                    all_descriptions[cls] = cleaned
                else:
                    all_descriptions[cls] = [f"a {cls.replace('_', ' ')}"]
        except Exception as e:
            logger.warning(f"Description batch failed ({llm_model}): {e}")
            for cls in batch:
                all_descriptions[cls] = [f"a {cls.replace('_', ' ')}"]

    return all_descriptions, cost_tracker.summary()


def build_prompts(class_names, descriptions, base_weight, desc_weight, templates):
    """Build group-normalized prompts."""
    prompts_per_class = {}
    for cls_name in class_names:
        base_prompts = [t.format(cls_name) for t in templates]
        desc_prompts = descriptions.get(cls_name, [])
        n_base = len(base_prompts)
        n_desc = len(desc_prompts)

        if n_desc > 0 and desc_weight > 0:
            per_base = base_weight / n_base
            per_desc = desc_weight / n_desc
            all_prompts = base_prompts + desc_prompts
            all_weights = [per_base] * n_base + [per_desc] * n_desc
        else:
            all_prompts = base_prompts
            all_weights = [1.0 / n_base] * n_base

        prompts_per_class[cls_name] = {"prompts": all_prompts, "weights": all_weights}
    return prompts_per_class


def evaluate_classification(task_runner, prompts_per_class, task_spec):
    """Run classification evaluation."""
    prompts = {
        "type": "classification",
        "prompts_per_class": prompts_per_class,
        "ensemble_method": "weighted_average",
    }
    result = task_runner.evaluate(prompts, task_spec)
    return result.primary_metric


def evaluate_retrieval(task_runner, prompts_per_class, class_names, labels):
    """Run retrieval evaluation."""
    import torch

    # Encode text
    all_class_emb = []
    for cls_name in class_names:
        cls_info = prompts_per_class.get(cls_name, {"prompts": [cls_name], "weights": [1.0]})
        cls_prompts = cls_info["prompts"]
        cls_weights = cls_info.get("weights", [1.0] * len(cls_prompts))

        text_tokens = task_runner._tokenizer(cls_prompts).to(task_runner.device)
        with torch.no_grad():
            text_features = task_runner._clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        weights = torch.tensor(cls_weights, device=task_runner.device, dtype=text_features.dtype)
        weights = weights / weights.sum()
        weighted = (text_features * weights.unsqueeze(-1)).sum(dim=0)
        weighted = weighted / weighted.norm()
        all_class_emb.append(weighted)

    text_emb = torch.stack(all_class_emb)

    # Get cached image features
    image_features = task_runner._encode_images_cached(
        "orig", lambda img: task_runner._clip_preprocess(img)
    )

    # Compute similarities and mAP
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    n_classes = len(class_names)

    all_aps = []
    for cls_idx in range(n_classes):
        ranked = np.argsort(-sims[cls_idx])
        relevant = (labels == cls_idx)
        n_rel = relevant.sum()
        if n_rel == 0:
            continue
        tp = 0
        precs = []
        for rank, idx in enumerate(ranked):
            if relevant[idx]:
                tp += 1
                precs.append(tp / (rank + 1))
        all_aps.append(np.mean(precs) if precs else 0.0)

    return float(np.mean(all_aps)) if all_aps else 0.0


def main():
    parser = argparse.ArgumentParser(description="Full cross-product ablation")
    parser.add_argument("--datasets", nargs="+", default=["cifar100", "flowers102"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--backbones", nargs="+",
                        default=["CLIP-B/32", "CLIP-B/16", "CLIP-L/14",
                                 "EVA02-B/16", "MetaCLIP-B/32", "MetaCLIP-L/14",
                                 "SigLIP-B/16"],
                        help="Backbone labels to test (from CLIP_BACKBONES)")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--anthropic-key", type=str, help="Anthropic API key")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="experiments/cross_ablation")
    parser.add_argument("--llms", nargs="+", default=None,
                        help="Run only specific LLMs by label (e.g. --llms GPT-5.2 Claude-Opus-4.5)")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args for build_task_spec compatibility
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Set API keys
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key

    # Filter LLMs if --llms specified
    if args.llms:
        llm_configs = [c for c in LLM_CONFIGS if c["label"] in args.llms]
        if not llm_configs:
            print(f"ERROR: No matching LLMs found. Available: {[c['label'] for c in LLM_CONFIGS]}")
            sys.exit(1)
        print(f"Running only LLMs: {[c['label'] for c in llm_configs]}")
    else:
        llm_configs = LLM_CONFIGS

    from scripts.run import build_task_spec, build_task_runner
    from visprompt.baselines import IMAGENET_TEMPLATES

    all_results = {}

    for dataset in args.datasets:
        print(f"\n{'#'*70}")
        print(f"#  DATASET: {dataset.upper()}")
        print(f"{'#'*70}")

        # Build task spec
        args.dataset = dataset
        args.task = "classification"
        args.val_size = DATASET_CONFIGS[dataset]["val_size"]
        task_spec = build_task_spec(args)
        class_names = list(task_spec.class_names)

        dataset_results = {}

        # Filter backbones by user selection
        selected_backbones = [
            bb for bb in CLIP_BACKBONES if bb[2] in args.backbones
        ]
        if not selected_backbones:
            logger.warning(f"No matching backbones found for {args.backbones}")
            continue

        for bb_model, bb_pretrained, bb_label in selected_backbones:
            print(f"\n{'='*65}")
            print(f"  BACKBONE: {bb_label} ({bb_model}, pretrained={bb_pretrained})")
            print(f"{'='*65}")

            # Build runner for this backbone
            args.clip_model = bb_model
            # Create runner directly with pretrained param
            from visprompt.tasks.classification import CLIPClassificationRunner
            from scripts.run import _load_pil_dataset

            args.val_size = DATASET_CONFIGS[dataset]["val_size"]

            # CIFAR-10/100 have .data attribute (numpy arrays directly)
            if dataset in ("cifar10", "cifar100"):
                import torchvision
                ds_cls = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10
                ds = ds_cls(root=args.data_dir or "./data", train=False, download=True)
                n_val = DATASET_CONFIGS[dataset]["val_size"]
                indices = np.random.RandomState(42).permutation(len(ds))[:n_val]
                images = np.array(ds.data)[indices]
                labels_arr = np.array(ds.targets)[indices]
            else:
                # All other datasets use _load_pil_dataset
                DATASET_LOADERS = {
                    "flowers102": ("Flowers102", {"split": "test"}),
                    "dtd": ("DTD", {"split": "test"}),
                    "food101": ("Food101", {"split": "test"}),
                    "eurosat": ("EuroSAT", {}),
                    "oxford_pets": ("OxfordIIITPet", {"split": "test"}),
                    "caltech101": ("Caltech101", {}),
                    "fgvc_aircraft": ("FGVCAircraft", {"split": "test"}),
                    "country211": ("Country211", {"split": "test"}),
                }
                if dataset not in DATASET_LOADERS:
                    raise ValueError(f"Dataset {dataset} not supported in cross-ablation")
                loader_name, loader_kwargs = DATASET_LOADERS[dataset]
                images, labels_arr = _load_pil_dataset(
                    loader_name, args, split=loader_kwargs.get("split"),
                    dataset_kwargs=loader_kwargs,
                )

            task_runner = CLIPClassificationRunner(
                clip_model_name=bb_model,
                pretrained=bb_pretrained,
                device=args.device,
                images=images,
                labels=labels_arr,
            )
            task_runner._ensure_model()
            task_runner.load_data()

            # Cache image features once per backbone
            from PIL import ImageOps
            task_runner._encode_images_cached(
                "orig", lambda img: task_runner._clip_preprocess(img)
            )
            labels = task_runner._labels

            # Templates-only baseline (no LLM needed)
            ppc_templates = build_prompts(
                class_names, {}, 1.0, 0.0, IMAGENET_TEMPLATES
            )
            cls_baseline = evaluate_classification(task_runner, ppc_templates, task_spec)
            ret_baseline = evaluate_retrieval(task_runner, ppc_templates, class_names, labels)
            print(f"  Templates-only: cls={cls_baseline:.4f}, ret_mAP={ret_baseline:.4f}")

            backbone_results = {
                "templates_only": {
                    "classification": cls_baseline,
                    "retrieval_mAP": ret_baseline,
                }
            }

            for llm_cfg in llm_configs:
                llm_model = llm_cfg["model"]
                llm_provider = llm_cfg["provider"]
                llm_label = llm_cfg["label"]

                # Check API key availability
                if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
                    print(f"\n  Skipping {llm_label} — no OPENAI_API_KEY")
                    continue
                if llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
                    print(f"\n  Skipping {llm_label} — no ANTHROPIC_API_KEY")
                    continue

                print(f"\n  --- LLM: {llm_label} ---")

                # Generate descriptions
                t0 = time.time()
                descriptions, desc_cost = generate_descriptions(
                    task_spec, llm_model, llm_provider
                )
                cost = desc_cost.get("total_cost_usd", 0)
                print(f"  Descriptions generated ({len(descriptions)} classes, "
                      f"${cost:.4f}, {time.time()-t0:.0f}s)")

                llm_results = {"cost": cost, "weights": []}

                # Test weight configs
                print(f"  {'Config':<12} {'Cls':>8} {'Ret mAP':>8} "
                      f"{'Δ Cls':>8} {'Δ Ret':>8}")
                print(f"  {'-'*50}")

                for base_w, desc_w, label in WEIGHT_CONFIGS:
                    ppc = build_prompts(
                        class_names, descriptions, base_w, desc_w, IMAGENET_TEMPLATES
                    )
                    cls_acc = evaluate_classification(task_runner, ppc, task_spec)
                    ret_map = evaluate_retrieval(task_runner, ppc, class_names, labels)
                    d_cls = cls_acc - cls_baseline
                    d_ret = ret_map - ret_baseline

                    print(f"  {label:<12} {cls_acc:>8.4f} {ret_map:>8.4f} "
                          f"{d_cls:>+7.4f} {d_ret:>+7.4f}")

                    llm_results["weights"].append({
                        "config": label,
                        "base_weight": base_w,
                        "desc_weight": desc_w,
                        "classification": cls_acc,
                        "retrieval_mAP": ret_map,
                    })

                # Find best for this LLM
                best_cls = max(llm_results["weights"], key=lambda x: x["classification"])
                best_ret = max(llm_results["weights"], key=lambda x: x["retrieval_mAP"])
                print(f"\n  Best cls: {best_cls['config']} → {best_cls['classification']:.4f} "
                      f"(+{best_cls['classification']-cls_baseline:.4f})")
                print(f"  Best ret: {best_ret['config']} → {best_ret['retrieval_mAP']:.4f} "
                      f"(+{best_ret['retrieval_mAP']-ret_baseline:.4f})")

                backbone_results[llm_label] = llm_results

            dataset_results[bb_label] = backbone_results

        all_results[dataset] = dataset_results

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CROSS-ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'Backbone':<12} {'LLM':<15} {'Best Cls':>10} "
          f"{'Δ Cls':>8} {'Best Ret':>10} {'Δ Ret':>8}")
    print(f"{'-'*78}")

    for dataset, ds_res in all_results.items():
        for backbone, bb_res in ds_res.items():
            tmpl_cls = bb_res["templates_only"]["classification"]
            tmpl_ret = bb_res["templates_only"]["retrieval_mAP"]
            print(f"{dataset:<12} {backbone:<12} {'templates':<15} "
                  f"{tmpl_cls:>10.4f} {'---':>8} {tmpl_ret:>10.4f} {'---':>8}")

            for llm_label, llm_res in bb_res.items():
                if llm_label == "templates_only":
                    continue
                best_cls = max(llm_res["weights"], key=lambda x: x["classification"])
                best_ret = max(llm_res["weights"], key=lambda x: x["retrieval_mAP"])
                d_cls = best_cls["classification"] - tmpl_cls
                d_ret = best_ret["retrieval_mAP"] - tmpl_ret
                print(f"{'':<12} {'':<12} {llm_label:<15} "
                      f"{best_cls['classification']:>10.4f} {d_cls:>+7.4f} "
                      f"{best_ret['retrieval_mAP']:>10.4f} {d_ret:>+7.4f}")

    # Save
    output_path = Path(args.output_dir) / "cross_ablation_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
