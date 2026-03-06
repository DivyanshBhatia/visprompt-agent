#!/usr/bin/env python3
"""Zero-shot image retrieval baselines using CLIP.

Same CLIP model + embeddings as classification. Text query → retrieve images.
Our prompt ensembling directly transfers: better text embeddings → better retrieval.

Example:
    python scripts/run_retrieval_baselines.py --dataset cifar100 --clip-model ViT-L/14
    python scripts/run_retrieval_baselines.py --dataset flowers102 --clip-model ViT-L/14 --run-ablation
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.run import build_task_runner, build_task_spec
from visprompt.baselines import IMAGENET_TEMPLATES

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(similarities, labels, class_names, k_values=[1, 5, 10, 20]):
    """Compute retrieval metrics from similarity matrix.

    Args:
        similarities: (n_classes, n_images) cosine similarity matrix
        labels: (n_images,) ground truth class indices
        class_names: list of class names
        k_values: list of K values for Recall@K

    Returns:
        dict with mAP, Recall@K, per-class AP
    """
    n_classes = similarities.shape[0]
    n_images = similarities.shape[1]

    # Per-class Average Precision
    per_class_ap = {}
    all_aps = []

    for cls_idx in range(n_classes):
        # Rank images by similarity to this class query
        sims = similarities[cls_idx]
        ranked_indices = np.argsort(-sims)  # descending

        # Ground truth: which images belong to this class
        relevant = (labels == cls_idx)
        n_relevant = relevant.sum()

        if n_relevant == 0:
            continue

        # Compute AP
        tp_cumsum = 0
        precisions = []
        for rank, img_idx in enumerate(ranked_indices):
            if relevant[img_idx]:
                tp_cumsum += 1
                precisions.append(tp_cumsum / (rank + 1))

        ap = np.mean(precisions) if precisions else 0.0
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        per_class_ap[cls_name] = float(ap)
        all_aps.append(ap)

    mAP = float(np.mean(all_aps)) if all_aps else 0.0

    # Recall@K (averaged across classes)
    recall_at_k = {}
    for k in k_values:
        recalls = []
        for cls_idx in range(n_classes):
            relevant = (labels == cls_idx)
            n_relevant = relevant.sum()
            if n_relevant == 0:
                continue

            sims = similarities[cls_idx]
            top_k_indices = np.argsort(-sims)[:k]
            retrieved_relevant = relevant[top_k_indices].sum()
            recalls.append(retrieved_relevant / min(n_relevant, k))

        recall_at_k[f"R@{k}"] = float(np.mean(recalls)) if recalls else 0.0

    return {
        "mAP": mAP,
        **recall_at_k,
        "per_class_ap": per_class_ap,
    }


def encode_text_prompts(runner, prompts_per_class, class_names):
    """Encode text prompts into averaged class embeddings using CLIP."""
    torch = runner._torch
    all_class_embeddings = []

    for cls_name in class_names:
        cls_info = prompts_per_class.get(cls_name, {
            "prompts": [cls_name], "weights": [1.0]
        })
        cls_prompts = cls_info["prompts"]
        cls_weights = cls_info.get("weights", [1.0] * len(cls_prompts))

        if not cls_prompts:
            cls_prompts = [cls_name]
            cls_weights = [1.0]

        # Encode and weight-average
        text_tokens = runner._tokenizer(cls_prompts).to(runner.device)
        with torch.no_grad():
            text_features = runner._clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        weights = torch.tensor(cls_weights, device=runner.device, dtype=text_features.dtype)
        weights = weights / weights.sum()
        weighted = (text_features * weights.unsqueeze(-1)).sum(dim=0)
        weighted = weighted / weighted.norm()
        all_class_embeddings.append(weighted)

    # (n_classes, embed_dim)
    return torch.stack(all_class_embeddings)


def build_prompts_with_weights(class_names, descriptions, base_weight, desc_weight):
    """Build prompt dict with group-normalized weights."""
    prompts_per_class = {}

    for cls_name in class_names:
        base_prompts = [t.format(cls_name) for t in IMAGENET_TEMPLATES]
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

        prompts_per_class[cls_name] = {
            "prompts": all_prompts,
            "weights": all_weights,
        }

    return prompts_per_class


def generate_descriptions(task_spec, llm_model, llm_provider):
    """Generate descriptions for retrieval classes."""
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
            f"Classes: {', '.join(batch)}\n\n"
            f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
        )
        try:
            result = llm.call_json(
                prompt=prompt,
                system="Generate visual descriptions for image retrieval.",
                agent_name="retrieval_descriptions",
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
            logger.warning(f"Description batch failed: {e}")
            for cls in batch:
                all_descriptions[cls] = [f"a {cls.replace('_', ' ')}"]

    return all_descriptions, cost_tracker.summary()


def main():
    parser = argparse.ArgumentParser(description="Zero-shot retrieval baselines")
    parser.add_argument("--task", default="classification")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--output-dir", type=str, default="experiments/retrieval")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    # Ensure model is loaded
    task_runner._ensure_model()
    task_runner.load_data()

    torch = task_runner._torch
    class_names = list(task_spec.class_names)

    # ── Encode all images once (cached) ───────────────────────────────
    from PIL import Image, ImageOps

    print(f"\n{'='*65}")
    print(f"  ZERO-SHOT RETRIEVAL — {args.dataset}, {args.clip_model}")
    print(f"  {len(class_names)} classes, {len(task_runner._labels)} images")
    print(f"{'='*65}")

    image_features = task_runner._encode_images_cached(
        "orig", lambda img: task_runner._clip_preprocess(img)
    )
    labels = task_runner._labels
    n_images = image_features.shape[0]

    print(f"  Image features: {image_features.shape}")

    results = {}

    # ── Baseline 1: Class name only ───────────────────────────────────
    print(f"\n--- Baseline 1: Class name only ---")
    t0 = time.time()
    ppc = {cls: {"prompts": [cls], "weights": [1.0]} for cls in class_names}
    text_emb = encode_text_prompts(task_runner, ppc, class_names)
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    metrics = compute_retrieval_metrics(sims, labels, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}, R@1: {metrics['R@1']:.4f}, "
          f"R@5: {metrics['R@5']:.4f}, R@10: {metrics['R@10']:.4f}  ({time.time()-t0:.1f}s)")
    results["class_name_only"] = metrics

    # ── Baseline 2: "a photo of a {class}" ────────────────────────────
    print(f"\n--- Baseline 2: 'a photo of a {{class}}' ---")
    t0 = time.time()
    ppc = {cls: {"prompts": [f"a photo of a {cls}"], "weights": [1.0]} for cls in class_names}
    text_emb = encode_text_prompts(task_runner, ppc, class_names)
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    metrics = compute_retrieval_metrics(sims, labels, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}, R@1: {metrics['R@1']:.4f}, "
          f"R@5: {metrics['R@5']:.4f}, R@10: {metrics['R@10']:.4f}  ({time.time()-t0:.1f}s)")
    results["photo_template"] = metrics

    # ── Baseline 3: 80-template ensemble ──────────────────────────────
    print(f"\n--- Baseline 3: 80-template ensemble ---")
    t0 = time.time()
    ppc = {
        cls: {
            "prompts": [t.format(cls) for t in IMAGENET_TEMPLATES],
            "weights": [1.0] * len(IMAGENET_TEMPLATES),
        }
        for cls in class_names
    }
    text_emb = encode_text_prompts(task_runner, ppc, class_names)
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    metrics = compute_retrieval_metrics(sims, labels, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}, R@1: {metrics['R@1']:.4f}, "
          f"R@5: {metrics['R@5']:.4f}, R@10: {metrics['R@10']:.4f}  ({time.time()-t0:.1f}s)")
    results["80_template_ensemble"] = metrics
    baseline_map = metrics["mAP"]

    # ── Generate descriptions ─────────────────────────────────────────
    print(f"\n--- Generating LLM descriptions ---")
    descriptions, desc_cost = generate_descriptions(
        task_spec, args.llm, args.llm_provider
    )
    print(f"  Generated for {len(descriptions)} classes "
          f"(cost: ${desc_cost.get('total_cost_usd', 0):.4f})")

    # ── Baseline 4: Descriptions only ─────────────────────────────────
    print(f"\n--- Baseline 4: LLM descriptions only ---")
    t0 = time.time()
    ppc = {
        cls: {"prompts": descriptions.get(cls, [cls]),
              "weights": [1.0] * len(descriptions.get(cls, [cls]))}
        for cls in class_names
    }
    text_emb = encode_text_prompts(task_runner, ppc, class_names)
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    metrics = compute_retrieval_metrics(sims, labels, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}, R@1: {metrics['R@1']:.4f}, "
          f"R@5: {metrics['R@5']:.4f}, R@10: {metrics['R@10']:.4f}  ({time.time()-t0:.1f}s)")
    results["llm_descriptions"] = metrics

    # ── Baseline 5: CuPL+ensemble (equal weight) ─────────────────────
    print(f"\n--- Baseline 5: CuPL+ensemble (equal weight) ---")
    t0 = time.time()
    ppc = {}
    for cls in class_names:
        tmpl = [t.format(cls) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(cls, [])
        all_p = tmpl + descs
        ppc[cls] = {"prompts": all_p, "weights": [1.0] * len(all_p)}
    text_emb = encode_text_prompts(task_runner, ppc, class_names)
    sims = (text_emb @ image_features.T).detach().cpu().numpy()
    metrics = compute_retrieval_metrics(sims, labels, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}, R@1: {metrics['R@1']:.4f}, "
          f"R@5: {metrics['R@5']:.4f}, R@10: {metrics['R@10']:.4f}  ({time.time()-t0:.1f}s)")
    results["cupl_ensemble"] = metrics

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RETRIEVAL COMPARISON — {args.dataset}")
    print(f"{'='*65}")
    print(f"{'Method':<25} {'mAP':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print(f"{'-'*60}")
    for name in ["class_name_only", "photo_template", "80_template_ensemble",
                  "llm_descriptions", "cupl_ensemble"]:
        if name in results:
            m = results[name]
            print(f"  {name:<23} {m['mAP']:>8.4f} {m['R@1']:>8.4f} "
                  f"{m['R@5']:>8.4f} {m['R@10']:>8.4f}")
    print(f"{'='*65}")

    # ── Weight ablation ───────────────────────────────────────────────
    if args.run_ablation:
        print(f"\n{'='*65}")
        print(f"  WEIGHT ABLATION — {args.dataset}")
        print(f"{'='*65}")
        print(f"{'Config':<30} {'mAP':>8} {'R@1':>8} {'Δ mAP':>10}")
        print(f"{'-'*60}")

        ablation_results = []
        for base_w, desc_w, label in [
            (1.0, 0.0, "100/0 (templates only)"),
            (0.95, 0.05, "95/5"),
            (0.9, 0.1, "90/10"),
            (0.85, 0.15, "85/15"),
            (0.8, 0.2, "80/20"),
            (0.7, 0.3, "70/30"),
            (0.55, 0.45, "55/45"),
            (0.4, 0.6, "40/60"),
            (0.2, 0.8, "20/80"),
            (0.0, 1.0, "0/100 (descriptions only)"),
        ]:
            ppc = build_prompts_with_weights(class_names, descriptions, base_w, desc_w)
            text_emb = encode_text_prompts(task_runner, ppc, class_names)
            sims = (text_emb @ image_features.T).detach().cpu().numpy()
            metrics = compute_retrieval_metrics(sims, labels, class_names)
            delta = metrics["mAP"] - baseline_map
            print(f"  {label:<28} {metrics['mAP']:>8.4f} {metrics['R@1']:>8.4f} {delta:>+9.4f}")
            ablation_results.append({
                "base_weight": base_w, "desc_weight": desc_w,
                "label": label, **metrics,
            })

        best = max(ablation_results, key=lambda x: x["mAP"])
        print(f"\n  Best: {best['label']} → mAP={best['mAP']:.4f} "
              f"(Δ={best['mAP']-baseline_map:+.4f} vs templates)")
        results["ablation"] = ablation_results

    # ── Save ──────────────────────────────────────────────────────────
    output_path = Path(args.output_dir) / f"retrieval_{args.dataset}_{args.clip_model.replace('/', '_')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove per_class_ap from saved results (too large)
    save_results = {}
    for k, v in results.items():
        if isinstance(v, dict) and "per_class_ap" in v:
            v = {kk: vv for kk, vv in v.items() if kk != "per_class_ap"}
        save_results[k] = v

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
