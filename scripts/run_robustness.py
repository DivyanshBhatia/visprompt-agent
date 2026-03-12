#!/usr/bin/env python3
"""Robustness experiments for Appendix:
  1. Held-out validation: split datasets into selection/held-out sets
  2. Template sensitivity: test with 7, 20, 40, 80 templates

Usage:
    python scripts/run_robustness.py \
        --datasets cifar10 cifar100 flowers102 dtd food101 \
                   oxford_pets caltech101 fgvc_aircraft eurosat country211 \
        --clip-model ViT-L/14 --llm gpt-4o \
        --experiments heldout templates
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_spec, build_task_runner
from visprompt.baselines import IMAGENET_TEMPLATES

logger = logging.getLogger(__name__)

# 7-template subset (commonly used in early CLIP work)
SEVEN_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
    "a photo of a {} in a video game.",
    "art of a {}.",
    "a photo of the {}.",
]

# 20-template diverse subset
TWENTY_TEMPLATES = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a dark photo of the {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a photo of a dirty {}.",
    "a photo of a cool {}.",
    "a painting of the {}.",
    "a pixelated photo of the {}.",
    "a jpeg corrupted photo of a {}.",
    "a good photo of a {}.",
    "a photo of the nice {}.",
]


def load_descriptions(dataset_name, llm, output_dir, task_spec=None, llm_provider="openai"):
    """Load cached descriptions, or generate and cache them if not found."""
    # Try multiple naming patterns
    patterns = [
        f"descriptions_{dataset_name}_{llm}.json",
        f"desc_{dataset_name}_{llm}.json",
        f"desc_{dataset_name}_general_{llm}.json",
    ]
    for pat in patterns:
        p = Path(output_dir) / pat
        if p.exists():
            with open(p) as f:
                print(f"  Loaded cached descriptions from {p.name}")
                return json.load(f)

    # Also search recursively
    for p in Path(output_dir).rglob(f"*{dataset_name}*desc*.json"):
        with open(p) as f:
            print(f"  Found descriptions at {p}")
            return json.load(f)
    for p in Path(output_dir).rglob(f"*desc*{dataset_name}*.json"):
        with open(p) as f:
            print(f"  Found descriptions at {p}")
            return json.load(f)

    # Not found — generate
    if task_spec is None:
        print(f"  No cached descriptions and no task_spec to generate")
        return None

    print(f"  Generating descriptions for {dataset_name} with {llm}...")
    from scripts.run_weight_ablation import generate_descriptions
    descriptions, cost = generate_descriptions(task_spec, llm, llm_provider)
    print(f"  Generated {len(descriptions)} classes (cost: {cost})")

    # Cache for next time
    cache_path = Path(output_dir) / f"descriptions_{dataset_name}_{llm}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(descriptions, f, indent=1)
    print(f"  Cached to {cache_path}")

    return descriptions


@torch.no_grad()
def evaluate_config(task_runner, task_spec, class_names, descriptions,
                    templates, alpha, beta, device):
    """Evaluate a single alpha/beta config with given templates."""
    import open_clip

    model = task_runner.model
    tokenizer = task_runner.tokenizer

    M = len(templates)
    ppc = {}
    for cls in class_names:
        tmpl_prompts = [t.format(cls) for t in templates] if alpha > 0 else []
        descs = descriptions.get(cls, [f"a photo of a {cls}"]) if beta > 0 else []
        N = len(descs)

        prompts = []
        weights = []
        if alpha > 0:
            prompts.extend(tmpl_prompts)
            weights.extend([alpha / M] * M)
        if beta > 0:
            prompts.extend(descs)
            weights.extend([beta / N] * N)
        if not prompts:
            prompts = [f"a photo of a {cls}"]
            weights = [1.0]

        ppc[cls] = {"prompts": prompts, "weights": weights}

    # Encode text
    class_embs = []
    for cls in class_names:
        p = ppc[cls]
        tokens = tokenizer(p["prompts"]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        w = torch.tensor(p["weights"], dtype=torch.float32, device=device)
        w = w / w.sum()
        cls_emb = (emb * w.unsqueeze(1)).sum(dim=0)
        cls_emb = cls_emb / cls_emb.norm()
        class_embs.append(cls_emb)
    class_embs = torch.stack(class_embs)

    # Get image features from task runner
    if hasattr(task_runner, '_cached_image_features'):
        image_features = task_runner._cached_image_features
        labels = task_runner._cached_labels
    else:
        # Run through dataset
        image_features, labels = task_runner.encode_images()
        task_runner._cached_image_features = image_features
        task_runner._cached_labels = labels

    sims = image_features.to(device) @ class_embs.T
    preds = sims.argmax(dim=1).cpu()
    labels_t = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
    acc = (preds == labels_t).float().mean().item()
    return acc


def run_heldout_experiment(args, all_datasets):
    """Experiment 1: Held-out validation for 55/45 default."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1: Held-Out Validation for 55/45 Default")
    print(f"{'='*70}")

    # Two splits: A selects, B validates, then swap
    split_A = ["cifar100", "flowers102", "dtd", "oxford_pets", "eurosat"]
    split_B = ["cifar10", "food101", "caltech101", "fgvc_aircraft", "country211"]

    configs = [
        (1.0, 0.0, "100/0"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"),
        (0.40, 0.60, "40/60"),
        (0.0, 1.0, "0/100"),
    ]

    results = {}

    for split_name, selection, heldout in [("A→B", split_A, split_B), ("B→A", split_B, split_A)]:
        print(f"\n  Split {split_name}: Select on {selection}, validate on {heldout}")

        # Find best config on selection set
        selection_accs = {label: [] for _, _, label in configs}
        for ds in selection:
            if ds not in all_datasets:
                continue
            ds_results = all_datasets[ds]
            for acc, label in zip(ds_results["accs"], ds_results["labels"]):
                selection_accs[label].append(acc)

        print(f"\n  {'Config':<12} {'Selection avg':>15} {'Held-out avg':>15}")
        print(f"  {'-'*45}")

        heldout_accs = {label: [] for _, _, label in configs}
        for ds in heldout:
            if ds not in all_datasets:
                continue
            ds_results = all_datasets[ds]
            for acc, label in zip(ds_results["accs"], ds_results["labels"]):
                heldout_accs[label].append(acc)

        for _, _, label in configs:
            sel_avg = np.mean(selection_accs[label]) if selection_accs[label] else 0
            hld_avg = np.mean(heldout_accs[label]) if heldout_accs[label] else 0
            marker = " ← best on selection" if sel_avg == max(np.mean(selection_accs[l]) for _, _, l in configs if selection_accs[l]) else ""
            print(f"  {label:<12} {sel_avg*100:>14.2f}% {hld_avg*100:>14.2f}%{marker}")

        results[split_name] = {
            "selection": {l: float(np.mean(selection_accs[l])) for _, _, l in configs if selection_accs[l]},
            "heldout": {l: float(np.mean(heldout_accs[l])) for _, _, l in configs if heldout_accs[l]},
        }

    return results


def run_template_sensitivity(args, model, preprocess, tokenizer, device):
    """Experiment 2: Template set sensitivity."""
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 2: Template Sensitivity")
    print(f"{'='*70}")

    template_sets = [
        ("7 templates", SEVEN_TEMPLATES),
        ("20 templates", TWENTY_TEMPLATES),
        ("40 templates", IMAGENET_TEMPLATES[:40]),
        ("80 templates", IMAGENET_TEMPLATES),
    ]

    all_results = {}

    for ds_name in args.datasets:
        print(f"\n  --- {ds_name} ---")
        args.dataset = ds_name
        try:
            task_spec = build_task_spec(args)
            task_runner = build_task_runner(args, task_spec)
            class_names = task_spec.class_names

            descriptions = load_descriptions(ds_name, args.llm, args.output_dir,
                                              task_spec=task_spec, llm_provider=args.llm_provider)
            if descriptions is None:
                print(f"  Could not load or generate descriptions, skipping")
                continue

            ds_results = []
            for tmpl_name, templates in template_sets:
                M = len(templates)

                # Templates only
                acc_tmpl = evaluate_config(
                    task_runner, task_spec, class_names, descriptions,
                    templates, alpha=1.0, beta=0.0, device=device
                )

                # CuPL+e (uniform)
                acc_cupl = evaluate_config(
                    task_runner, task_spec, class_names, descriptions,
                    templates, alpha=M/(M+10), beta=10/(M+10), device=device
                )

                # NETRA 55/45
                acc_netra = evaluate_config(
                    task_runner, task_spec, class_names, descriptions,
                    templates, alpha=0.55, beta=0.45, device=device
                )

                delta_cupl = acc_cupl - acc_tmpl
                delta_netra = acc_netra - acc_tmpl

                ds_results.append({
                    "templates": tmpl_name,
                    "M": M,
                    "acc_templates": float(acc_tmpl),
                    "acc_cupl_e": float(acc_cupl),
                    "acc_netra": float(acc_netra),
                    "delta_cupl": float(delta_cupl),
                    "delta_netra": float(delta_netra),
                })

            print(f"  {'Templates':<16} {'Tmpl only':>10} {'CuPL+e':>10} {'NETRA':>10} {'Δ CuPL':>10} {'Δ NETRA':>10}")
            print(f"  {'-'*68}")
            for r in ds_results:
                print(f"  {r['templates']:<16} {r['acc_templates']*100:>9.2f}% "
                      f"{r['acc_cupl_e']*100:>9.2f}% {r['acc_netra']*100:>9.2f}% "
                      f"{r['delta_cupl']*100:>+9.2f}% {r['delta_netra']*100:>+9.2f}%")

            all_results[ds_name] = ds_results

            # Clear cache for next dataset
            if hasattr(task_runner, '_cached_image_features'):
                del task_runner._cached_image_features
                del task_runner._cached_labels

        except Exception as e:
            logger.warning(f"  {ds_name} failed: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Robustness experiments")
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar10", "cifar100", "flowers102", "dtd", "food101",
                                 "oxford_pets", "caltech101", "fgvc_aircraft", "eurosat", "country211"])
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--experiments", nargs="+", default=["heldout", "templates"],
                        choices=["heldout", "templates"])
    # Compatibility args
    parser.add_argument("--annotation-dir", type=str, default=None)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str, default=None)
    parser.add_argument("--gdino-checkpoint", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import open_clip
    model_name = args.clip_model.replace('/', '-')
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai', device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    all_results = {}

    # ── Held-out experiment needs per-dataset sweep results ──
    if "heldout" in args.experiments:
        print(f"\n  Computing per-dataset accuracy at each config...")
        configs = [
            (1.0, 0.0, "100/0"),
            (0.70, 0.30, "70/30"),
            (0.55, 0.45, "55/45"),
            (0.40, 0.60, "40/60"),
            (0.0, 1.0, "0/100"),
        ]

        dataset_configs = {}
        for ds_name in args.datasets:
            args.dataset = ds_name
            try:
                task_spec = build_task_spec(args)
                task_runner = build_task_runner(args, task_spec)
                class_names = task_spec.class_names
                descriptions = load_descriptions(ds_name, args.llm, args.output_dir,
                                                  task_spec=task_spec, llm_provider=args.llm_provider)
                if descriptions is None:
                    print(f"  {ds_name}: could not load or generate descriptions, skipping")
                    continue

                accs = []
                labels = []
                for alpha, beta, label in configs:
                    acc = evaluate_config(
                        task_runner, task_spec, class_names, descriptions,
                        IMAGENET_TEMPLATES, alpha, beta, args.device
                    )
                    accs.append(acc)
                    labels.append(label)
                    print(f"  {ds_name} @ {label}: {acc*100:.2f}%")

                dataset_configs[ds_name] = {"accs": accs, "labels": labels}

                if hasattr(task_runner, '_cached_image_features'):
                    del task_runner._cached_image_features
                    del task_runner._cached_labels

            except Exception as e:
                logger.warning(f"  {ds_name} failed: {e}")

        heldout_results = run_heldout_experiment(args, dataset_configs)
        all_results["heldout"] = heldout_results

    # ── Template sensitivity ──
    if "templates" in args.experiments:
        template_results = run_template_sensitivity(args, model, preprocess, tokenizer, args.device)
        all_results["templates"] = template_results

    # Save
    out_path = Path(args.output_dir) / "robustness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
