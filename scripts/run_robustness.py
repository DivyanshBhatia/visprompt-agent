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

    # Map short names to full API model strings
    API_NAMES = {
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    }
    api_model = API_NAMES.get(llm, llm)
    provider = "google" if "gemini" in llm else llm_provider

    print(f"  Generating descriptions for {dataset_name} with {api_model}...")
    from scripts.run_weight_ablation import generate_descriptions
    descriptions, cost = generate_descriptions(task_spec, api_model, provider)
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
                    templates, alpha, beta):
    """Evaluate a single alpha/beta config with given templates."""

    # Ensure model is loaded
    task_runner._ensure_model()
    task_runner.load_data()
    model = task_runner._clip_model
    tokenizer = task_runner._tokenizer

    M = len(templates)

    # Build prompts per class
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

    # Use the runner's evaluate method for consistency (logit-level + TTA)
    result = task_runner.evaluate(
        {"prompts_per_class": ppc},
        task_spec
    )
    return result.primary_metric


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
        print(f"  Available datasets: {list(all_datasets.keys())}")

        # Find best config on selection set
        selection_accs = {label: [] for _, _, label in configs}
        for ds in selection:
            if ds not in all_datasets:
                print(f"    {ds} not in results, skipping")
                continue
            ds_results = all_datasets[ds]
            for acc, label in zip(ds_results["accs"], ds_results["labels"]):
                selection_accs[label].append(acc)

        print(f"\n  {'Config':<12} {'Selection avg':>15} {'Held-out avg':>15}")
        print(f"  {'-'*45}")

        heldout_accs = {label: [] for _, _, label in configs}
        for ds in heldout:
            if ds not in all_datasets:
                print(f"    {ds} not in results, skipping")
                continue
            ds_results = all_datasets[ds]
            for acc, label in zip(ds_results["accs"], ds_results["labels"]):
                heldout_accs[label].append(acc)

        # Find best config on selection set
        best_sel_avg = -1
        best_sel_label = None
        for _, _, label in configs:
            if selection_accs[label]:
                avg = np.mean(selection_accs[label])
                if avg > best_sel_avg:
                    best_sel_avg = avg
                    best_sel_label = label

        for _, _, label in configs:
            sel_avg = np.mean(selection_accs[label]) if selection_accs[label] else 0
            hld_avg = np.mean(heldout_accs[label]) if heldout_accs[label] else 0
            marker = " ← best on selection" if label == best_sel_label else ""
            print(f"  {label:<12} {sel_avg*100:>14.2f}% {hld_avg*100:>14.2f}%{marker}")

        results[split_name] = {
            "selection": {l: float(np.mean(selection_accs[l])) for _, _, l in configs if selection_accs[l]},
            "heldout": {l: float(np.mean(heldout_accs[l])) for _, _, l in configs if heldout_accs[l]},
        }

    return results


def run_template_sensitivity(args):
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
                    templates, alpha=1.0, beta=0.0
                )

                # CuPL+e (uniform)
                acc_cupl = evaluate_config(
                    task_runner, task_spec, class_names, descriptions,
                    templates, alpha=M/(M+10), beta=10/(M+10)
                )

                # NETRA 55/45
                acc_netra = evaluate_config(
                    task_runner, task_spec, class_names, descriptions,
                    templates, alpha=0.55, beta=0.45
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
                        IMAGENET_TEMPLATES, alpha, beta
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

        # ── Leave-One-Out validation (computed from same data) ──
        print(f"\n{'='*70}")
        print(f"  LEAVE-ONE-OUT VALIDATION")
        print(f"{'='*70}")
        print(f"  For each dataset: select best config on other 9, report held-out acc")
        print(f"\n  {'Held-out':<18} {'Best on 9':>10} {'55/45 on 9':>12} {'Best cfg':>10} {'Held-out acc':>13} {'55/45 acc':>10}")
        print(f"  {'-'*75}")

        ds_names = sorted(dataset_configs.keys())
        loo_results = []

        for held_out in ds_names:
            # Average accuracy across other 9 at each config
            train_sets = [d for d in ds_names if d != held_out]
            config_avgs = {}
            for cfg_idx, (_, _, label) in enumerate(configs):
                train_accs = [dataset_configs[d]["accs"][cfg_idx] for d in train_sets
                             if cfg_idx < len(dataset_configs[d]["accs"])]
                if train_accs:
                    config_avgs[label] = np.mean(train_accs)

            if not config_avgs:
                continue

            # Best config on train sets
            best_label = max(config_avgs, key=config_avgs.get)
            best_train_avg = config_avgs[best_label]
            train_55 = config_avgs.get("55/45", 0)

            # What does that config score on held-out?
            cfg_labels = dataset_configs[held_out]["labels"]
            cfg_accs = dataset_configs[held_out]["accs"]
            best_idx = cfg_labels.index(best_label) if best_label in cfg_labels else -1
            idx_55 = cfg_labels.index("55/45") if "55/45" in cfg_labels else -1

            held_acc = cfg_accs[best_idx] if best_idx >= 0 else 0
            held_55 = cfg_accs[idx_55] if idx_55 >= 0 else 0

            print(f"  {held_out:<18} {best_train_avg*100:>9.2f}% {train_55*100:>11.2f}% {best_label:>10} {held_acc*100:>12.2f}% {held_55*100:>9.2f}%")

            loo_results.append({
                "held_out": held_out,
                "best_config_on_9": best_label,
                "best_train_avg": float(best_train_avg),
                "train_55_avg": float(train_55),
                "held_out_acc_at_best": float(held_acc),
                "held_out_acc_at_55": float(held_55),
                "gap": float(held_acc - held_55),
            })

        # Summary
        gaps = [r["gap"] for r in loo_results]
        best_configs = [r["best_config_on_9"] for r in loo_results]
        from collections import Counter
        config_counts = Counter(best_configs)

        print(f"\n  LOO Summary:")
        print(f"  Best config distribution: {dict(config_counts)}")
        print(f"  Mean gap (best vs 55/45): {np.mean(gaps)*100:+.2f}%")
        print(f"  Max gap:                  {np.max(gaps)*100:+.2f}%")
        print(f"  Datasets where 55/45 = best: {sum(1 for r in loo_results if r['best_config_on_9'] == '55/45')}/{len(loo_results)}")

        all_results["loo"] = loo_results

    # ── Template sensitivity ──
    if "templates" in args.experiments:
        template_results = run_template_sensitivity(args)
        all_results["templates"] = template_results

    # Save
    out_path = Path(args.output_dir) / "robustness_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
