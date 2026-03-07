#!/usr/bin/env python3
"""Measure accuracy as a function of number of descriptions per class.

Shows diminishing returns curve: how many LLM descriptions are needed?

Example:
    python scripts/run_desc_scaling.py --dataset flowers102 --clip-model ViT-L/14
    python scripts/run_desc_scaling.py --dataset dtd --clip-model ViT-L/14
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Description count scaling experiment")
    parser.add_argument("--dataset", type=str, default="flowers102",
                        choices=["cifar100", "flowers102", "dtd", "oxford_pets", "food101"])
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--base-weight", type=float, default=None,
                        help="Base weight (auto-selects best if not specified)")
    parser.add_argument("--output-dir", type=str, default="experiments/desc_scaling")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args for build_task_spec compatibility
    parser.add_argument("--config", type=str)
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

    from scripts.run import build_task_spec, build_task_runner
    from scripts.run_weight_ablation import generate_descriptions, build_prompts_with_weights
    from visprompt.baselines import IMAGENET_TEMPLATES

    # Best weights per dataset (from weight ablation results)
    BEST_WEIGHTS = {
        "cifar100": (0.7, 0.3),
        "flowers102": (0.0, 1.0),
        "dtd": (0.4, 0.6),
        "oxford_pets": (0.0, 1.0),
        "food101": (0.4, 0.6),
    }

    if args.base_weight is not None:
        base_w = args.base_weight
        desc_w = 1.0 - base_w
    else:
        base_w, desc_w = BEST_WEIGHTS.get(args.dataset, (0.55, 0.45))

    print(f"\n{'='*65}")
    print(f"  DESCRIPTION SCALING — {args.dataset.upper()}")
    print(f"  Weight: {base_w:.0%}/{desc_w:.0%}, Model: {args.clip_model}")
    print(f"{'='*65}")

    # Build task
    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    # Templates-only baseline
    print("\nRunning templates-only baseline...")
    baseline_prompts = build_prompts_with_weights(
        task_spec.class_names, {}, 1.0, 0.0
    )
    baseline_result = task_runner.evaluate(baseline_prompts, task_spec)
    baseline_acc = baseline_result.primary_metric
    print(f"  Templates-only: {baseline_acc:.4f}")

    # Generate a large set of descriptions (request more than usual)
    print(f"\nGenerating descriptions with {args.llm}...")
    descriptions, desc_cost = generate_descriptions(
        task_spec, args.llm, args.llm_provider
    )

    # Count how many descriptions we got per class
    desc_counts = [len(descriptions.get(cn, [])) for cn in task_spec.class_names]
    max_descs = max(desc_counts) if desc_counts else 0
    avg_descs = np.mean(desc_counts) if desc_counts else 0
    min_descs = min(desc_counts) if desc_counts else 0
    print(f"  Descriptions per class: min={min_descs}, avg={avg_descs:.1f}, max={max_descs}")

    # Test accuracy at different description counts: 1, 2, 3, 4, 5, 6, 8, 10, 15, all
    test_counts = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, max_descs]
    test_counts = sorted(set(c for c in test_counts if c <= max_descs))

    print(f"\n{'N_desc':>8} {'Accuracy':>10} {'Δ vs base':>12} {'Δ vs prev':>12}")
    print(f"{'-'*45}")

    results = []
    prev_acc = baseline_acc

    for n_desc in test_counts:
        # Truncate descriptions to n_desc per class
        truncated = {}
        for cn in task_spec.class_names:
            all_descs = descriptions.get(cn, [])
            truncated[cn] = all_descs[:n_desc]

        # Build prompts with truncated descriptions
        prompts = build_prompts_with_weights(
            task_spec.class_names, truncated, base_w, desc_w
        )
        result = task_runner.evaluate(prompts, task_spec)
        acc = result.primary_metric

        delta_base = acc - baseline_acc
        delta_prev = acc - prev_acc

        print(f"  {n_desc:>6} {acc:>10.4f} {delta_base:>+11.4f} {delta_prev:>+11.4f}")

        results.append({
            "n_descriptions": n_desc,
            "accuracy": float(acc),
            "delta_vs_baseline": float(delta_base),
            "delta_vs_previous": float(delta_prev),
        })
        prev_acc = acc

    # Save results
    output = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "llm": args.llm,
        "base_weight": base_w,
        "desc_weight": desc_w,
        "baseline_accuracy": float(baseline_acc),
        "max_descriptions_per_class": max_descs,
        "avg_descriptions_per_class": float(avg_descs),
        "scaling_results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"desc_scaling_{args.dataset}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'='*45}")
    print(f"  Baseline (0 desc): {baseline_acc:.4f}")
    if results:
        print(f"  Best ({results[-1]['n_descriptions']} desc): {results[-1]['accuracy']:.4f}")
        print(f"  Gain: {results[-1]['delta_vs_baseline']:+.4f}")
        # Find diminishing returns point (< 0.1% marginal gain)
        for i, r in enumerate(results):
            if i > 0 and r["delta_vs_previous"] < 0.001:
                print(f"  Diminishing returns at: {results[i-1]['n_descriptions']} descriptions")
                break


if __name__ == "__main__":
    main()
