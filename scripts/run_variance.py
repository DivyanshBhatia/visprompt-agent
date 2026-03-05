#!/usr/bin/env python3
"""Run multiple trials to measure description variance.

Generates fresh descriptions each trial, evaluates at 70/30,
reports mean ± std.

Example:
    python scripts/run_variance.py --task classification \
        --dataset cifar100 --clip-model ViT-L/14 --val-size 10000 --n-trials 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec
from scripts.run_weight_ablation import generate_descriptions, build_prompts_with_weights
from visprompt.baselines import IMAGENET_TEMPLATES

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Variance measurement over multiple trials")
    parser.add_argument("--task", choices=["classification"], default="classification")
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
    parser.add_argument("--llm-api-key", type=str)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--base-weight", type=float, default=0.7)
    parser.add_argument("--desc-weight", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="experiments/variance")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args for build_task_spec compatibility
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

    # ── Run templates-only baseline once ──────────────────────────────
    print(f"\n{'='*65}")
    print(f"  VARIANCE TEST — {args.n_trials} trials, {args.clip_model} on {args.dataset}")
    print(f"  Weight config: {args.base_weight:.0%} base / {args.desc_weight:.0%} desc")
    print(f"{'='*65}")

    print("\nRunning templates-only baseline (fixed, no variance)...")
    baseline_prompts = build_prompts_with_weights(
        task_spec.class_names, {}, 1.0, 0.0
    )
    baseline_result = task_runner.evaluate(baseline_prompts, task_spec)
    baseline_acc = baseline_result.primary_metric
    print(f"Templates-only baseline: {baseline_acc:.4f}\n")

    # ── Run N trials with fresh descriptions ──────────────────────────
    print(f"{'Trial':<8} {'Accuracy':>10} {'Δ vs base':>12} {'Desc cost':>12} {'Time':>8}")
    print(f"{'-'*55}")

    trials = []
    total_cost = 0.0

    for trial in range(args.n_trials):
        t0 = time.time()

        # Generate fresh descriptions (no caching between trials)
        descriptions, desc_cost = generate_descriptions(
            task_spec, args.llm, args.llm_provider
        )
        cost = desc_cost.get("total_cost_usd", 0) if isinstance(desc_cost, dict) else 0

        # Build prompts with group-normalized weights
        prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions,
            args.base_weight, args.desc_weight
        )

        # Evaluate (image features cached after first trial)
        result = task_runner.evaluate(prompts, task_spec)
        acc = result.primary_metric
        elapsed = time.time() - t0
        delta = acc - baseline_acc
        total_cost += cost

        # Per-class for variance analysis
        per_class = result.per_class_metrics if hasattr(result, 'per_class_metrics') else {}

        trials.append({
            "trial": trial,
            "accuracy": acc,
            "delta_vs_baseline": delta,
            "cost_usd": cost,
            "duration_s": elapsed,
            "worst_5": result.worst_classes(5) if hasattr(result, 'worst_classes') else [],
            "per_class_metrics": per_class,
        })

        print(f"  {trial+1:<6} {acc:>10.4f} {delta:>+11.4f} ${cost:>10.4f} {elapsed:>7.0f}s")

    # ── Summary statistics ────────────────────────────────────────────
    accuracies = [t["accuracy"] for t in trials]
    import statistics
    mean_acc = statistics.mean(accuracies)
    std_acc = statistics.stdev(accuracies) if len(accuracies) > 1 else 0
    min_acc = min(accuracies)
    max_acc = max(accuracies)
    median_acc = statistics.median(accuracies)

    print(f"\n{'='*65}")
    print(f"  RESULTS — {args.n_trials} trials")
    print(f"{'='*65}")
    print(f"  Templates-only baseline:  {baseline_acc:.4f}")
    print(f"  Mean accuracy:            {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Median accuracy:          {median_acc:.4f}")
    print(f"  Range:                    [{min_acc:.4f}, {max_acc:.4f}]")
    print(f"  Mean Δ vs baseline:       {mean_acc - baseline_acc:+.4f}")
    print(f"  Min Δ vs baseline:        {min_acc - baseline_acc:+.4f}")
    print(f"  All trials beat baseline: {'YES' if min_acc > baseline_acc else 'NO'}")
    print(f"  Total LLM cost:           ${total_cost:.4f}")
    print(f"{'='*65}")

    # Paper-ready format
    print(f"\n  For paper: {mean_acc:.2%} ± {std_acc:.2%} (n={args.n_trials})")
    print(f"            +{mean_acc - baseline_acc:.2%} over 80-template baseline")

    # ── Save results ──────────────────────────────────────────────────
    output_path = Path(args.output_dir) / f"variance_{args.clip_model.replace('/', '_')}_{args.n_trials}trials.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "clip_model": args.clip_model,
            "dataset": args.dataset,
            "val_size": args.val_size,
            "n_trials": args.n_trials,
            "base_weight": args.base_weight,
            "desc_weight": args.desc_weight,
            "baseline_accuracy": baseline_acc,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "median_accuracy": median_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "improvement_mean": mean_acc - baseline_acc,
            "improvement_min": min_acc - baseline_acc,
            "all_beat_baseline": min_acc > baseline_acc,
            "total_cost_usd": total_cost,
            "trials": trials,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
