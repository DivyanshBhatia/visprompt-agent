#!/usr/bin/env python3
"""Run all baseline methods for comparison (Table 2 in the paper).

Examples:
    python scripts/run_baselines.py --task classification --dataset cifar100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec
from visprompt.baselines import BaselineRunner


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--task", choices=["classification", "segmentation", "detection"], required=True)
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--output-dir", type=str, default="experiments/baselines")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    runner = BaselineRunner(task_runner=task_runner, task_spec=task_spec)
    results = runner.run_all(llm_model=args.llm, llm_provider=args.llm_provider)

    # Save results
    output_path = Path(args.output_dir) / f"{args.dataset}_baselines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for name, result in results.items():
        serializable[name] = {
            "primary_metric": result.primary_metric,
            "metric_name": result.primary_metric_name,
            "per_class_stats": result.class_accuracy_stats(),
            "worst_classes": result.worst_classes(5),
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nBaseline results saved to: {output_path}")


if __name__ == "__main__":
    main()
