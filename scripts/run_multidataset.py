#!/usr/bin/env python3
"""Run full benchmark across multiple datasets.

For each dataset: baselines + weight ablation + variance measurement.

Example:
    python scripts/run_multidataset.py --clip-model ViT-L/14 \
        --datasets cifar100 flowers102 dtd eurosat
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


DATASET_CONFIGS = {
    "cifar10": {"val_size": 10000, "domain": "10 basic categories"},
    "cifar100": {"val_size": 10000, "domain": "natural objects"},
    "flowers102": {"val_size": 6149, "domain": "fine-grained flowers"},
    "dtd": {"val_size": 1880, "domain": "textures"},
    "eurosat": {"val_size": 5000, "domain": "satellite imagery"},
    "food101": {"val_size": 10000, "domain": "food categories"},
    "fgvc_aircraft": {"val_size": 3333, "domain": "fine-grained aircraft"},
}


def run_command(cmd, description):
    """Run a command and capture output."""
    print(f"\n{'='*65}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*65}\n")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset benchmark")
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar10", "cifar100", "flowers102", "dtd", "eurosat", "food101", "fgvc_aircraft"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--n-variance-trials", type=int, default=3,
                        help="Number of variance trials per dataset")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-variance", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    base_args = [
        sys.executable,
    ]
    common_args = [
        "--clip-model", args.clip_model,
        "--device", args.device,
    ]
    llm_args = [
        "--llm", args.llm,
        "--llm-provider", args.llm_provider,
    ]

    results_summary = {}

    for dataset in args.datasets:
        config = DATASET_CONFIGS[dataset]
        val_size = str(config["val_size"])
        ds_args = ["--task", "classification", "--dataset", dataset, "--val-size", val_size]

        print(f"\n{'#'*65}")
        print(f"#  DATASET: {dataset.upper()} ({config['domain']})")
        print(f"#  Val size: {val_size}")
        print(f"{'#'*65}")

        # Step 1: Baselines
        if not args.skip_baselines:
            run_command(
                base_args + ["scripts/run_baselines.py"] + ds_args + common_args + llm_args,
                f"BASELINES — {dataset}",
            )

        # Step 2: Weight ablation + temperature
        if not args.skip_ablation:
            run_command(
                base_args + ["scripts/run_weight_ablation.py"] + ds_args + common_args + llm_args,
                f"WEIGHT ABLATION — {dataset}",
            )

        # Step 3: Variance measurement
        if not args.skip_variance:
            run_command(
                base_args + ["scripts/run_variance.py"] + ds_args + common_args + llm_args
                + ["--n-trials", str(args.n_variance_trials)],
                f"VARIANCE ({args.n_variance_trials} trials) — {dataset}",
            )

    print(f"\n{'='*65}")
    print(f"  ALL DATASETS COMPLETE")
    print(f"  Results saved in experiments/ directory")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
