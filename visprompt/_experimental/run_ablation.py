#!/usr/bin/env python3
"""Run ablation experiments for VisPromptAgent.

Produces Table 3 from the paper: agent ablation, iteration ablation,
unit test ablation, and VLM backbone ablation.

Examples:
    # Full ablation on CIFAR-100
    python scripts/run_ablation.py --task classification --dataset cifar100

    # Specific ablation only
    python scripts/run_ablation.py --task classification --dataset cifar100 \\
        --ablations agent iteration
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec
from visprompt.pipeline import AblationPipeline, VisPromptPipeline


def run_iteration_ablation(args, task_spec, task_runner) -> dict:
    """Test different iteration counts: 1, 2, 3, 5."""
    results = {}
    for n_iter in [1, 2, 3, 5]:
        logging.info(f"=== ITERATION ABLATION: {n_iter} iterations ===")
        pipeline = VisPromptPipeline(
            task_spec=task_spec,
            task_runner=task_runner,
            llm_model=args.llm,
            llm_provider=args.llm_provider,
            output_dir=f"{args.output_dir}/ablation/iter_{n_iter}",
        )
        result = pipeline.run(max_iterations=n_iter)
        results[f"iter_{n_iter}"] = result.to_dict()
    return results


def run_backbone_ablation(args, task_spec, task_runner) -> dict:
    """Test different VLM backbones."""
    results = {}
    backbones = [
        ("gpt-4o", "openai"),
        ("gpt-4o-mini", "openai"),
    ]
    # Add provider-specific models if keys available
    if args.anthropic_key:
        backbones.append(("claude-sonnet-4-20250514", "anthropic"))

    for model, provider in backbones:
        logging.info(f"=== BACKBONE ABLATION: {model} ===")
        try:
            pipeline = VisPromptPipeline(
                task_spec=task_spec,
                task_runner=task_runner,
                llm_model=model,
                llm_provider=provider,
                output_dir=f"{args.output_dir}/ablation/backbone_{model}",
            )
            result = pipeline.run(max_iterations=args.max_iter)
            results[model] = result.to_dict()
        except Exception as e:
            logging.warning(f"Backbone {model} failed: {e}")
            results[model] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Run VisPromptAgent ablation experiments")
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
    parser.add_argument("--anthropic-key", type=str)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=["agent", "iteration", "backbone", "all"],
        default=["all"],
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    all_results = {}
    ablations = args.ablations
    if "all" in ablations:
        ablations = ["agent", "iteration", "backbone"]

    # Agent ablation (full, no_analyst, no_critic, no_strategist, single_agent)
    if "agent" in ablations:
        logging.info("Starting agent ablation...")
        ablation = AblationPipeline(
            task_spec=task_spec,
            task_runner=task_runner,
            llm_model=args.llm,
            llm_provider=args.llm_provider,
            output_dir=f"{args.output_dir}/ablation",
        )
        agent_results = ablation.run_all(max_iterations=args.max_iter)
        all_results["agent_ablation"] = {
            k: v.to_dict() for k, v in agent_results.items()
        }

    # Iteration ablation
    if "iteration" in ablations:
        logging.info("Starting iteration ablation...")
        all_results["iteration_ablation"] = run_iteration_ablation(
            args, task_spec, task_runner
        )

    # Backbone ablation
    if "backbone" in ablations:
        logging.info("Starting backbone ablation...")
        all_results["backbone_ablation"] = run_backbone_ablation(
            args, task_spec, task_runner
        )

    # Save all results
    output_path = Path(args.output_dir) / "ablation" / "all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll ablation results saved to: {output_path}")


if __name__ == "__main__":
    main()
