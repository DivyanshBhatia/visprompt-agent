#!/usr/bin/env python3
"""Run agent ablation: Full system vs removing each agent.

Uses the existing AblationPipeline to produce Table 3:
  - Full system (Analyst → Planner → Executor → Critic → Strategist)
  - w/o Dataset Analyst
  - w/o Quality Critic  
  - w/o Refinement Strategist (single iteration)
  - Single agent (no role decomposition)

Example:
    python scripts/run_agent_ablation.py --dataset flowers102 --clip-model ViT-L/14 --val-size 6149
    python scripts/run_agent_ablation.py --dataset dtd --clip-model ViT-L/14 --val-size 1880
    python scripts/run_agent_ablation.py --dataset cifar100 --clip-model ViT-L/14
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_spec, build_task_runner
from visprompt.pipeline import AblationPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Agent ablation study")
    parser.add_argument("--dataset", type=str, default="flowers102")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="experiments/agent_ablation")
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

    print(f"\n{'='*65}")
    print(f"  AGENT ABLATION — {args.dataset.upper()}")
    print(f"  LLM: {args.llm}, Model: {args.clip_model}")
    print(f"  Max iterations: {args.max_iter}")
    print(f"{'='*65}")

    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    # Compute text embeddings for Analyst
    text_embeddings = None
    if task_spec.task_type == "classification" and hasattr(task_runner, "get_text_embeddings"):
        print("Computing text embeddings for Analyst...")
        text_embeddings = task_runner.get_text_embeddings(task_spec.class_names)

    ablation = AblationPipeline(
        task_spec=task_spec,
        task_runner=task_runner,
        llm_model=args.llm,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key,
        output_dir=args.output_dir,
        text_embeddings=text_embeddings,
    )

    results = ablation.run_all(max_iterations=args.max_iter)

    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {}
    for name, result in results.items():
        summary[name] = {
            "final_metric": result.final_metric,
            "n_iterations": len(result.iterations),
            "cost": result.cost_summary,
            "per_iteration": [
                {"iteration": r.iteration, "metric": r.primary_metric}
                for r in result.iterations
            ],
        }

    with open(output_dir / f"agent_ablation_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}/agent_ablation_{args.dataset}.json")


if __name__ == "__main__":
    main()
