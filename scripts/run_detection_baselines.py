#!/usr/bin/env python3
"""Run detection baselines: single-name vs LLM-description prompts.

Example:
    python scripts/run_detection_baselines.py \
        --dataset coco \
        --data-dir ./data/coco/val2017 \
        --annotation-file ./data/coco/annotations/instances_val2017.json \
        --max-det-images 500
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec

logger = logging.getLogger(__name__)


def generate_detection_descriptions(task_spec, llm_model, llm_provider):
    """Generate visual descriptions for detection classes."""
    from visprompt.utils.llm import CostTracker, LLMClient

    cost_tracker = CostTracker()
    llm = LLMClient(model=llm_model, provider=llm_provider, cost_tracker=cost_tracker)

    all_descriptions = {}
    batch_size = 10

    for i in range(0, len(task_spec.class_names), batch_size):
        batch = task_spec.class_names[i:i + batch_size]
        prompt = (
            f"Generate 5 short visual descriptions for detecting each object class below.\n"
            f"Each description should help an object detector identify the object.\n"
            f"Format: \"a {{class_name}}, {{visual description}}\"\n"
            f"Keep under 15 words. Focus on shape, color, typical context.\n\n"
            f"Classes: {', '.join(batch)}\n\n"
            f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
        )

        try:
            result = llm.call_json(
                prompt=prompt,
                system="Generate visual descriptions for object detection.",
                agent_name="detection_descriptions",
            )
            for cls in batch:
                descs = result.get(cls, result.get(cls.replace("_", " "), []))
                if isinstance(descs, list) and descs:
                    all_descriptions[cls] = descs
                else:
                    all_descriptions[cls] = [cls]
        except Exception as e:
            logger.warning(f"Description batch failed: {e}")
            for cls in batch:
                all_descriptions[cls] = [cls]

    return all_descriptions, cost_tracker.summary()


def main():
    parser = argparse.ArgumentParser(description="Detection baselines")
    parser.add_argument("--task", default="detection")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--data-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--annotation-file", type=str, required=True, help="COCO annotations JSON")
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--val-size", type=int)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--det-model", type=str, default="owlvit")
    parser.add_argument("--owlvit-model", type=str, default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--max-det-images", type=int, default=500,
                        help="Max images (500 for fast eval, None for full)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--output-dir", type=str, default="experiments/detection")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args for compatibility
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

    results = {}

    # ── Baseline 1: Class name only ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE 1: Class names only")
    print(f"{'='*60}")

    t0 = time.time()
    prompts_simple = {
        "class_descriptions": {cls: [cls] for cls in task_spec.class_names},
        "confidence_threshold": 0.1,
    }
    result_simple = task_runner.evaluate(prompts_simple, task_spec)
    print(f"  mAP@50: {result_simple.primary_metric:.4f} ({time.time()-t0:.0f}s)")
    results["class_name_only"] = {
        "mAP": result_simple.primary_metric,
        "per_class": result_simple.per_class_metrics,
    }

    # ── Baseline 2: "a photo of a {class}" ───────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE 2: 'a photo of a {{class}}'")
    print(f"{'='*60}")

    t0 = time.time()
    prompts_photo = {
        "class_descriptions": {
            cls: [f"a photo of a {cls}"] for cls in task_spec.class_names
        },
        "confidence_threshold": 0.1,
    }
    result_photo = task_runner.evaluate(prompts_photo, task_spec)
    print(f"  mAP@50: {result_photo.primary_metric:.4f} ({time.time()-t0:.0f}s)")
    results["photo_template"] = {
        "mAP": result_photo.primary_metric,
        "per_class": result_photo.per_class_metrics,
    }

    # ── Baseline 3: LLM descriptions ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BASELINE 3: LLM-generated descriptions")
    print(f"{'='*60}")

    descriptions, desc_cost = generate_detection_descriptions(
        task_spec, args.llm, args.llm_provider
    )
    print(f"  Generated descriptions for {len(descriptions)} classes "
          f"(cost: ${desc_cost.get('total_cost_usd', 0):.4f})")

    t0 = time.time()
    prompts_desc = {
        "class_descriptions": descriptions,
        "confidence_threshold": 0.1,
    }
    result_desc = task_runner.evaluate(prompts_desc, task_spec)
    print(f"  mAP@50: {result_desc.primary_metric:.4f} ({time.time()-t0:.0f}s)")
    results["llm_descriptions"] = {
        "mAP": result_desc.primary_metric,
        "per_class": result_desc.per_class_metrics,
        "cost": desc_cost,
    }

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DETECTION BASELINE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'mAP@50':>10}")
    print(f"{'-'*45}")
    for name, res in sorted(results.items(), key=lambda x: x[1]["mAP"]):
        print(f"  {name:<28} {res['mAP']:>10.4f}")
    print(f"{'='*60}")

    # Save
    output_path = Path(args.output_dir) / "detection_baselines.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
