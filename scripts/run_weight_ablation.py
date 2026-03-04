#!/usr/bin/env python3
"""Weight ablation: test different base/description weight ratios.

Generates descriptions ONCE, then tests many weight ratios using
cached image features. Very fast after first eval.

Example:
    python scripts/run_weight_ablation.py --task classification \
        --dataset cifar100 --clip-model ViT-L/14 --val-size 10000
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec
from visprompt.baselines import IMAGENET_TEMPLATES

logger = logging.getLogger(__name__)


def generate_descriptions(task_spec, llm_model, llm_provider):
    """Generate CuPL-style descriptions for all classes (one LLM call)."""
    from visprompt.utils.llm import CostTracker, LLMClient

    cost_tracker = CostTracker()
    llm = LLMClient(model=llm_model, provider=llm_provider, cost_tracker=cost_tracker)

    all_descriptions = {}
    batch_size = 10

    for i in range(0, len(task_spec.class_names), batch_size):
        batch = task_spec.class_names[i:i + batch_size]
        prompt = (
            f"Generate 10-15 short visual descriptions for each class below.\n"
            f"Format each as: \"a {{class_name}}, {{short visual description}}\"\n"
            f"Keep descriptions under 15 words. Focus on shape, color, size, texture, habitat.\n\n"
            f"IMPORTANT: Make each description DIFFERENT — vary the visual angle:\n"
            f"  - Overall shape and silhouette\n"
            f"  - Dominant color/pattern\n"
            f"  - Typical context/setting/background\n"
            f"  - Key distinguishing feature vs similar classes\n\n"
            f"Classes: {', '.join(batch)}\n\n"
            f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
        )

        try:
            result = llm.call_json(
                prompt=prompt,
                system="Generate visual descriptions for CLIP zero-shot classification.",
                agent_name="weight_ablation",
            )
            for cls in batch:
                descs = result.get(cls, result.get(cls.replace("_", " "), []))
                if isinstance(descs, list) and descs:
                    # Ensure class name is in each description
                    cleaned = []
                    for d in descs:
                        cls_display = cls.replace("_", " ")
                        if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                            d = f"a {cls_display}, {d}"
                        cleaned.append(d)
                    all_descriptions[cls] = cleaned
                else:
                    all_descriptions[cls] = [f"a {cls.replace('_', ' ')} in a typical setting"]
        except Exception as e:
            logger.warning(f"Description batch failed: {e}")
            for cls in batch:
                all_descriptions[cls] = [f"a {cls.replace('_', ' ')} in a typical setting"]

    logger.info(f"Generated descriptions for {len(all_descriptions)} classes "
                f"(cost: {cost_tracker.summary()})")
    return all_descriptions, cost_tracker.summary()


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
            # Templates only
            all_prompts = base_prompts
            all_weights = [1.0 / n_base] * n_base

        prompts_per_class[cls_name] = {
            "prompts": all_prompts,
            "weights": all_weights,
        }

    return {
        "type": "classification",
        "prompts_per_class": prompts_per_class,
        "ensemble_method": "weighted_average",
    }


def main():
    parser = argparse.ArgumentParser(description="Weight ratio ablation")
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
    parser.add_argument("--output-dir", type=str, default="experiments/ablation")
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

    # ── Step 1: Generate descriptions once ────────────────────────────
    print("\n=== Generating descriptions (one-time LLM cost) ===")
    descriptions, desc_cost = generate_descriptions(
        task_spec, args.llm, args.llm_provider
    )

    n_descs = [len(v) for v in descriptions.values()]
    print(f"Descriptions per class: min={min(n_descs)}, max={max(n_descs)}, "
          f"avg={sum(n_descs)/len(n_descs):.1f}")

    # ── Step 2: Test weight ratios ────────────────────────────────────
    # Weight ratios to test: (base_weight, desc_weight)
    weight_configs = [
        (1.00, 0.00, "100/0 (templates only)"),
        (0.95, 0.05, "95/5"),
        (0.90, 0.10, "90/10"),
        (0.85, 0.15, "85/15"),
        (0.80, 0.20, "80/20"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45 (current default)"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.00, 1.00, "0/100 (descriptions only)"),
    ]

    results = []
    print(f"\n{'='*60}")
    print(f"  WEIGHT ABLATION — {args.clip_model} on {args.dataset}")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Accuracy':>10} {'Delta vs 100/0':>15}")
    print(f"{'-'*60}")

    baseline_acc = None

    for base_w, desc_w, label in weight_configs:
        prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions, base_w, desc_w
        )
        result = task_runner.evaluate(prompts, task_spec)
        acc = result.primary_metric

        if baseline_acc is None:
            baseline_acc = acc

        delta = acc - baseline_acc
        marker = " ← BEST" if acc >= max((r["accuracy"] for r in results), default=0) else ""
        print(f"{label:<30} {acc:>10.4f} {delta:>+14.4f}{marker}")

        results.append({
            "base_weight": base_w,
            "desc_weight": desc_w,
            "label": label,
            "accuracy": acc,
            "delta_vs_baseline": delta,
            "worst_5": result.worst_classes(5) if hasattr(result, 'worst_classes') else [],
        })

    print(f"{'='*60}")

    best = max(results, key=lambda r: r["accuracy"])
    print(f"\nBest config: {best['label']} → {best['accuracy']:.4f}")
    print(f"Templates-only baseline: {baseline_acc:.4f}")
    print(f"Improvement: {best['accuracy'] - baseline_acc:+.4f}")

    # ── Step 3: Temperature search at best weight ─────────────────────
    print(f"\n{'='*60}")
    print(f"  TEMPERATURE SEARCH — at {best['label']}")
    print(f"{'='*60}")
    print(f"{'Temperature':<20} {'Accuracy':>10} {'Delta vs best':>15}")
    print(f"{'-'*60}")

    best_weight_prompts = build_prompts_with_weights(
        task_spec.class_names, descriptions,
        best["base_weight"], best["desc_weight"]
    )

    temp_results = []
    for temp in [None, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
        result = task_runner.evaluate(best_weight_prompts, task_spec, temperature=temp)
        acc = result.primary_metric
        delta = acc - best["accuracy"]
        label = f"T={temp}" if temp is not None else "T=default (CLIP)"
        marker = " ← BEST" if acc >= max((r["accuracy"] for r in temp_results), default=0) else ""
        print(f"{label:<20} {acc:>10.4f} {delta:>+14.4f}{marker}")
        temp_results.append({
            "temperature": temp,
            "label": label,
            "accuracy": acc,
            "delta_vs_no_temp": delta,
        })

    best_temp = max(temp_results, key=lambda r: r["accuracy"])
    print(f"{'='*60}")
    print(f"\nBest temperature: {best_temp['label']} → {best_temp['accuracy']:.4f}")
    print(f"Combined improvement over templates-only: "
          f"{best_temp['accuracy'] - baseline_acc:+.4f}")

    # ── Step 4: Also test templates-only + best temperature ───────────
    print(f"\n--- Templates-only + best temperature ---")
    if best_temp["temperature"] is not None:
        templates_only_prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions, 1.0, 0.0
        )
        result = task_runner.evaluate(
            templates_only_prompts, task_spec,
            temperature=best_temp["temperature"]
        )
        print(f"Templates-only + {best_temp['label']}: {result.primary_metric:.4f}")
        print(f"→ Temperature alone contributes: "
              f"{result.primary_metric - baseline_acc:+.4f}")
        print(f"→ Descriptions contribute: "
              f"{best_temp['accuracy'] - result.primary_metric:+.4f}")
    else:
        print(f"Best temperature is default — no isolated test needed")

    # ── Save results ──────────────────────────────────────────────────
    output_path = Path(args.output_dir) / f"weight_ablation_{args.clip_model.replace('/', '_')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "clip_model": args.clip_model,
            "dataset": args.dataset,
            "val_size": args.val_size,
            "description_cost": desc_cost,
            "weight_results": results,
            "best_weight_config": best,
            "temperature_results": temp_results,
            "best_temperature_config": best_temp,
            "final_best_accuracy": best_temp["accuracy"],
            "improvement_over_baseline": best_temp["accuracy"] - baseline_acc,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
