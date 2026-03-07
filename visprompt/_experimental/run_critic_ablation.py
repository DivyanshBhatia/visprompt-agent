#!/usr/bin/env python3
"""Ablation: impact of critic-guided refinement on description quality.

Compares:
  1. No critic (single-shot generation) — current baseline
  2. Critic (generate → critic reviews → refine) — 1 iteration
  3. Multi-iteration critic — 2-3 iterations

The critic reviews descriptions for:
  - Visual specificity (are descriptions actually visual?)
  - Distinctiveness (do descriptions differentiate from similar classes?)
  - CLIP-alignment (will CLIP understand these descriptions?)

Example:
    python scripts/run_critic_ablation.py --dataset flowers102 --clip-model ViT-L/14 --val-size 6149
    python scripts/run_critic_ablation.py --dataset dtd --clip-model ViT-L/14 --val-size 1880
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


CRITIC_PROMPT = """You are a visual description critic for zero-shot image classification using CLIP.

Review these visual descriptions for the class "{class_name}" and identify problems:

Descriptions:
{descriptions}

{similar_classes_context}

Score each description 1-5 on:
1. **Visual specificity**: Does it describe something you can SEE in a photo? (not abstract/conceptual)
2. **Distinctiveness**: Would this description distinguish {class_name} from similar classes?
3. **CLIP-friendliness**: Is it a natural image caption that CLIP would match to the right image?

Then provide:
- Which descriptions to KEEP (good ones)
- Which to REMOVE (too vague, not visual, or confusing for CLIP)
- 5 NEW replacement descriptions that are more specific and distinctive

Return ONLY valid JSON:
{{
  "keep_indices": [0, 2, 5],
  "remove_indices": [1, 3, 4],
  "reasons": ["desc at index 1 is too abstract", ...],
  "new_descriptions": ["a {class_name}, ...", "a {class_name}, ...", ...]
}}"""


REFINE_PROMPT = """Generate improved visual descriptions for the class "{class_name}" for CLIP zero-shot classification.

The critic identified these issues with previous descriptions:
{critic_feedback}

Good descriptions that were kept:
{kept_descriptions}

{similar_classes_context}

Generate {n_new} NEW descriptions that:
1. Are visually specific (describe what you SEE in a photo)
2. Distinguish {class_name} from similar classes: {similar_classes}
3. Are natural image captions CLIP would understand
4. Each under 15 words
5. Cover different visual angles: shape, color, texture, setting, distinguishing features

Format: "a {class_name}, {{visual description}}"

Return ONLY valid JSON: {{"descriptions": ["desc1", "desc2", ...]}}"""


def get_similar_classes(class_name, all_class_names, n=5):
    """Get similar class names for distinctiveness prompting."""
    # Simple heuristic: classes with shared words or similar length
    cn_lower = class_name.lower().split()
    scored = []
    for other in all_class_names:
        if other == class_name:
            continue
        other_lower = other.lower().split()
        shared = len(set(cn_lower) & set(other_lower))
        scored.append((other, shared))
    scored.sort(key=lambda x: -x[1])
    return [s[0] for s in scored[:n]]


def generate_with_critic(
    class_names, llm, n_iterations=1, batch_size=10
):
    """Generate descriptions with critic-guided refinement.
    
    Pipeline per iteration:
      1. Generate initial descriptions (or use previous iteration's output)
      2. Critic reviews each class's descriptions
      3. Refine: keep good ones, replace bad ones
    
    Returns descriptions at each iteration for comparison.
    """
    import json as json_mod
    
    all_iterations = {}  # {iteration_num: {class_name: [descriptions]}}
    current_descriptions = {}
    
    # ── Iteration 0: Initial generation (no critic) ──────────────
    print(f"\n  Iteration 0: Initial generation...")
    t0 = time.time()
    
    for i in range(0, len(class_names), batch_size):
        batch = class_names[i:i + batch_size]
        prompt = (
            f"Generate 10 short visual descriptions for each class below.\n"
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
                agent_name="critic_gen_iter0",
            )
            for cls in batch:
                descs = result.get(cls, result.get(cls.replace("_", " "), []))
                if isinstance(descs, list) and descs:
                    cleaned = []
                    for d in descs:
                        cls_display = cls.replace("_", " ")
                        if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                            d = f"a {cls_display}, {d}"
                        cleaned.append(d)
                    current_descriptions[cls] = cleaned
                else:
                    current_descriptions[cls] = [f"a {cls.replace('_', ' ')} in a typical setting"]
        except Exception as e:
            logger.warning(f"Gen batch failed: {e}")
            for cls in batch:
                current_descriptions[cls] = [f"a {cls.replace('_', ' ')} in a typical setting"]
    
    elapsed = time.time() - t0
    avg_descs = np.mean([len(v) for v in current_descriptions.values()])
    print(f"    Generated {len(current_descriptions)} classes, avg {avg_descs:.1f} desc/class ({elapsed:.0f}s)")
    all_iterations[0] = {k: list(v) for k, v in current_descriptions.items()}
    
    # ── Iterations 1..N: Critic + Refine ──────────────────────────
    for iteration in range(1, n_iterations + 1):
        print(f"\n  Iteration {iteration}: Critic review + refinement...")
        t0 = time.time()
        refined_descriptions = {}
        
        for i in range(0, len(class_names), batch_size):
            batch = class_names[i:i + batch_size]
            
            for cls in batch:
                descs = current_descriptions.get(cls, [])
                if not descs:
                    refined_descriptions[cls] = [f"a {cls.replace('_', ' ')} in a typical setting"]
                    continue
                
                similar = get_similar_classes(cls, class_names)
                similar_context = f"Similar classes to distinguish from: {', '.join(similar)}" if similar else ""
                
                # ── Step 1: Critic reviews ────────────────────────
                desc_list = "\n".join(f"  [{j}] {d}" for j, d in enumerate(descs))
                critic_prompt = CRITIC_PROMPT.format(
                    class_name=cls.replace("_", " "),
                    descriptions=desc_list,
                    similar_classes_context=similar_context,
                )
                
                try:
                    critic_result = llm.call_json(
                        prompt=critic_prompt,
                        system="You are a strict critic evaluating visual descriptions for image classification.",
                        agent_name=f"critic_review_iter{iteration}",
                    )
                    
                    keep_indices = critic_result.get("keep_indices", list(range(len(descs))))
                    remove_indices = critic_result.get("remove_indices", [])
                    reasons = critic_result.get("reasons", [])
                    new_from_critic = critic_result.get("new_descriptions", [])
                    
                    # Keep good descriptions
                    kept = [descs[j] for j in keep_indices if j < len(descs)]
                    
                    # Add critic's new descriptions
                    for d in new_from_critic:
                        cls_display = cls.replace("_", " ")
                        if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                            d = f"a {cls_display}, {d}"
                        kept.append(d)
                    
                    # ── Step 2: Generate replacements if needed ───
                    n_needed = max(0, 10 - len(kept))
                    if n_needed > 0:
                        refine_prompt = REFINE_PROMPT.format(
                            class_name=cls.replace("_", " "),
                            critic_feedback="; ".join(reasons[:3]) if reasons else "descriptions were too generic",
                            kept_descriptions="\n".join(f"  - {d}" for d in kept[:5]),
                            similar_classes_context=similar_context,
                            n_new=n_needed,
                            similar_classes=", ".join(similar[:3]),
                        )
                        try:
                            refine_result = llm.call_json(
                                prompt=refine_prompt,
                                system="Generate improved visual descriptions.",
                                agent_name=f"critic_refine_iter{iteration}",
                            )
                            new_descs = refine_result.get("descriptions", [])
                            for d in new_descs:
                                cls_display = cls.replace("_", " ")
                                if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                                    d = f"a {cls_display}, {d}"
                                kept.append(d)
                        except Exception:
                            pass
                    
                    refined_descriptions[cls] = kept[:15]  # cap at 15
                    
                except Exception as e:
                    logger.warning(f"Critic failed for {cls}: {e}")
                    refined_descriptions[cls] = descs  # keep original
        
        elapsed = time.time() - t0
        avg_descs = np.mean([len(v) for v in refined_descriptions.values()])
        n_changed = sum(1 for cls in class_names 
                       if refined_descriptions.get(cls, []) != current_descriptions.get(cls, []))
        print(f"    Refined {n_changed}/{len(class_names)} classes, "
              f"avg {avg_descs:.1f} desc/class ({elapsed:.0f}s)")
        
        current_descriptions = refined_descriptions
        all_iterations[iteration] = {k: list(v) for k, v in current_descriptions.items()}
    
    return all_iterations


def main():
    parser = argparse.ArgumentParser(description="Critic ablation study")
    parser.add_argument("--dataset", type=str, default="flowers102")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--n-iterations", type=int, default=2,
                        help="Number of critic iterations (0=no critic)")
    parser.add_argument("--output-dir", type=str, default="experiments/critic_ablation")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args
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
    from scripts.run_weight_ablation import build_prompts_with_weights
    from visprompt.utils.llm import CostTracker, LLMClient

    # Best weights per dataset
    BEST_WEIGHTS = {
        "cifar100": (0.7, 0.3),
        "flowers102": (0.0, 1.0),
        "dtd": (0.4, 0.6),
        "oxford_pets": (0.0, 1.0),
        "food101": (0.4, 0.6),
    }
    base_w, desc_w = BEST_WEIGHTS.get(args.dataset, (0.55, 0.45))

    print(f"\n{'='*65}")
    print(f"  CRITIC ABLATION — {args.dataset.upper()}")
    print(f"  LLM: {args.llm}, Weight: {base_w:.0%}/{desc_w:.0%}")
    print(f"  Iterations: 0 (no critic) → {args.n_iterations}")
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

    # Generate descriptions with critic iterations
    cost_tracker = CostTracker()
    llm = LLMClient(
        model=args.llm, provider=args.llm_provider, cost_tracker=cost_tracker
    )

    all_iterations = generate_with_critic(
        task_spec.class_names, llm,
        n_iterations=args.n_iterations,
    )

    # Evaluate each iteration
    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"{'Iteration':<12} {'Method':<25} {'Accuracy':>10} {'Δ vs base':>12} {'Δ vs prev':>12}")
    print(f"{'-'*72}")

    results = []
    prev_acc = baseline_acc

    print(f"  {'—':<10} {'Templates only':<25} {baseline_acc:>10.4f} {'—':>12} {'—':>12}")

    for iteration in sorted(all_iterations.keys()):
        descriptions = all_iterations[iteration]
        method = "No critic" if iteration == 0 else f"Critic iter {iteration}"

        prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions, base_w, desc_w
        )
        result = task_runner.evaluate(prompts, task_spec)
        acc = result.primary_metric

        delta_base = acc - baseline_acc
        delta_prev = acc - prev_acc

        print(f"  {iteration:<10} {method:<25} {acc:>10.4f} {delta_base:>+11.4f} {delta_prev:>+11.4f}")

        results.append({
            "iteration": iteration,
            "method": method,
            "accuracy": float(acc),
            "delta_vs_baseline": float(delta_base),
            "delta_vs_previous": float(delta_prev),
            "n_descriptions_avg": float(np.mean([len(v) for v in descriptions.values()])),
        })
        prev_acc = acc

    # Save
    output = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "llm": args.llm,
        "base_weight": base_w,
        "desc_weight": desc_w,
        "baseline_accuracy": float(baseline_acc),
        "n_iterations": args.n_iterations,
        "cost": cost_tracker.summary(),
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"critic_ablation_{args.dataset}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print(f"Total cost: ${cost_tracker.summary()['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
