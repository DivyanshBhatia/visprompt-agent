"""Prompt Executor agent: runs foundation models and produces rich diagnostics.

The Executor is intentionally simple — it translates the Planner's strategy
into concrete prompts, runs the foundation model, and returns structured
evaluation results. The intelligence is in other agents; the Executor just executes.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

import numpy as np

from visprompt.agents.base import AgentMessage, BaseAgent, TaskSpec
from visprompt.utils.metrics import EvalResult, MetricsComputer

logger = logging.getLogger(__name__)


class PromptExecutor(BaseAgent):
    """Executes prompts against foundation models and collects diagnostics.

    Task-agnostic: delegates actual model inference to a TaskRunner.
    """

    name = "prompt_executor"
    role_description = (
        "Execute the Planner's strategy against the foundation model, "
        "collect per-class metrics, confusion matrices, and confidence distributions."
    )

    def __init__(self, llm, task_runner=None):
        """
        Args:
            llm: LLM client (used minimally — mostly for prompt materialization).
            task_runner: A TaskRunner instance that handles model inference.
        """
        super().__init__(llm)
        self.task_runner = task_runner
        self._description_cache: dict[str, list[str]] = {}

    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        logger.info(f"[{self.name}] Executing prompts (iter {iteration})")

        planner_msg = upstream_messages.get("prompt_planner")
        if not planner_msg:
            raise ValueError("Executor requires Planner output")

        strategy = planner_msg.content

        # ── 1. Materialize prompts from strategy ─────────────────────────
        prompts = self._materialize_prompts(task_spec, strategy)

        # ── 2. Run foundation model via TaskRunner ────────────────────────
        if self.task_runner is None:
            raise ValueError(
                "No TaskRunner provided. Set executor.task_runner to a "
                "ClassificationRunner, SegmentationRunner, or DetectionRunner."
            )

        eval_result = self.task_runner.evaluate(prompts, task_spec)

        # ── 3. Package rich diagnostics ───────────────────────────────────
        diagnostics = self._build_diagnostics(eval_result, task_spec)

        content = {
            "eval_result": {
                "primary_metric": eval_result.primary_metric,
                "primary_metric_name": eval_result.primary_metric_name,
                "per_class_metrics": eval_result.per_class_metrics,
                "class_stats": eval_result.class_accuracy_stats(),
                "worst_classes": eval_result.worst_classes(10),
                "confusion_pairs": eval_result.confusion_pairs(10),
            },
            "diagnostics": diagnostics,
            "prompts_used": prompts,
            "strategy_name": strategy.get("strategy_name", ""),
        }

        logger.info(
            f"[{self.name}] {eval_result.primary_metric_name}: "
            f"{eval_result.primary_metric:.4f}"
        )

        return AgentMessage(
            sender=self.name,
            content=content,
            iteration=iteration,
            raw_text=json.dumps(content, indent=2, default=str),
        )

    def _materialize_prompts(
        self, task_spec: TaskSpec, strategy: dict
    ) -> dict[str, Any]:
        """Convert Planner's strategy into concrete prompts for the model.

        Returns a dict that the TaskRunner can consume directly.
        """
        if task_spec.task_type == "classification":
            return self._materialize_classification_prompts(task_spec, strategy)
        elif task_spec.task_type == "segmentation":
            return self._materialize_segmentation_prompts(task_spec, strategy)
        elif task_spec.task_type == "detection":
            return self._materialize_detection_prompts(task_spec, strategy)
        else:
            raise ValueError(f"Unknown task type: {task_spec.task_type}")

    def _materialize_classification_prompts(
        self, task_spec: TaskSpec, strategy: dict
    ) -> dict[str, Any]:
        """Build per-class prompt lists from the strategy.

        Hybrid approach for maximum accuracy:
        1. 80-template ensemble as stable base (low per-prompt weight)
        2. LLM-generated CuPL-style visual descriptions (high per-prompt weight)
        3. Class-specific discriminative prompts for hardest classes (highest weight)
        """
        from visprompt.baselines import IMAGENET_TEMPLATES

        prompts_per_class = {}

        # ── Extract strategy fields ──────────────────────────────────────
        description_prompt = strategy.get("description_prompt", "")
        class_specific = strategy.get("class_specific_prompts", {})
        base_weight = strategy.get("base_weight", 0.3)
        description_weight = strategy.get("description_weight", 0.7)

        # ── Generate per-class descriptions via LLM (cached) ─────────────
        class_descriptions = {}
        if description_prompt:
            # Only generate for classes not already cached
            uncached = [c for c in task_spec.class_names
                        if c not in self._description_cache]
            if uncached:
                new_descs = self._generate_class_descriptions(
                    task_spec, description_prompt, class_names=uncached
                )
                self._description_cache.update(new_descs)
            class_descriptions = {
                c: self._description_cache[c]
                for c in task_spec.class_names
                if c in self._description_cache
            }

        # ── Assemble per-class prompts ───────────────────────────────────
        for cls_name in task_spec.class_names:
            cls_prompts = []
            cls_weights = []

            # 1. Full 80-template ensemble as stable base
            for template in IMAGENET_TEMPLATES:
                filled = template.format(cls_name)
                cls_prompts.append(filled)
                cls_weights.append(base_weight)

            # 2. LLM-generated CuPL-style descriptions (high weight each)
            if cls_name in class_descriptions:
                for desc in class_descriptions[cls_name]:
                    cls_prompts.append(desc)
                    cls_weights.append(description_weight)

            # 3. Class-specific overrides for hardest classes (highest weight)
            if cls_name in class_specific:
                for p in class_specific[cls_name].get("prompts", []):
                    filled_p = p
                    if "{class}" in filled_p:
                        filled_p = filled_p.replace("{class}", cls_name)
                    elif "{}" in filled_p:
                        filled_p = filled_p.replace("{}", cls_name, 1)
                    cls_prompts.append(filled_p)
                    cls_weights.append(description_weight * 1.5)

            if not cls_prompts:
                cls_prompts = [f"a photo of a {cls_name}"]
                cls_weights = [1.0]

            prompts_per_class[cls_name] = {
                "prompts": cls_prompts,
                "weights": cls_weights,
            }

        logger.info(f"[{self.name}] Sample prompts after materialization:")
        self._log_sample_prompts(prompts_per_class, n=5)

        return {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": strategy.get("ensemble_method", "weighted_average"),
        }

    def _generate_class_descriptions(
        self, task_spec: TaskSpec, description_prompt: str,
        class_names: list[str] | None = None,
    ) -> dict[str, list[str]]:
        """Call LLM to generate CuPL-style visual descriptions for classes.

        Generates short, CLIP-friendly descriptions (not full sentences).
        Format: "a {class}, which is {visual description}"
        """
        all_descriptions = {}
        target_classes = class_names or task_spec.class_names
        batch_size = 25  # Process in batches to stay within token limits

        for i in range(0, len(target_classes), batch_size):
            batch = target_classes[i:i + batch_size]

            user_prompt = (
                f"Generate 7-10 short visual descriptions for each class below.\n"
                f"Format each as: \"a {{class_name}}, {{short visual description}}\"\n"
                f"Keep descriptions under 15 words. Focus on shape, color, size, texture, habitat.\n\n"
                f"Examples:\n"
                f'  "a camel, a large tan animal with humps on its back in a desert"\n'
                f'  "a ray, a flat diamond-shaped fish with wing-like fins in blue water"\n'
                f'  "a beetle, a small dark oval insect with a shiny hard shell"\n\n'
                f"Dataset: {task_spec.dataset_name}\n"
                f"Resolution: {task_spec.image_resolution or 'unknown'}\n"
                f"Classes: {', '.join(batch)}\n\n"
                f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
            )

            try:
                result = self.llm.call_json(
                    prompt=user_prompt,
                    system=description_prompt,
                    agent_name="description_generator",
                )

                for cls in batch:
                    descs = result.get(cls, result.get(cls.replace("_", " "), []))
                    if isinstance(descs, list) and descs:
                        # Ensure each description contains the class name
                        cleaned = []
                        for d in descs:
                            cls_display = cls.replace("_", " ")
                            if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                                d = f"a {cls_display}, {d}"
                            cleaned.append(d)
                        all_descriptions[cls] = cleaned
                    else:
                        all_descriptions[cls] = [
                            f"a {cls.replace('_', ' ')} in a typical setting"
                        ]

            except Exception as e:
                logger.warning(f"Description generation failed for batch {i}: {e}")
                for cls in batch:
                    all_descriptions[cls] = [
                        f"a {cls.replace('_', ' ')} in a typical setting"
                    ]

        logger.info(
            f"[{self.name}] Generated descriptions for "
            f"{len(all_descriptions)} classes"
        )
        return all_descriptions

    def _log_sample_prompts(self, prompts_per_class: dict, n: int = 3) -> None:
        """Log a few sample prompts for debugging."""
        for i, (cls, info) in enumerate(prompts_per_class.items()):
            if i >= n:
                break
            total = len(info['prompts'])
            # Show the first non-template prompt (i.e., descriptions)
            descs = [p for p in info['prompts'] if not p.startswith(('a bad photo', 'a photo of', 'a rendering', 'graffiti', 'a cropped', 'a tattoo', 'the embroidered', 'a bright', 'a dark', 'a drawing', 'the plastic', 'a close-up', 'a black and white', 'a painting', 'a pixelated', 'a sculpture', 'a jpeg', 'a blurry', 'a good', 'a doodle', 'the origami', 'a sketch', 'a origami', 'a low resolution', 'the toy', 'a rendition', 'a cartoon', 'art of', 'a embroidered', 'itap', 'a plushie', 'the cartoon', 'the plushie', 'a toy', 'a clear', 'a snapshot'))]
            sample = descs[:3] if descs else info['prompts'][:3]
            logger.info(
                f"  {cls}: {sample} "
                f"(total: {total} prompts, {len(descs)} descriptions)"
            )

    def _materialize_segmentation_prompts(
        self, task_spec: TaskSpec, strategy: dict
    ) -> dict[str, Any]:
        """Build segmentation prompt config from strategy."""
        return {
            "type": "segmentation",
            "point_strategy": strategy.get("point_strategy", {
                "initial_points": 1,
                "placement": "center",
                "negative_points": False,
            }),
            "box_strategy": strategy.get("box_strategy", {}),
            "text_prompts": strategy.get("text_prompts", {}),
            "multi_object_handling": strategy.get("multi_object_handling", "per_instance"),
        }

    def _materialize_detection_prompts(
        self, task_spec: TaskSpec, strategy: dict
    ) -> dict[str, Any]:
        """Build detection prompt config from strategy."""
        class_descriptions = {}
        levels = strategy.get("levels", [])
        class_specific = strategy.get("class_specific_descriptions", {})

        for cls_name in task_spec.class_names:
            descriptions = []
            for level in levels:
                target = level.get("target_classes", "all")
                if target != "all" and cls_name not in target:
                    continue
                template = level.get("template", "{class}")
                descriptions.append(template.replace("{class}", cls_name))

            if cls_name in class_specific:
                descriptions.extend(class_specific[cls_name].get("prompts", []))

            class_descriptions[cls_name] = descriptions or [cls_name]

        return {
            "type": "detection",
            "class_descriptions": class_descriptions,
            "confidence_threshold": strategy.get("confidence_threshold", 0.3),
            "nms_threshold": strategy.get("nms_threshold", 0.5),
        }

    def _build_diagnostics(
        self, eval_result: EvalResult, task_spec: TaskSpec
    ) -> dict[str, Any]:
        """Build rich diagnostic information for the Critic."""
        diagnostics = {
            "metric_summary": {
                "primary": eval_result.primary_metric,
                "name": eval_result.primary_metric_name,
            },
        }

        # Per-class stats
        stats = eval_result.class_accuracy_stats()
        if stats:
            diagnostics["class_distribution"] = stats

        # Confidence analysis
        if eval_result.confidence_scores is not None:
            scores = eval_result.confidence_scores
            if eval_result.predictions is not None and eval_result.ground_truth is not None:
                correct = eval_result.predictions == eval_result.ground_truth
                if correct.any():
                    diagnostics["confidence"] = {
                        "correct_mean": float(scores[correct].mean()),
                        "correct_std": float(scores[correct].std()),
                        "wrong_mean": float(scores[~correct].mean()) if (~correct).any() else 0.0,
                        "wrong_std": float(scores[~correct].std()) if (~correct).any() else 0.0,
                        "separation_ratio": (
                            float(scores[correct].mean() / max(scores[~correct].mean(), 1e-8))
                            if (~correct).any() else float("inf")
                        ),
                    }

        return diagnostics
