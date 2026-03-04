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
        """Build per-class prompt lists from the hierarchical strategy."""
        prompts_per_class = {}
        levels = strategy.get("levels", [])
        class_specific = strategy.get("class_specific_prompts", {})
        ensemble_weights = strategy.get("ensemble_weights", [1.0] * len(levels))

        for cls_name in task_spec.class_names:
            cls_prompts = []
            cls_weights = []

            for level, weight in zip(levels, ensemble_weights):
                target = level.get("target_classes", "all")
                # If target is a list, check if it contains actual class names
                # or category labels (like "animals") that won't match individual classes
                if target != "all" and isinstance(target, list):
                    # Check if any target matches actual class names
                    if not any(t in task_spec.class_names for t in target):
                        # These are category labels, not class names — apply to all
                        target = "all"
                    elif cls_name not in target:
                        continue
                elif target != "all":
                    continue

                for template in level.get("templates", []):
                    # Handle multiple placeholder styles the LLM might use
                    filled = template
                    if "{class}" in filled:
                        filled = filled.replace("{class}", cls_name)
                    elif "{}" in filled:
                        filled = filled.replace("{}", cls_name, 1)
                    elif cls_name not in filled.lower():
                        # Template has no placeholder — prepend class name
                        filled = f"a photo of a {cls_name}"
                    # Also handle {superclass} if hierarchy exists
                    if "{superclass}" in filled and task_spec.class_hierarchy:
                        for super_cls, sub_classes in task_spec.class_hierarchy.items():
                            if cls_name in sub_classes:
                                filled = filled.replace("{superclass}", super_cls)
                                break
                    cls_prompts.append(filled)
                    cls_weights.append(weight)

            # Add class-specific overrides
            if cls_name in class_specific:
                for p in class_specific[cls_name].get("prompts", []):
                    # Fill in placeholders in class-specific prompts too
                    filled_p = p
                    if "{class}" in filled_p:
                        filled_p = filled_p.replace("{class}", cls_name)
                    elif "{}" in filled_p:
                        filled_p = filled_p.replace("{}", cls_name, 1)
                    cls_prompts.append(filled_p)
                    cls_weights.append(max(ensemble_weights) if ensemble_weights else 1.0)

            prompts_per_class[cls_name] = {
                "prompts": cls_prompts if cls_prompts else [f"a photo of a {cls_name}"],
                "weights": cls_weights if cls_weights else [1.0],
            }

        logger.info(f"[{self.name}] Sample prompts after materialization:")
        self._log_sample_prompts(prompts_per_class)

        return {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": strategy.get("ensemble_method", "weighted_average"),
        }

    def _log_sample_prompts(self, prompts_per_class: dict, n: int = 3) -> None:
        """Log a few sample prompts for debugging."""
        for i, (cls, info) in enumerate(prompts_per_class.items()):
            if i >= n:
                break
            logger.info(
                f"  {cls}: {info['prompts'][:3]} "
                f"(total: {len(info['prompts'])} prompts)"
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
