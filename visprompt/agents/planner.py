"""Prompt Planner agent: designs hierarchical prompt strategies.

Receives the Analyst's report and designs a multi-level strategy.
Task-agnostic: produces text prompts for classification, point/box
strategies for segmentation, and text descriptions for detection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from visprompt.agents.base import AgentMessage, BaseAgent, TaskSpec

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_CLASSIFICATION = """You are the Prompt Planner for zero-shot CLASSIFICATION.

Given the Dataset Analyst's report, design a hierarchical prompt strategy for CLIP.

Output JSON with:
{
  "strategy_name": str,
  "levels": [
    {
      "level": int,
      "name": str,
      "description": str,
      "templates": [str],
      "target_classes": "all" | [str],
      "weight": float
    }
  ],
  "class_specific_prompts": {
    "class_name": {
      "prompts": [str],
      "rationale": str
    }
  },
  "ensemble_method": "weighted_average" | "max_similarity" | "rank_fusion",
  "ensemble_weights": [float],
  "rationale": str
}

Guidelines:
- Level 1: Base templates (all classes). Include resolution-aware variants.
- Level 2: Hierarchical context using superclasses or domain groupings.
- Level 3: Confusion-pair discriminators targeting the Analyst's hardest pairs.
- class_specific_prompts: Override prompts for the most confused classes.
- Be specific about visual attributes: texture, color, shape, habitat, context.
"""

PLANNER_SYSTEM_SEGMENTATION = """You are the Prompt Planner for interactive SEGMENTATION.

Given the Dataset Analyst's report, design a prompt placement strategy for SAM/SAM2.

Output JSON with:
{
  "strategy_name": str,
  "prompt_types": ["point", "box", "text"],
  "point_strategy": {
    "initial_points": int,
    "placement": "center" | "grid" | "saliency" | "adaptive",
    "negative_points": bool,
    "refinement_strategy": str
  },
  "box_strategy": {
    "source": "ground_truth" | "detector" | "saliency",
    "padding_ratio": float,
    "multi_box": bool
  },
  "text_prompts": {
    "template": str,
    "class_specific": {str: str}
  },
  "multi_object_handling": str,
  "difficulty_adaptation": {
    "easy": str,
    "hard": str
  },
  "rationale": str
}
"""

PLANNER_SYSTEM_DETECTION = """You are the Prompt Planner for open-vocabulary DETECTION.

Given the Dataset Analyst's report, design text prompt strategies for GroundingDINO.

Output JSON with:
{
  "strategy_name": str,
  "levels": [
    {
      "level": int,
      "name": str,
      "description": str,
      "template": str,
      "target_classes": "all" | [str]
    }
  ],
  "class_specific_descriptions": {
    "class_name": {
      "prompts": [str],
      "rationale": str
    }
  },
  "confidence_threshold": float,
  "nms_threshold": float,
  "rare_class_strategy": str,
  "rationale": str
}

Focus on:
- Descriptive prompts that disambiguate visually similar categories
- Extra detail for rare classes (few training examples = weaker features)
- Context-aware descriptions (habitat, typical co-occurring objects)
"""

PLANNER_SYSTEMS = {
    "classification": PLANNER_SYSTEM_CLASSIFICATION,
    "segmentation": PLANNER_SYSTEM_SEGMENTATION,
    "detection": PLANNER_SYSTEM_DETECTION,
}

REFINEMENT_ADDENDUM = """
REFINEMENT CONTEXT:
This is iteration {iteration}. You have access to the previous strategy AND
the Strategist's diagnosis of what went wrong.

Previous strategy summary:
{prev_strategy}

Strategist's refinement recommendations:
{strategist_recs}

Update your strategy to address the diagnosed issues.
Keep what worked, change what didn't. Be specific about what changed and why.
"""


class PromptPlanner(BaseAgent):
    """Designs task-appropriate prompt strategies informed by the Analyst.

    For classification: hierarchical text templates with confusion-pair targeting.
    For segmentation: point/box placement strategies with difficulty adaptation.
    For detection: descriptive text prompts with rare-class emphasis.
    """

    name = "prompt_planner"
    role_description = (
        "Design a hierarchical, targeted prompt strategy based on the Dataset "
        "Analyst's findings. Allocate 'prompt budget' to the hardest cases."
    )

    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        logger.info(f"[{self.name}] Designing prompt strategy (iter {iteration})")

        system = PLANNER_SYSTEMS.get(task_spec.task_type, PLANNER_SYSTEM_CLASSIFICATION)

        # ── Build prompt ──────────────────────────────────────────────────
        prompt_parts = [
            f"Design a prompt strategy for this task:\n{task_spec.summary()}\n",
        ]

        # Include Analyst report
        analyst_msg = upstream_messages.get("dataset_analyst")
        if analyst_msg:
            prompt_parts.append("=== Dataset Analyst Report ===")
            prompt_parts.append(analyst_msg.raw_text[:4000])

        # Include refinement context if not first iteration
        if iteration > 0:
            strategist_msg = upstream_messages.get("refinement_strategist")
            prev_planner_msg = upstream_messages.get("prompt_planner")

            refinement_ctx = REFINEMENT_ADDENDUM.format(
                iteration=iteration,
                prev_strategy=(
                    prev_planner_msg.raw_text[:2000] if prev_planner_msg else "N/A"
                ),
                strategist_recs=(
                    strategist_msg.raw_text[:2000] if strategist_msg else "N/A"
                ),
            )
            prompt_parts.append(refinement_ctx)

        prompt = "\n\n".join(prompt_parts)

        # ── Call LLM ──────────────────────────────────────────────────────
        strategy = self.llm.call_json(
            prompt=prompt,
            system=system,
            agent_name=self.name,
        )

        logger.info(f"[{self.name}] Strategy: {strategy.get('strategy_name', 'unnamed')}")

        return AgentMessage(
            sender=self.name,
            content=strategy,
            iteration=iteration,
            raw_text=json.dumps(strategy, indent=2),
        )
