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

PLANNER_SYSTEM_CLASSIFICATION = """You are the Prompt Planner for zero-shot CLASSIFICATION with CLIP.

Given the Dataset Analyst's report, design a prompt strategy that BEATS standard template ensembles.

KEY INSIGHT: CLIP benefits from class-specific visual descriptions far more than generic template
variations. "a photo of a {class}" and "a blurry photo of a {class}" produce nearly identical
embeddings. What works is describing WHAT the class looks like:
  - "a camel, a large tan animal with one or two humps on its back in a desert"
  - "a ray, a flat diamond-shaped fish with a long thin tail swimming in the ocean"

Output JSON with:
{
  "strategy_name": str,
  "base_templates": [str],
  "description_prompt": str,
  "class_specific_prompts": {
    "class_name": {
      "prompts": [str],
      "rationale": str
    }
  },
  "ensemble_method": "weighted_average",
  "base_weight": float,
  "description_weight": float,
  "rationale": str
}

RULES:
1. base_templates: 3-5 simple templates using {class} placeholder.
   Example: ["a photo of a {class}.", "a photo of the {class}."]
   These get base_weight (should be LOW, e.g. 0.2-0.3).

2. description_prompt: A system prompt that will be sent to the LLM to generate
   visual descriptions for EVERY class. This is the most important part.
   The descriptions should emphasize: shape, color, texture, size, habitat/context,
   and distinctive features visible even at low resolution.
   Be specific: "Describe what each class looks like visually, focusing on
   features distinguishable at 32x32 resolution: overall shape, dominant colors,
   typical background/setting."

3. class_specific_prompts: Write prompts ONLY for the Analyst's hardest confusion
   pairs (5-15 classes max). These should be discriminative descriptions.
   Write COMPLETE prompts with class names filled in.
   Example: "ray": {"prompts": ["a manta ray, flat diamond-shaped body gliding through blue water",
                                 "a stingray, a flat grey fish with wing-like fins near the ocean floor"]}
   These get description_weight (should be HIGH, e.g. 0.7-0.8).

CRITICAL:
- NEVER use negation ("not a X") — CLIP ignores negation.
- NEVER use unreachable placeholders like {habitat}. Only {class} is supported.
- base_weight + description_weight should roughly sum to 1.0.
- Focus class_specific_prompts on classes the Analyst flagged as hardest.
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
- Keep base_templates that worked.
- Refine description_prompt to generate better descriptions for failing classes.
- Add/update class_specific_prompts for classes the Strategist identified.
- Adjust base_weight and description_weight if the balance was wrong.
- NEVER use negation ("not a X") in any prompt.
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
