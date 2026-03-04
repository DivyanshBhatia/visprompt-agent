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

ARCHITECTURE: Your strategy uses a HYBRID approach:
- 80-template ensemble (Radford et al.) is AUTOMATICALLY included as a stable base
- You design the LLM description generation and class-specific overrides that ADD to it

KEY INSIGHT: CLIP was trained on short alt-text captions. Descriptions must be SHORT (under 15 words)
and use natural language, NOT full sentences. Good vs bad:
  GOOD: "a camel, a large tan animal with humps in a desert"
  GOOD: "a ray, a flat diamond-shaped fish with wing-like fins"
  BAD: "The camel is a large animal that is typically tan in color and has one or two humps on its back."
  BAD: "A photograph showing a ray which is a type of flat fish."

Output JSON with:
{
  "strategy_name": str,
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
1. base_weight (0.2-0.4): Weight per 80-template prompt. Low but provides stability.
   The 80 templates are added automatically — do NOT include base_templates.

2. description_prompt: A system prompt sent to the LLM to generate 7-10 SHORT visual
   descriptions per class. This is the most important part.
   Emphasize: "Generate short descriptions under 15 words each. Format: 'a {class}, {visual features}'.
   Focus on shape, dominant color, size, texture, and typical setting visible at low resolution."

3. description_weight (0.6-0.8): Weight per description prompt. Higher than base_weight.

4. class_specific_prompts: Write DISCRIMINATIVE prompts ONLY for the hardest confusion
   pairs (5-15 classes). These should highlight what makes each class DIFFERENT from
   its confusable counterpart.
   Example: "whale": {"prompts": ["a whale, massive dark body with water spout, much larger than a dolphin",
                                    "a whale, huge grey marine mammal with wide flat tail"]}
   These get description_weight * 1.5.

CRITICAL:
- NEVER use negation ("not a X") — CLIP ignores negation.
- NEVER include base_templates — the 80-template ensemble is added automatically.
- Keep ALL descriptions SHORT (under 15 words).
- base_weight + description_weight need NOT sum to 1.0 (they are per-prompt weights, not totals).
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
- The 80-template ensemble is included automatically as base — focus on descriptions.
- Refine description_prompt to generate better SHORT descriptions for failing classes.
- Add/update class_specific_prompts for classes the Strategist identified.
- Adjust base_weight and description_weight if the balance was wrong.
- NEVER use negation ("not a X") in any prompt.
- Keep ALL descriptions under 15 words.
- NOTE: Descriptions from previous iterations are CACHED. Only class_specific_prompts
  and weight adjustments take effect in refinement iterations.
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
