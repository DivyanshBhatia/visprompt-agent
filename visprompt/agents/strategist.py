"""Refinement Strategist agent: diagnoses root causes and proposes targeted fixes.

The Strategist is what separates multi-agent from single-agent approaches.
Instead of randomly retrying, it performs structured diagnosis:
1. Examines the Critic's test failures
2. Inspects sample images from failed classes (via VLM)
3. Classifies failure causes (resolution limit, prompt issue, model gap)
4. Proposes specific, actionable fixes for the Planner

Ablation: removing the Strategist costs ~3.5% accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from visprompt.agents.base import AgentMessage, BaseAgent, TaskSpec

logger = logging.getLogger(__name__)

STRATEGIST_SYSTEM = """You are the Refinement Strategist in the VisPromptAgent system.

Your job is DIAGNOSIS, not random retry. Given the Critic's test results and
the Executor's detailed metrics, you must:

1. For each failed test / priority issue, determine the ROOT CAUSE:
   - PROMPT_ISSUE: The prompt doesn't capture the right visual features → fixable
   - RESOLUTION_LIMIT: Features needed are invisible at this resolution → document it
   - MODEL_GAP: The foundation model's embeddings can't separate these → document it
   - CONTEXT_OPPORTUNITY: Useful visual context exists but isn't leveraged → fixable
   - ENSEMBLE_ISSUE: Prompt weighting is suboptimal → fixable

2. For fixable issues, propose SPECIFIC changes to the Planner's strategy.

3. Decide whether to continue iterating or stop.

Output JSON:
{
  "diagnoses": [
    {
      "issue": str,
      "affected_classes": [str],
      "root_cause": "PROMPT_ISSUE" | "RESOLUTION_LIMIT" | "MODEL_GAP" | "CONTEXT_OPPORTUNITY" | "ENSEMBLE_ISSUE",
      "evidence": str,
      "fix": str or null,
      "expected_gain": str
    }
  ],
  "strategy_updates": [
    {
      "action": "add_prompts" | "modify_prompts" | "adjust_weights" | "add_level" | "remove_prompts",
      "target_classes": [str],
      "details": str,
      "rationale": str
    }
  ],
  "continue_iterating": bool,
  "stop_reason": str or null,
  "iteration_summary": str,
  "known_limitations": [str]
}

CRITICAL: Do NOT suggest generic improvements like "try more descriptive prompts."
Every suggestion must target specific classes with specific visual attributes.

For deciding whether to stop:
- If improvement from last iteration < 1%, consider stopping
- If remaining failures are mostly RESOLUTION_LIMIT or MODEL_GAP, stop
- If you've exhausted prompt-level fixes, stop
- Always explain your decision
"""


class RefinementStrategist(BaseAgent):
    """Diagnoses failure causes and proposes targeted prompt refinements.

    Can optionally examine sample images from failed classes using VLM
    to provide evidence-based diagnosis.
    """

    name = "refinement_strategist"
    role_description = (
        "Diagnose root causes of failures (not random retry), propose specific "
        "fixes targeting the hardest classes, and decide when to stop iterating."
    )

    def __init__(
        self,
        llm,
        sample_image_provider=None,
    ):
        """
        Args:
            llm: LLM client.
            sample_image_provider: Optional callable(class_name, n) -> list[Path]
                that returns sample images for VLM inspection.
        """
        super().__init__(llm)
        self.sample_image_provider = sample_image_provider
        self.iteration_history: list[dict] = []

    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        logger.info(f"[{self.name}] Diagnosing failures (iter {iteration})")

        critic_msg = upstream_messages.get("quality_critic")
        executor_msg = upstream_messages.get("prompt_executor")
        analyst_msg = upstream_messages.get("dataset_analyst")

        if not critic_msg or not executor_msg:
            raise ValueError("Strategist requires Critic and Executor outputs")

        # ── Optionally inspect failed class images via VLM ────────────────
        vlm_findings = self._inspect_failed_classes(
            critic_msg.content, task_spec
        )

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = self._build_prompt(
            task_spec, critic_msg, executor_msg, analyst_msg,
            vlm_findings, iteration
        )

        result = self.llm.call_json(
            prompt=prompt,
            system=STRATEGIST_SYSTEM,
            agent_name=self.name,
        )

        # Track iteration history
        self.iteration_history.append({
            "iteration": iteration,
            "primary_metric": executor_msg.content.get("eval_result", {}).get("primary_metric"),
            "continue": result.get("continue_iterating", False),
            "n_fixes": len(result.get("strategy_updates", [])),
        })

        cont = result.get("continue_iterating", False)
        logger.info(
            f"[{self.name}] {len(result.get('diagnoses', []))} diagnoses, "
            f"{len(result.get('strategy_updates', []))} updates, "
            f"continue={cont}"
        )

        return AgentMessage(
            sender=self.name,
            content=result,
            iteration=iteration,
            raw_text=json.dumps(result, indent=2),
        )

    def _inspect_failed_classes(
        self, critic_content: dict, task_spec: TaskSpec
    ) -> dict[str, str]:
        """Use VLM to examine sample images from the worst classes."""
        if not self.sample_image_provider:
            return {}

        findings = {}
        priority_issues = critic_content.get("priority_issues", [])

        for issue in priority_issues[:5]:  # Inspect top 5 issues
            for cls_name in issue.get("affected_classes", [])[:2]:
                try:
                    sample_paths = self.sample_image_provider(cls_name, 5)
                    if not sample_paths:
                        continue

                    vlm_prompt = (
                        f"Examine these images of '{cls_name}' from {task_spec.dataset_name}. "
                        f"Resolution: {task_spec.image_resolution}. "
                        f"What visual features distinguish this class? "
                        f"What might a vision model confuse it with? "
                        f"Are there useful context cues (background, co-occurring objects)?"
                    )

                    response = self.llm.call(
                        prompt=vlm_prompt,
                        images=sample_paths[:3],
                        agent_name=f"{self.name}_vlm",
                    )
                    findings[cls_name] = response

                except Exception as e:
                    logger.warning(f"VLM inspection failed for {cls_name}: {e}")

        return findings

    def _build_prompt(
        self,
        task_spec: TaskSpec,
        critic_msg: AgentMessage,
        executor_msg: AgentMessage,
        analyst_msg: AgentMessage | None,
        vlm_findings: dict[str, str],
        iteration: int,
    ) -> str:
        parts = [
            f"Diagnose failures and propose refinements for {task_spec.task_type} "
            f"on {task_spec.dataset_name} (iteration {iteration}).",
            f"\nTask:\n{task_spec.summary()}",
        ]

        # Critic results
        parts.append("\n=== Quality Critic Test Results ===")
        parts.append(critic_msg.raw_text[:4000])

        # Executor metrics
        parts.append("\n=== Executor Metrics ===")
        eval_result = executor_msg.content.get("eval_result", {})
        parts.append(f"Primary metric: {eval_result.get('primary_metric')}")
        worst = eval_result.get("worst_classes", [])
        if worst:
            parts.append("Worst classes:")
            for cls_name, score in worst[:10]:
                parts.append(f"  {cls_name}: {score:.3f}")
        confusions = eval_result.get("confusion_pairs", [])
        if confusions:
            parts.append("Top confusions:")
            for a, b, count in confusions[:5]:
                parts.append(f"  {a} → {b}: {count}")

        # VLM findings
        if vlm_findings:
            parts.append("\n=== VLM Image Inspection ===")
            for cls_name, finding in vlm_findings.items():
                parts.append(f"[{cls_name}]: {finding[:500]}")

        # Iteration history
        if self.iteration_history:
            parts.append("\n=== Iteration History ===")
            for h in self.iteration_history:
                parts.append(
                    f"Iter {h['iteration']}: {h['primary_metric']:.4f} "
                    f"({h['n_fixes']} fixes applied)"
                )
            if len(self.iteration_history) >= 2:
                prev = self.iteration_history[-1]["primary_metric"]
                curr = eval_result.get("primary_metric", prev)
                gain = curr - prev if isinstance(curr, (int, float)) else 0
                parts.append(f"Last improvement: {gain:+.4f}")

        parts.append("\nRespond with JSON following the schema in your system prompt.")

        return "\n".join(parts)
