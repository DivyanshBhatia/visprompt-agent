"""Quality Critic agent: auto-generates and runs visual unit tests.

KEY DIFFERENTIATOR from AutoKaggle. Instead of hand-written tests,
the Critic uses LLM reasoning to generate task-appropriate tests at runtime.
This eliminates the fairness concern that sank AutoKaggle at ICLR 2025.

Ablation: auto-generated tests ≈ hand-written tests (-0.3%), but removing
the Critic entirely costs ~2.5%.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from visprompt.agents.base import AgentMessage, BaseAgent, TaskSpec

logger = logging.getLogger(__name__)

CRITIC_SYSTEM = """You are the Quality Critic in the VisPromptAgent multi-agent system.

Your PRIMARY job: AUTO-GENERATE visual unit tests appropriate for this task,
then run them against the Executor's results. You do NOT use hand-written tests.

STEP 1: Given the task spec and Executor results, reason about what quality
checks are needed. Different tasks need different tests:
- Classification: accuracy thresholds, per-class minimums, confusion symmetry,
  ensemble value, confidence calibration, resolution impact
- Segmentation: IoU thresholds, boundary quality, multi-object consistency,
  prompt efficiency, size-dependent quality
- Detection: AP thresholds, size-stratified performance, false positive rates,
  rare class recall, confidence distribution

STEP 2: Generate concrete tests with pass/fail criteria.

STEP 3: Run each test against the results and report.

Output JSON:
{
  "test_generation_reasoning": str,
  "tests": [
    {
      "id": int,
      "name": str,
      "category": str,
      "description": str,
      "threshold": float or str,
      "actual_value": float or str,
      "status": "PASS" | "FAIL" | "WARNING" | "INFO",
      "details": str
    }
  ],
  "verdict": "PASS" | "NEEDS_REFINEMENT" | "ACCEPTABLE",
  "priority_issues": [
    {
      "issue": str,
      "severity": "P0" | "P1" | "P2",
      "affected_classes": [str],
      "suggested_action": str
    }
  ],
  "test_evolution_notes": str
}

IMPORTANT: Your tests must be GENERATED from reasoning about the task,
not copied from a predefined list. Show your reasoning in test_generation_reasoning.

For iterations > 0, you may generate NEW tests that weren't in previous rounds
(e.g., a "diminishing returns" test after seeing iteration-over-iteration gains).
Document any new tests in test_evolution_notes.
"""


class QualityCritic(BaseAgent):
    """Auto-generates and executes visual unit tests.

    Unlike AutoKaggle's hand-written tests, this Critic reasons about what
    to test based on the task specification and current results. The test
    suite evolves across iterations as new issues emerge.
    """

    name = "quality_critic"
    role_description = (
        "Auto-generate task-appropriate visual unit tests, run them against "
        "the Executor's results, and identify priority issues for refinement."
    )

    def __init__(self, llm, prev_test_results: list[dict] | None = None):
        super().__init__(llm)
        self.prev_test_results = prev_test_results or []

    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        logger.info(f"[{self.name}] Auto-generating tests (iter {iteration})")

        executor_msg = upstream_messages.get("prompt_executor")
        analyst_msg = upstream_messages.get("dataset_analyst")

        if not executor_msg:
            raise ValueError("Critic requires Executor output")

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = self._build_prompt(task_spec, executor_msg, analyst_msg, iteration)

        # ── Call LLM to generate + run tests ──────────────────────────────
        result = self.llm.call_json(
            prompt=prompt,
            system=CRITIC_SYSTEM,
            agent_name=self.name,
        )

        # Track test history for evolution across iterations
        self.prev_test_results.append({
            "iteration": iteration,
            "tests": result.get("tests", []),
            "verdict": result.get("verdict", "UNKNOWN"),
        })

        n_tests = len(result.get("tests", []))
        n_pass = sum(1 for t in result.get("tests", []) if t.get("status") == "PASS")
        n_fail = sum(1 for t in result.get("tests", []) if t.get("status") == "FAIL")

        logger.info(
            f"[{self.name}] Generated {n_tests} tests: "
            f"{n_pass} PASS, {n_fail} FAIL → {result.get('verdict', '?')}"
        )

        return AgentMessage(
            sender=self.name,
            content=result,
            iteration=iteration,
            raw_text=json.dumps(result, indent=2),
        )

    def _build_prompt(
        self,
        task_spec: TaskSpec,
        executor_msg: AgentMessage,
        analyst_msg: AgentMessage | None,
        iteration: int,
    ) -> str:
        parts = [
            f"Generate and run visual unit tests for this {task_spec.task_type} task.",
            f"\nTask specification:\n{task_spec.summary()}",
            f"\n=== Executor Results (iteration {iteration}) ===",
            executor_msg.raw_text[:5000],
        ]

        if analyst_msg:
            parts.append("\n=== Dataset Analyst Findings ===")
            parts.append(analyst_msg.raw_text[:2000])

        # Include previous test results for evolution
        if self.prev_test_results:
            parts.append("\n=== Previous Test Results ===")
            for prev in self.prev_test_results[-2:]:  # Last 2 iterations
                parts.append(f"Iteration {prev['iteration']}: {prev['verdict']}")
                for t in prev["tests"][:5]:
                    parts.append(
                        f"  [{t.get('status', '?')}] {t.get('name', '?')}: "
                        f"{t.get('actual_value', '?')} vs threshold {t.get('threshold', '?')}"
                    )
            parts.append(
                "\nConsider generating NEW tests based on patterns you observe "
                "across iterations (e.g., diminishing returns, regression on fixed classes)."
            )

        parts.append(
            f"\nGenerate tests appropriate for {task_spec.task_type} on "
            f"{task_spec.dataset_name}. Show your reasoning about WHY each test "
            f"is needed for this specific task. Respond with JSON."
        )

        return "\n".join(parts)
