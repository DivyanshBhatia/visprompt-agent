"""VisPromptAgent Pipeline: orchestrates multi-agent iterative prompt refinement.

This is the main entry point. The pipeline:
1. Runs the Dataset Analyst once
2. Iteratively runs: Planner → Executor → Critic → Strategist
3. Stops when the Strategist says to stop or max_iterations is reached
4. Returns full results with cost tracking and reasoning traces
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from visprompt.agents.analyst import DatasetAnalyst
from visprompt.agents.base import AgentMessage, TaskSpec
from visprompt.agents.critic import QualityCritic
from visprompt.agents.executor import PromptExecutor
from visprompt.agents.planner import PromptPlanner
from visprompt.agents.strategist import RefinementStrategist
from visprompt.tasks.base import BaseTaskRunner
from visprompt.utils.llm import CostTracker, LLMClient

logger = logging.getLogger(__name__)


@dataclass
class IterationRecord:
    """Record of a single refinement iteration."""

    iteration: int
    primary_metric: float
    primary_metric_name: str
    planner_strategy: dict = field(default_factory=dict)
    executor_diagnostics: dict = field(default_factory=dict)
    critic_tests: list = field(default_factory=list)
    critic_verdict: str = ""
    strategist_diagnoses: list = field(default_factory=list)
    continue_iterating: bool = True
    duration_s: float = 0.0


@dataclass
class PipelineResult:
    """Full result of a VisPromptAgent pipeline run."""

    task_spec: TaskSpec
    iterations: list[IterationRecord] = field(default_factory=list)
    final_metric: float = 0.0
    final_metric_name: str = ""
    cost_summary: dict = field(default_factory=dict)
    analyst_report: dict = field(default_factory=dict)
    reasoning_trace: list[dict] = field(default_factory=list)
    total_duration_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"=== VisPromptAgent Results ===",
            f"Task: {self.task_spec.task_type} on {self.task_spec.dataset_name}",
            f"Final {self.final_metric_name}: {self.final_metric:.4f}",
            f"Iterations: {len(self.iterations)}",
            f"Total cost: ${self.cost_summary.get('total_cost_usd', 0):.4f}",
            f"Total time: {self.total_duration_s:.1f}s",
            "",
            "Per-iteration metrics:",
        ]
        for rec in self.iterations:
            lines.append(
                f"  Iter {rec.iteration}: {rec.primary_metric:.4f} "
                f"({rec.critic_verdict}) [{rec.duration_s:.1f}s]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task": {
                "type": self.task_spec.task_type,
                "dataset": self.task_spec.dataset_name,
                "model": self.task_spec.foundation_model,
            },
            "final_metric": self.final_metric,
            "final_metric_name": self.final_metric_name,
            "n_iterations": len(self.iterations),
            "per_iteration": [
                {
                    "iteration": r.iteration,
                    "metric": r.primary_metric,
                    "verdict": r.critic_verdict,
                    "continue": r.continue_iterating,
                    "duration_s": r.duration_s,
                }
                for r in self.iterations
            ],
            "cost": self.cost_summary,
            "total_duration_s": self.total_duration_s,
        }


class VisPromptPipeline:
    """Main orchestrator for the multi-agent VisPromptAgent system.

    Usage:
        pipeline = VisPromptPipeline(
            task_spec=TaskSpec(...),
            task_runner=CLIPClassificationRunner(...),
            llm_model="gpt-4o",
        )
        result = pipeline.run(max_iterations=3)
        print(result.summary())
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        task_runner: BaseTaskRunner,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
        llm_api_key: Optional[str] = None,
        llm_temperature: float = 0.3,
        sample_images: Optional[list[str | Path]] = None,
        sample_image_provider=None,
        text_embeddings=None,
        output_dir: Optional[str | Path] = None,
    ):
        self.task_spec = task_spec
        self.task_runner = task_runner
        self.output_dir = Path(output_dir) if output_dir else Path("experiments")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Shared cost tracker ───────────────────────────────────────────
        self.cost_tracker = CostTracker()

        # ── LLM client (shared by all agents) ─────────────────────────────
        self.llm = LLMClient(
            model=llm_model,
            provider=llm_provider,
            api_key=llm_api_key,
            temperature=llm_temperature,
            cost_tracker=self.cost_tracker,
        )

        # ── Initialize agents ─────────────────────────────────────────────
        self.analyst = DatasetAnalyst(
            llm=self.llm,
            sample_images=sample_images,
            text_embeddings=text_embeddings,
            class_names=task_spec.class_names,
        )
        self.planner = PromptPlanner(llm=self.llm)
        self.executor = PromptExecutor(llm=self.llm, task_runner=task_runner)
        self.critic = QualityCritic(llm=self.llm)
        self.strategist = RefinementStrategist(
            llm=self.llm,
            sample_image_provider=sample_image_provider,
        )

    def run(self, max_iterations: int = 3) -> PipelineResult:
        """Execute the full multi-agent pipeline.

        Args:
            max_iterations: Maximum refinement iterations (default 3).

        Returns:
            PipelineResult with full metrics, costs, and reasoning traces.
        """
        t_start = time.time()
        result = PipelineResult(task_spec=self.task_spec)

        logger.info(
            f"Starting VisPromptAgent pipeline: {self.task_spec.task_type} "
            f"on {self.task_spec.dataset_name}"
        )

        # ── Phase 1: Dataset Analysis (runs once) ─────────────────────────
        logger.info("Phase 1: Dataset Analysis")
        analyst_msg = self.analyst.run(self.task_spec, {}, iteration=0)
        result.analyst_report = analyst_msg.content
        result.reasoning_trace.append({
            "agent": "dataset_analyst",
            "iteration": 0,
            "output": analyst_msg.content,
        })

        # ── Phase 2: Iterative Refinement ─────────────────────────────────
        messages: dict[str, AgentMessage] = {
            "dataset_analyst": analyst_msg,
        }

        for iteration in range(max_iterations):
            t_iter = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'='*60}")

            # ── 2a. Prompt Planning ───────────────────────────────────────
            planner_msg = self.planner.run(self.task_spec, messages, iteration)
            messages["prompt_planner"] = planner_msg
            result.reasoning_trace.append({
                "agent": "prompt_planner",
                "iteration": iteration,
                "output": planner_msg.content,
            })

            # ── 2b. Prompt Execution ──────────────────────────────────────
            executor_msg = self.executor.run(self.task_spec, messages, iteration)
            messages["prompt_executor"] = executor_msg
            result.reasoning_trace.append({
                "agent": "prompt_executor",
                "iteration": iteration,
                "output": executor_msg.content,
            })

            # ── 2c. Quality Critique ──────────────────────────────────────
            critic_msg = self.critic.run(self.task_spec, messages, iteration)
            messages["quality_critic"] = critic_msg
            result.reasoning_trace.append({
                "agent": "quality_critic",
                "iteration": iteration,
                "output": critic_msg.content,
            })

            # ── 2d. Record iteration ──────────────────────────────────────
            eval_result = executor_msg.content.get("eval_result", {})
            record = IterationRecord(
                iteration=iteration,
                primary_metric=eval_result.get("primary_metric", 0.0),
                primary_metric_name=eval_result.get("primary_metric_name", ""),
                planner_strategy=planner_msg.content,
                executor_diagnostics=executor_msg.content.get("diagnostics", {}),
                critic_tests=critic_msg.content.get("tests", []),
                critic_verdict=critic_msg.content.get("verdict", "UNKNOWN"),
                duration_s=time.time() - t_iter,
            )

            # ── 2e. Check if we should stop ───────────────────────────────
            if critic_msg.content.get("verdict") == "PASS":
                record.continue_iterating = False
                result.iterations.append(record)
                logger.info(f"Critic says PASS — stopping after iteration {iteration}")
                break

            # Hard early-stop: if we've done 2+ iterations and best metric
            # hasn't improved by > 0.5%, stop regardless of Strategist
            if iteration >= 1 and result.iterations:
                best_so_far = max(r.primary_metric for r in result.iterations)
                current = record.primary_metric
                if current <= best_so_far and (best_so_far - current) < 0.005:
                    # Check if ANY iteration improved by > 0.5% over iter 0
                    iter0_metric = result.iterations[0].primary_metric
                    if current - iter0_metric < 0.005:
                        record.continue_iterating = False
                        result.iterations.append(record)
                        logger.info(
                            f"Early stop: no meaningful improvement after {iteration + 1} "
                            f"iterations (best={best_so_far:.4f}, current={current:.4f})"
                        )
                        break

            if iteration == max_iterations - 1:
                record.continue_iterating = False
                result.iterations.append(record)
                logger.info(f"Max iterations ({max_iterations}) reached")
                break

            # ── 2f. Refinement Strategy ───────────────────────────────────
            strategist_msg = self.strategist.run(self.task_spec, messages, iteration)
            messages["refinement_strategist"] = strategist_msg
            result.reasoning_trace.append({
                "agent": "refinement_strategist",
                "iteration": iteration,
                "output": strategist_msg.content,
            })

            record.strategist_diagnoses = strategist_msg.content.get("diagnoses", [])
            record.continue_iterating = strategist_msg.content.get("continue_iterating", False)

            result.iterations.append(record)

            if not record.continue_iterating:
                logger.info(
                    f"Strategist says stop: "
                    f"{strategist_msg.content.get('stop_reason', 'no reason given')}"
                )
                break

        # ── Finalize ──────────────────────────────────────────────────────
        if result.iterations:
            last = result.iterations[-1]
            result.final_metric = last.primary_metric
            result.final_metric_name = last.primary_metric_name

        result.cost_summary = self.cost_tracker.summary()
        result.total_duration_s = time.time() - t_start

        # ── Save results ──────────────────────────────────────────────────
        self._save_results(result)

        logger.info(f"\n{result.summary()}")
        return result

    def _save_results(self, result: PipelineResult) -> None:
        """Save results to disk."""
        run_name = (
            f"{self.task_spec.dataset_name}_{self.task_spec.task_type}_"
            f"{int(time.time())}"
        )
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Summary
        with open(run_dir / "summary.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Full reasoning trace
        with open(run_dir / "reasoning_trace.json", "w") as f:
            json.dump(result.reasoning_trace, f, indent=2, default=str)

        # Analyst report
        with open(run_dir / "analyst_report.json", "w") as f:
            json.dump(result.analyst_report, f, indent=2, default=str)

        # Cost breakdown
        with open(run_dir / "cost.json", "w") as f:
            json.dump(result.cost_summary, f, indent=2)

        logger.info(f"Results saved to {run_dir}")


class AblationPipeline:
    """Runs ablation experiments by disabling individual agents.

    Produces Table 3 from the paper:
    - Full system
    - w/o Dataset Analyst
    - w/o Quality Critic
    - w/o Refinement Strategist
    - Single agent (no roles)
    """

    def __init__(
        self,
        task_spec: TaskSpec,
        task_runner: BaseTaskRunner,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
        llm_api_key: Optional[str] = None,
        output_dir: Optional[str | Path] = None,
        **kwargs,
    ):
        self.task_spec = task_spec
        self.task_runner = task_runner
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.output_dir = Path(output_dir) if output_dir else Path("experiments/ablation")
        self.kwargs = kwargs

    def run_all(self, max_iterations: int = 3) -> dict[str, PipelineResult]:
        """Run all ablation configurations."""
        results = {}

        # Full system
        logger.info("=== ABLATION: Full System ===")
        results["full"] = self._run_full(max_iterations)

        # w/o Analyst (skip analysis, use generic prompts)
        logger.info("=== ABLATION: w/o Dataset Analyst ===")
        results["no_analyst"] = self._run_no_analyst(max_iterations)

        # w/o Critic (no tests, no quality feedback)
        logger.info("=== ABLATION: w/o Quality Critic ===")
        results["no_critic"] = self._run_no_critic(max_iterations)

        # w/o Strategist (no refinement, single iteration)
        logger.info("=== ABLATION: w/o Refinement Strategist ===")
        results["no_strategist"] = self._run_single_iteration()

        # Single agent (one LLM call does everything)
        logger.info("=== ABLATION: Single Agent ===")
        results["single_agent"] = self._run_single_agent()

        # Print comparison
        self._print_comparison(results)
        return results

    def _run_full(self, max_iterations: int) -> PipelineResult:
        pipeline = VisPromptPipeline(
            task_spec=self.task_spec,
            task_runner=self.task_runner,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            output_dir=self.output_dir / "full",
            **self.kwargs,
        )
        return pipeline.run(max_iterations)

    def _run_no_analyst(self, max_iterations: int) -> PipelineResult:
        """Run without Dataset Analyst — Planner gets no domain intelligence."""
        pipeline = VisPromptPipeline(
            task_spec=self.task_spec,
            task_runner=self.task_runner,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            output_dir=self.output_dir / "no_analyst",
            **self.kwargs,
        )
        # Override: make Analyst return empty report
        original_run = pipeline.analyst.run
        def dummy_analyst(task_spec, upstream, iteration=0):
            return AgentMessage(
                sender="dataset_analyst",
                content={"dataset_summary": {"name": task_spec.dataset_name}, "confusion_pairs": [], "domain_insights": [], "class_groups": {"easy": [], "medium": [], "hard": []}, "recommendations": []},
                iteration=iteration,
                raw_text="{}",
            )
        pipeline.analyst.run = dummy_analyst
        return pipeline.run(max_iterations)

    def _run_no_critic(self, max_iterations: int) -> PipelineResult:
        """Run without Quality Critic — no test feedback, always continue."""
        pipeline = VisPromptPipeline(
            task_spec=self.task_spec,
            task_runner=self.task_runner,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            output_dir=self.output_dir / "no_critic",
            **self.kwargs,
        )
        # Override: Critic always passes with no test info
        original_run = pipeline.critic.run
        def dummy_critic(task_spec, upstream, iteration=0):
            return AgentMessage(
                sender="quality_critic",
                content={"tests": [], "verdict": "NEEDS_REFINEMENT", "priority_issues": [], "test_generation_reasoning": "Critic disabled for ablation"},
                iteration=iteration,
                raw_text="{}",
            )
        pipeline.critic.run = dummy_critic
        return pipeline.run(max_iterations)

    def _run_single_iteration(self) -> PipelineResult:
        """Run with only 1 iteration (removes Strategist's value)."""
        pipeline = VisPromptPipeline(
            task_spec=self.task_spec,
            task_runner=self.task_runner,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            output_dir=self.output_dir / "single_iter",
            **self.kwargs,
        )
        return pipeline.run(max_iterations=1)

    def _run_single_agent(self) -> PipelineResult:
        """Run with a single LLM call doing everything (no agent decomposition)."""
        pipeline = VisPromptPipeline(
            task_spec=self.task_spec,
            task_runner=self.task_runner,
            llm_model=self.llm_model,
            llm_provider=self.llm_provider,
            llm_api_key=self.llm_api_key,
            output_dir=self.output_dir / "single_agent",
            **self.kwargs,
        )

        # Override Analyst + Planner with single generic call
        def single_agent_analyst(task_spec, upstream, iteration=0):
            return AgentMessage(
                sender="dataset_analyst", content={}, iteration=iteration, raw_text="{}"
            )

        def single_agent_planner(task_spec, upstream, iteration=0):
            prompt = (
                f"Generate the best text prompts for zero-shot {task_spec.task_type} "
                f"on {task_spec.dataset_name} ({task_spec.num_classes} classes) "
                f"using {task_spec.foundation_model}.\n"
                f"Classes: {', '.join(task_spec.class_names[:50])}\n"
                f"Output a JSON prompt strategy."
            )
            strategy = pipeline.llm.call_json(
                prompt=prompt,
                system="You are an AI assistant. Generate prompt strategies as JSON.",
                agent_name="single_agent",
            )
            return AgentMessage(
                sender="prompt_planner",
                content=strategy,
                iteration=iteration,
                raw_text=json.dumps(strategy, indent=2),
            )

        pipeline.analyst.run = single_agent_analyst
        pipeline.planner.run = single_agent_planner
        return pipeline.run(max_iterations=1)

    def _print_comparison(self, results: dict[str, PipelineResult]) -> None:
        print("\n" + "=" * 60)
        print("ABLATION RESULTS")
        print("=" * 60)
        print(f"{'Configuration':<30} {'Metric':>10} {'Δ':>8} {'Cost':>8}")
        print("-" * 60)

        full_metric = results["full"].final_metric if "full" in results else 0

        for name, result in results.items():
            delta = result.final_metric - full_metric
            cost = result.cost_summary.get("total_cost_usd", 0)
            print(
                f"{name:<30} {result.final_metric:>10.4f} "
                f"{delta:>+8.4f} ${cost:>7.4f}"
            )
        print("=" * 60)
