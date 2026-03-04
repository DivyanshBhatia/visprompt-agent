"""Base task runner interface for foundation model inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from visprompt.agents.base import TaskSpec
from visprompt.utils.metrics import EvalResult


class BaseTaskRunner(ABC):
    """Abstract interface for task-specific model inference.

    Each TaskRunner wraps a foundation model and handles:
    - Loading the model and data
    - Converting prompts into model-specific inputs
    - Running inference on a validation split
    - Computing task-appropriate metrics
    """

    @abstractmethod
    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        """Run inference with given prompts and return evaluation results.

        Args:
            prompts: Materialized prompts from the Executor.
            task_spec: Task specification.

        Returns:
            EvalResult with primary metric, per-class breakdown, etc.
        """
        ...

    @abstractmethod
    def load_data(self, split: str = "val") -> None:
        """Load the dataset split for evaluation."""
        ...
