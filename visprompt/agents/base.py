"""Base agent class for the multi-agent VisPromptAgent system."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from visprompt.utils.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Structured message passed between agents."""

    sender: str
    content: dict[str, Any]
    iteration: int = 0
    raw_text: str = ""

    def __repr__(self) -> str:
        keys = list(self.content.keys())
        return f"AgentMessage(sender={self.sender}, keys={keys}, iter={self.iteration})"


@dataclass
class TaskSpec:
    """Dataset-generic task specification passed to all agents.

    This is the central configuration that makes VisPromptAgent task-agnostic.
    """

    task_type: str  # "classification", "segmentation", "detection"
    dataset_name: str  # e.g. "cifar100", "davis2017", "lvis"
    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    class_hierarchy: Optional[dict[str, list[str]]] = None  # superclass -> classes
    image_resolution: Optional[tuple[int, int]] = None
    domain: str = "natural"  # "natural", "medical", "remote_sensing", etc.
    foundation_model: str = "clip"  # "clip", "sam2", "grounding_dino"
    prompt_modality: str = "text"  # "text", "point", "box", "mixed"
    metric_name: str = "accuracy"  # primary metric name
    val_split_size: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary for LLM prompts."""
        lines = [
            f"Task: {self.task_type}",
            f"Dataset: {self.dataset_name} ({self.num_classes} classes)",
            f"Foundation model: {self.foundation_model}",
            f"Prompt modality: {self.prompt_modality}",
            f"Primary metric: {self.metric_name}",
            f"Domain: {self.domain}",
        ]
        if self.image_resolution:
            lines.append(f"Resolution: {self.image_resolution[0]}×{self.image_resolution[1]}")
        if self.class_hierarchy:
            lines.append(f"Hierarchy: {len(self.class_hierarchy)} superclasses")
        return "\n".join(lines)


class BaseAgent(ABC):
    """Abstract base class for all VisPromptAgent agents.

    Each agent:
    - Receives messages from upstream agents
    - Uses an LLM to reason about its specific role
    - Produces a structured AgentMessage for downstream agents
    - Tracks its own cost via the shared CostTracker
    """

    name: str = "base_agent"
    role_description: str = "A generic agent."

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        """Execute this agent's role and produce an output message.

        Args:
            task_spec: The task specification (dataset-generic).
            upstream_messages: Messages from other agents keyed by agent name.
            iteration: Current refinement iteration (0-indexed).

        Returns:
            AgentMessage with structured content.
        """
        ...

    def _build_system_prompt(self) -> str:
        """Construct the system prompt for this agent."""
        return (
            f"You are the {self.name} in the VisPromptAgent multi-agent system.\n"
            f"Your role: {self.role_description}\n\n"
            "Always respond with structured, actionable output.\n"
            "Be specific and quantitative where possible.\n"
            "Format your response as valid JSON."
        )

    def _format_upstream(self, upstream_messages: dict[str, AgentMessage]) -> str:
        """Format upstream messages into a readable context string."""
        if not upstream_messages:
            return "No upstream messages available."

        parts = []
        for name, msg in upstream_messages.items():
            parts.append(f"=== From {name} (iteration {msg.iteration}) ===")
            if msg.raw_text:
                parts.append(msg.raw_text[:3000])  # Truncate for context window
            else:
                import json
                parts.append(json.dumps(msg.content, indent=2, default=str)[:3000])
        return "\n\n".join(parts)
