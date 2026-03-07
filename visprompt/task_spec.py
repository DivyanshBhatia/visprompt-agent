"""Task specification dataclass used throughout VisPrompt.

This is the central configuration that makes VisPrompt task-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskSpec:
    """Dataset-generic task specification.

    Defines what task to run, on what dataset, with which model.
    """

    task_type: str  # "classification", "segmentation", "detection", "retrieval"
    dataset_name: str  # e.g. "cifar100", "flowers102", "dtd"
    class_names: list[str] = field(default_factory=list)
    num_classes: int = 0
    class_hierarchy: Optional[dict[str, list[str]]] = None
    image_resolution: Optional[tuple[int, int]] = None
    domain: str = "natural"  # "natural", "medical", "remote_sensing", etc.
    foundation_model: str = "clip"
    prompt_modality: str = "text"
    metric_name: str = "accuracy"
    val_split_size: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Task: {self.task_type}",
            f"Dataset: {self.dataset_name} ({self.num_classes} classes)",
            f"Foundation model: {self.foundation_model}",
            f"Prompt modality: {self.prompt_modality}",
            f"Primary metric: {self.metric_name}",
            f"Domain: {self.domain}",
        ]
        if self.image_resolution:
            lines.append(f"Resolution: {self.image_resolution[0]}x{self.image_resolution[1]}")
        if self.class_hierarchy:
            lines.append(f"Hierarchy: {len(self.class_hierarchy)} superclasses")
        return "\n".join(lines)
