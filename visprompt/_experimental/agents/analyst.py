"""Dataset Analyst agent: examines dataset structure and predicts failure modes.

The Analyst is the first agent in the pipeline. It provides domain-aware
analysis that enables targeted prompt design rather than generic prompting.
Ablation shows removing this agent costs ~3.3% accuracy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from visprompt.agents.base import AgentMessage, BaseAgent, TaskSpec

logger = logging.getLogger(__name__)

ANALYST_SYSTEM = """You are the Dataset Analyst in the VisPromptAgent multi-agent system.

Your job is to analyze a vision dataset and predict where the foundation model will struggle,
BEFORE any prompts are designed. You provide structured intelligence to the Prompt Planner.

You must produce JSON with these keys:
{
  "dataset_summary": {
    "name": str,
    "num_classes": int,
    "resolution": str,
    "domain": str,
    "key_characteristics": [str]
  },
  "confusion_pairs": [
    {"class_a": str, "class_b": str, "similarity_score": float, "reason": str}
  ],
  "domain_insights": [
    {"insight": str, "implication_for_prompts": str}
  ],
  "class_groups": {
    "easy": [str],
    "medium": [str],
    "hard": [str]
  },
  "recommendations": [str]
}

Be specific. Use visual reasoning about what features distinguish classes.
Consider resolution, texture, color, shape, and context cues.
"""


class DatasetAnalyst(BaseAgent):
    """Examines dataset structure and predicts failure modes.

    For classification: analyzes class names, hierarchy, and computes
    text-embedding similarities to predict confusion pairs.

    For segmentation: analyzes object characteristics, boundary complexity,
    and domain-specific challenges.

    For detection: analyzes class distribution, object sizes, and
    visual ambiguity between categories.
    """

    name = "dataset_analyst"
    role_description = (
        "Analyze the dataset structure, predict confusion pairs and failure modes, "
        "and provide domain-specific insights to guide prompt design."
    )

    def __init__(
        self,
        llm,
        sample_images: Optional[list[str | Path]] = None,
        text_embeddings: Optional[np.ndarray] = None,
        class_names: Optional[list[str]] = None,
    ):
        super().__init__(llm)
        self.sample_images = sample_images or []
        self.text_embeddings = text_embeddings
        self.class_names = class_names

    def run(
        self,
        task_spec: TaskSpec,
        upstream_messages: dict[str, AgentMessage],
        iteration: int = 0,
    ) -> AgentMessage:
        """Analyze the dataset and produce structured intelligence."""
        logger.info(f"[{self.name}] Analyzing dataset: {task_spec.dataset_name}")

        # ── 1. Compute text-embedding confusion pairs (if available) ──────
        confusion_pairs = self._compute_confusion_pairs(task_spec)

        # ── 2. Build analysis prompt ──────────────────────────────────────
        prompt = self._build_prompt(task_spec, confusion_pairs)

        # ── 3. Call LLM (with optional sample images) ─────────────────────
        images_to_send = self.sample_images[:5] if self.sample_images else None
        response = self.llm.call_json(
            prompt=prompt,
            system=ANALYST_SYSTEM,
            images=images_to_send,
            agent_name=self.name,
        )

        # ── 4. Merge computed confusion pairs with LLM analysis ───────────
        if confusion_pairs and "confusion_pairs" in response:
            # Add embedding-based pairs that LLM might have missed
            existing = {(p["class_a"], p["class_b"]) for p in response["confusion_pairs"]}
            for pair in confusion_pairs:
                key = (pair["class_a"], pair["class_b"])
                rev = (pair["class_b"], pair["class_a"])
                if key not in existing and rev not in existing:
                    response["confusion_pairs"].append(pair)

        logger.info(
            f"[{self.name}] Found {len(response.get('confusion_pairs', []))} confusion pairs, "
            f"{len(response.get('domain_insights', []))} domain insights"
        )

        return AgentMessage(
            sender=self.name,
            content=response,
            iteration=iteration,
            raw_text=json.dumps(response, indent=2),
        )

    def _compute_confusion_pairs(
        self, task_spec: TaskSpec
    ) -> list[dict[str, Any]]:
        """Compute pairwise text-embedding similarities to find confusion pairs."""
        if self.text_embeddings is None or not task_spec.class_names:
            return []

        embeddings = self.text_embeddings
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-8)

        # Cosine similarity matrix
        sim_matrix = embeddings @ embeddings.T
        np.fill_diagonal(sim_matrix, 0)

        # Extract top pairs
        pairs = []
        n = sim_matrix.shape[0]
        flat_idx = np.argsort(sim_matrix.flatten())[::-1]
        seen = set()
        for idx in flat_idx[:50]:  # Top 50 candidates
            i, j = divmod(idx, n)
            if i >= j:
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            score = float(sim_matrix[i, j])
            if score < 0.6:
                break
            pairs.append({
                "class_a": task_spec.class_names[i],
                "class_b": task_spec.class_names[j],
                "similarity_score": round(score, 3),
                "reason": f"Text embedding cosine similarity: {score:.3f}",
            })
            if len(pairs) >= 15:
                break

        return pairs

    def _build_prompt(
        self,
        task_spec: TaskSpec,
        confusion_pairs: list[dict],
    ) -> str:
        """Build the analysis prompt for the LLM."""
        parts = [
            f"Analyze this {task_spec.task_type} dataset for visual prompt engineering:\n",
            task_spec.summary(),
            "",
        ]

        if task_spec.class_names:
            parts.append(f"Classes ({len(task_spec.class_names)}):")
            # Show all classes if ≤50, otherwise sample
            if len(task_spec.class_names) <= 50:
                parts.append(", ".join(task_spec.class_names))
            else:
                parts.append(", ".join(task_spec.class_names[:30]) + f"... (+{len(task_spec.class_names)-30} more)")

        if task_spec.class_hierarchy:
            parts.append("\nClass hierarchy (superclass → classes):")
            for super_cls, sub_classes in list(task_spec.class_hierarchy.items())[:10]:
                parts.append(f"  {super_cls}: {', '.join(sub_classes)}")

        if confusion_pairs:
            parts.append("\nPre-computed confusion pairs (text embedding similarity):")
            for p in confusion_pairs[:10]:
                parts.append(f"  {p['class_a']} ↔ {p['class_b']}: {p['similarity_score']}")

        parts.append(f"""
Task-specific analysis instructions:
- For CLASSIFICATION: Focus on which classes will be confused by {task_spec.foundation_model}.
  Consider texture, color, shape, context cues at the given resolution.
- For SEGMENTATION: Focus on boundary complexity, object size distribution,
  occlusion patterns, and multi-instance scenarios.
- For DETECTION: Focus on rare vs frequent classes, object scale variation,
  inter-class visual similarity, and context dependencies.

Provide your analysis as JSON matching the schema described in your system prompt.
""")

        return "\n".join(parts)
