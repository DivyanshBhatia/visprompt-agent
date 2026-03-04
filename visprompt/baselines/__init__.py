"""Baseline prompt methods for comparison.

Implements the methods in Table 2 of the paper:
- Single template ("a photo of a {class}")
- 80-template ensemble (Radford et al. 2021)
- CuPL (Pratt et al. 2023): LLM-generated descriptions
- WaffleCLIP (Roth et al. 2023): random descriptor ensembles
- Single LLM agent (no role decomposition)
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any, Optional

from visprompt.agents.base import TaskSpec
from visprompt.tasks.base import BaseTaskRunner
from visprompt.utils.llm import CostTracker, LLMClient
from visprompt.utils.metrics import EvalResult

logger = logging.getLogger(__name__)


# ── CLIP 80-template ensemble (Radford et al. 2021) ──────────────────────────
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


class BaselineRunner:
    """Unified interface for running baselines and collecting results."""

    def __init__(self, task_runner: BaseTaskRunner, task_spec: TaskSpec):
        self.task_runner = task_runner
        self.task_spec = task_spec

    def run_single_template(self) -> EvalResult:
        """Baseline: 'a photo of a {class}' only."""
        logger.info("Running baseline: Single Template")
        prompts = {
            "type": "classification",
            "prompts_per_class": {
                cls: {"prompts": [f"a photo of a {cls}"], "weights": [1.0]}
                for cls in self.task_spec.class_names
            },
            "ensemble_method": "weighted_average",
        }
        return self.task_runner.evaluate(prompts, self.task_spec)

    def run_80_template_ensemble(self) -> EvalResult:
        """Baseline: 80-template ensemble (Radford et al. 2021)."""
        logger.info("Running baseline: 80-Template Ensemble")
        prompts = {
            "type": "classification",
            "prompts_per_class": {
                cls: {
                    "prompts": [t.format(cls) for t in IMAGENET_TEMPLATES],
                    "weights": [1.0] * len(IMAGENET_TEMPLATES),
                }
                for cls in self.task_spec.class_names
            },
            "ensemble_method": "weighted_average",
        }
        return self.task_runner.evaluate(prompts, self.task_spec)

    def run_cupl(
        self,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
        n_descriptions: int = 5,
    ) -> EvalResult:
        """Baseline: CuPL (Pratt et al. 2023) - LLM generates class descriptions."""
        logger.info("Running baseline: CuPL")
        cost_tracker = CostTracker()
        llm = LLMClient(
            model=llm_model, provider=llm_provider, cost_tracker=cost_tracker
        )

        prompts_per_class = {}
        batch_size = 20  # Process classes in batches to reduce API calls

        for i in range(0, len(self.task_spec.class_names), batch_size):
            batch = self.task_spec.class_names[i:i + batch_size]
            prompt = (
                f"For each of these visual classes, generate {n_descriptions} "
                f"concise visual descriptions that would help identify the object "
                f"in an image. Focus on distinctive visual features.\n\n"
                f"Classes: {', '.join(batch)}\n\n"
                f"Respond with JSON: {{\"class_name\": [\"desc1\", \"desc2\", ...]}}"
            )

            try:
                descriptions = llm.call_json(
                    prompt=prompt,
                    system="Generate visual descriptions for image classification.",
                    agent_name="cupl_baseline",
                )

                for cls in batch:
                    descs = descriptions.get(cls, [f"a photo of a {cls}"])
                    full_prompts = [f"a photo of a {cls}"] + [
                        f"{desc}" for desc in descs
                    ]
                    prompts_per_class[cls] = {
                        "prompts": full_prompts,
                        "weights": [1.0] * len(full_prompts),
                    }
            except Exception as e:
                logger.warning(f"CuPL batch failed: {e}")
                for cls in batch:
                    prompts_per_class[cls] = {
                        "prompts": [f"a photo of a {cls}"],
                        "weights": [1.0],
                    }

        prompts = {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": "weighted_average",
        }

        result = self.task_runner.evaluate(prompts, self.task_spec)
        result.metadata["cupl_cost"] = cost_tracker.summary()
        return result

    def run_waffle_clip(self, n_random: int = 15, vocab_size: int = 1000) -> EvalResult:
        """Baseline: WaffleCLIP (Roth et al. 2023) - random descriptor ensembles."""
        logger.info("Running baseline: WaffleCLIP")

        # Random word vocabulary (simplified — real WaffleCLIP uses a larger vocab)
        random_words = [
            "bright", "dark", "colorful", "textured", "smooth", "rough",
            "large", "small", "round", "angular", "metallic", "wooden",
            "organic", "geometric", "striped", "spotted", "furry", "scaly",
            "shiny", "matte", "transparent", "opaque", "symmetric", "curved",
            "flat", "tall", "wide", "narrow", "detailed", "simple",
            "natural", "artificial", "indoor", "outdoor", "urban", "rural",
            "wet", "dry", "soft", "hard", "warm", "cool", "vibrant", "muted",
            "patterned", "solid", "complex", "elegant", "compact", "sprawling",
        ]

        prompts_per_class = {}
        for cls in self.task_spec.class_names:
            random_prompts = [f"a photo of a {cls}"]  # Always include base
            for _ in range(n_random):
                n_words = random.randint(2, 5)
                words = random.sample(random_words, n_words)
                random_prompts.append(f"a {' '.join(words)} photo of a {cls}")

            prompts_per_class[cls] = {
                "prompts": random_prompts,
                "weights": [1.0] * len(random_prompts),
            }

        prompts = {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": "weighted_average",
        }
        return self.task_runner.evaluate(prompts, self.task_spec)

    def run_all(
        self,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
    ) -> dict[str, EvalResult]:
        """Run all baselines and return results."""
        results = {}

        results["single_template"] = self.run_single_template()
        results["80_template_ensemble"] = self.run_80_template_ensemble()
        results["waffle_clip"] = self.run_waffle_clip()

        try:
            results["cupl"] = self.run_cupl(llm_model, llm_provider)
        except Exception as e:
            logger.warning(f"CuPL baseline failed: {e}")

        self._print_comparison(results)
        return results

    def _print_comparison(self, results: dict[str, EvalResult]) -> None:
        print("\n" + "=" * 50)
        print("BASELINE COMPARISON")
        print("=" * 50)
        print(f"{'Method':<30} {'Metric':>10}")
        print("-" * 50)
        for name, result in sorted(
            results.items(), key=lambda x: x[1].primary_metric
        ):
            print(f"{name:<30} {result.primary_metric:>10.4f}")
        print("=" * 50)
