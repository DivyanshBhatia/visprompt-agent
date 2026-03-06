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
        n_descriptions: int = 10,
    ) -> tuple[EvalResult, EvalResult]:
        """Baseline: CuPL (Pratt et al. 2023) - LLM generates class descriptions.

        Returns two results:
        - cupl: descriptions only (original CuPL)
        - cupl_ensemble: descriptions + 80-template ensemble (CuPL+e)
        """
        logger.info("Running baseline: CuPL")
        cost_tracker = CostTracker()
        llm = LLMClient(
            model=llm_model, provider=llm_provider, cost_tracker=cost_tracker
        )

        descriptions_per_class = {}
        batch_size = 10  # Small batches to avoid truncation

        for i in range(0, len(self.task_spec.class_names), batch_size):
            batch = self.task_spec.class_names[i:i + batch_size]
            prompt = (
                f"Generate {n_descriptions} short visual descriptions for each class below.\n"
                f"Format each as: \"a {{class_name}}, {{short visual description}}\"\n"
                f"Keep descriptions under 15 words. Focus on shape, color, size, texture.\n\n"
                f"Classes: {', '.join(batch)}\n\n"
                f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
            )

            try:
                result = llm.call_json(
                    prompt=prompt,
                    system="Generate visual descriptions for image classification.",
                    agent_name="cupl_baseline",
                )
                for cls in batch:
                    descs = result.get(cls, result.get(cls.replace("_", " "), []))
                    if isinstance(descs, list) and descs:
                        cleaned = []
                        for d in descs:
                            cls_display = cls.replace("_", " ")
                            if cls.lower() not in d.lower() and cls_display.lower() not in d.lower():
                                d = f"a {cls_display}, {d}"
                            cleaned.append(d)
                        descriptions_per_class[cls] = cleaned
                    else:
                        descriptions_per_class[cls] = [f"a photo of a {cls.replace('_', ' ')}"]
            except Exception as e:
                logger.warning(f"CuPL batch failed: {e}")
                for cls in batch:
                    descriptions_per_class[cls] = [f"a photo of a {cls.replace('_', ' ')}"]

        # ── Variant 1: CuPL (descriptions only) ──────────────────────────
        cupl_prompts = {
            "type": "classification",
            "prompts_per_class": {
                cls: {
                    "prompts": descriptions_per_class.get(cls, [f"a photo of a {cls}"]),
                    "weights": [1.0] * len(descriptions_per_class.get(cls, [f"a photo of a {cls}"])),
                }
                for cls in self.task_spec.class_names
            },
            "ensemble_method": "weighted_average",
        }
        cupl_result = self.task_runner.evaluate(cupl_prompts, self.task_spec)
        cupl_result.metadata["cupl_cost"] = cost_tracker.summary()

        # ── Variant 2: CuPL+e (descriptions + 80-template ensemble) ──────
        cupl_e_prompts = {
            "type": "classification",
            "prompts_per_class": {
                cls: {
                    "prompts": (
                        [t.format(cls) for t in IMAGENET_TEMPLATES] +
                        descriptions_per_class.get(cls, [])
                    ),
                    "weights": (
                        [1.0] * len(IMAGENET_TEMPLATES) +
                        [1.0] * len(descriptions_per_class.get(cls, []))
                    ),
                }
                for cls in self.task_spec.class_names
            },
            "ensemble_method": "weighted_average",
        }
        cupl_e_result = self.task_runner.evaluate(cupl_e_prompts, self.task_spec)
        cupl_e_result.metadata["cupl_cost"] = cost_tracker.summary()

        return cupl_result, cupl_e_result

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

    def run_dclip(
        self,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
    ) -> EvalResult:
        """Baseline: DCLIP (Menon & Vondrick, ICLR 2023).

        'Visual Classification via Description from Large Language Models.'
        Uses GPT with the prompt 'What does a {class} look like?' to generate
        visual descriptors, then classifies using descriptor similarities.
        """
        logger.info("Running baseline: DCLIP")
        cost_tracker = CostTracker()
        llm = LLMClient(
            model=llm_model, provider=llm_provider, cost_tracker=cost_tracker
        )

        prompts_per_class = {}
        batch_size = 10

        for i in range(0, len(self.task_spec.class_names), batch_size):
            batch = self.task_spec.class_names[i:i + batch_size]
            # DCLIP uses "What does a {class} look like?" style prompts
            prompt = (
                f"For each category below, describe what it looks like.\n"
                f"Give 5-8 short visual descriptors per category.\n"
                f"Format each as: \"{'{'}class{'}'} which has {{descriptor}}\"\n"
                f"Focus on distinctive visual attributes: color, shape, texture, parts.\n\n"
                f"Categories: {', '.join(batch)}\n\n"
                f'Respond ONLY with JSON: {{"category": ["descriptor1", "descriptor2", ...]}}\n'
            )

            try:
                descriptions = llm.call_json(
                    prompt=prompt,
                    system="Describe visual attributes of object categories for image classification.",
                    agent_name="dclip_baseline",
                )
                for cls in batch:
                    descs = descriptions.get(cls, descriptions.get(cls.replace("_", " "), []))
                    if isinstance(descs, list) and descs:
                        # DCLIP format: "{class} which has {descriptor}"
                        formatted = []
                        for d in descs:
                            cls_display = cls.replace("_", " ")
                            if cls_display.lower() not in d.lower():
                                d = f"{cls_display} which has {d}"
                            formatted.append(d)
                        prompts_per_class[cls] = {
                            "prompts": formatted,
                            "weights": [1.0] * len(formatted),
                        }
                    else:
                        prompts_per_class[cls] = {
                            "prompts": [f"a photo of a {cls.replace('_', ' ')}"],
                            "weights": [1.0],
                        }
            except Exception as e:
                logger.warning(f"DCLIP batch failed: {e}")
                for cls in batch:
                    prompts_per_class[cls] = {
                        "prompts": [f"a photo of a {cls.replace('_', ' ')}"],
                        "weights": [1.0],
                    }

        prompts = {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": "weighted_average",
        }
        result = self.task_runner.evaluate(prompts, self.task_spec)
        result.metadata["dclip_cost"] = cost_tracker.summary()
        return result

    def run_zpe(self) -> EvalResult:
        """Baseline: ZPE (Allingham et al., ICML 2024).

        'Zero-shot Prompt Ensembling' — weights prompts using unlabeled test
        image statistics. For each prompt, computes expected similarity over
        test images and uses inverse-bias weighting.

        Key idea: prompts that are uniformly similar to all images carry less
        discriminative information and should be downweighted.
        """
        logger.info("Running baseline: ZPE")
        import torch

        self.task_runner._ensure_model()
        self.task_runner.load_data()

        # Get image features (cached)
        from PIL import ImageOps
        image_features = self.task_runner._encode_images_cached(
            "orig", lambda img: self.task_runner._clip_preprocess(img)
        )

        prompts_per_class = {}
        all_prompt_features = []  # (total_prompts, embed_dim)
        prompt_to_class = []  # maps prompt index to class index

        # Encode all prompts
        for cls_idx, cls_name in enumerate(self.task_spec.class_names):
            cls_prompts = [t.format(cls_name) for t in IMAGENET_TEMPLATES]
            text_tokens = self.task_runner._tokenizer(cls_prompts).to(self.task_runner.device)
            with torch.no_grad():
                text_features = self.task_runner._clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_prompt_features.append(text_features)
            prompt_to_class.extend([cls_idx] * len(cls_prompts))

        all_prompt_features = torch.cat(all_prompt_features, dim=0)

        # ZPE: compute expected similarity of each prompt over all test images
        # E_image[sim(prompt, image)] — prompts with high expected sim are biased
        with torch.no_grad():
            # (n_prompts, n_images)
            all_sims = all_prompt_features @ image_features.T
            # Expected similarity per prompt
            expected_sims = all_sims.mean(dim=1)  # (n_prompts,)

        # Build per-class prompts with ZPE weights
        # Weight = 1 / (expected_sim + epsilon) — downweight biased prompts
        # Then normalize within class
        prompt_idx = 0
        for cls_idx, cls_name in enumerate(self.task_spec.class_names):
            cls_prompts = [t.format(cls_name) for t in IMAGENET_TEMPLATES]
            n_prompts = len(cls_prompts)

            # ZPE scores: inverse of expected similarity (de-bias)
            cls_expected = expected_sims[prompt_idx:prompt_idx + n_prompts]
            # Score = -expected_sim (lower bias = higher weight)
            zpe_weights = (-cls_expected).softmax(dim=0).cpu().tolist()

            prompts_per_class[cls_name] = {
                "prompts": cls_prompts,
                "weights": zpe_weights,
            }
            prompt_idx += n_prompts

        prompts = {
            "type": "classification",
            "prompts_per_class": prompts_per_class,
            "ensemble_method": "weighted_average",
        }
        return self.task_runner.evaluate(prompts, self.task_spec)

    def run_frolic(self) -> EvalResult:
        """Baseline: Frolic (NeurIPS 2024).

        'Label-Free Prompt Distribution Learning and Bias Correcting.'
        Uses unlabeled test images to:
        1. Estimate per-class logit bias from test image distribution
        2. Apply logit correction to remove bias

        Simplified implementation: compute mean logit per class over test images,
        subtract as bias correction before classification.
        """
        logger.info("Running baseline: Frolic")
        import torch

        self.task_runner._ensure_model()
        self.task_runner.load_data()

        image_features = self.task_runner._encode_images_cached(
            "orig", lambda img: self.task_runner._clip_preprocess(img)
        )

        # Encode class prompts (80-template ensemble, averaged)
        class_features = []
        for cls_name in self.task_spec.class_names:
            cls_prompts = [t.format(cls_name) for t in IMAGENET_TEMPLATES]
            text_tokens = self.task_runner._tokenizer(cls_prompts).to(self.task_runner.device)
            with torch.no_grad():
                text_features = self.task_runner._clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            avg = text_features.mean(dim=0)
            avg = avg / avg.norm()
            class_features.append(avg)
        class_features = torch.stack(class_features)

        # Compute logits
        with torch.no_grad():
            logit_scale = self.task_runner._clip_model.logit_scale.exp()
            logits = logit_scale * (image_features @ class_features.T)

            # Frolic: estimate per-class bias = mean logit per class over all images
            bias = logits.mean(dim=0, keepdim=True)  # (1, n_classes)

            # Corrected logits
            corrected_logits = logits - bias

            # Predict
            probs = corrected_logits.softmax(dim=-1)
            preds = probs.argmax(dim=-1).detach().cpu().numpy()

        import numpy as np
        from visprompt.utils.metrics import MetricsComputer

        targets = self.task_runner._labels[:len(preds)]
        result = MetricsComputer.classification_accuracy(
            preds, targets, list(self.task_spec.class_names)
        )
        result.metadata["method"] = "frolic"
        logger.info(f"[Frolic] Accuracy: {result.primary_metric:.4f}")
        return result

    def run_clip_enhance(self) -> EvalResult:
        """Baseline: CLIP-Enhance (2024).

        'Improving CLIP Zero-Shot Classification via von Mises-Fisher Clustering.'
        Uses test image features to refine class prototypes:
        1. Start with text embeddings as initial prototypes
        2. Assign images to nearest prototype (soft assignment)
        3. Update prototypes using weighted image features
        4. Iterate to convergence

        This is essentially k-means refinement of class centers using test images.
        """
        logger.info("Running baseline: CLIP-Enhance")
        import torch

        self.task_runner._ensure_model()
        self.task_runner.load_data()

        image_features = self.task_runner._encode_images_cached(
            "orig", lambda img: self.task_runner._clip_preprocess(img)
        )

        # Initial prototypes: 80-template averaged text embeddings
        class_features = []
        for cls_name in self.task_spec.class_names:
            cls_prompts = [t.format(cls_name) for t in IMAGENET_TEMPLATES]
            text_tokens = self.task_runner._tokenizer(cls_prompts).to(self.task_runner.device)
            with torch.no_grad():
                text_features = self.task_runner._clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            avg = text_features.mean(dim=0)
            avg = avg / avg.norm()
            class_features.append(avg)
        prototypes = torch.stack(class_features)  # (n_classes, embed_dim)

        # vMF-inspired iterative refinement
        n_iters = 5
        alpha = 0.3  # mixing weight: new = alpha*image_centroid + (1-alpha)*text_prototype

        with torch.no_grad():
            for iteration in range(n_iters):
                # Soft assignment: similarity-based
                sims = image_features @ prototypes.T  # (n_images, n_classes)
                assignments = sims.softmax(dim=-1)  # soft cluster assignments

                # Update prototypes: weighted average of image features
                # (n_classes, embed_dim) = (n_classes, n_images) @ (n_images, embed_dim)
                image_centroids = assignments.T @ image_features
                image_centroids = image_centroids / image_centroids.norm(dim=-1, keepdim=True)

                # Mix with original text prototypes
                prototypes = alpha * image_centroids + (1 - alpha) * prototypes
                prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

            # Final classification
            logit_scale = self.task_runner._clip_model.logit_scale.exp()
            logits = logit_scale * (image_features @ prototypes.T)
            preds = logits.argmax(dim=-1).detach().cpu().numpy()

        import numpy as np
        from visprompt.utils.metrics import MetricsComputer

        targets = self.task_runner._labels[:len(preds)]
        result = MetricsComputer.classification_accuracy(
            preds, targets, list(self.task_spec.class_names)
        )
        result.metadata["method"] = "clip_enhance"
        result.metadata["n_iters"] = n_iters
        result.metadata["alpha"] = alpha
        logger.info(f"[CLIP-Enhance] Accuracy: {result.primary_metric:.4f}")
        return result

    def run_all(
        self,
        llm_model: str = "gpt-4o",
        llm_provider: str = "openai",
    ) -> dict[str, EvalResult]:
        """Run all baselines and return results."""
        results = {}

        # Free baselines (no LLM cost)
        results["single_template"] = self.run_single_template()
        results["80_template_ensemble"] = self.run_80_template_ensemble()
        results["waffle_clip"] = self.run_waffle_clip()

        # Test-time adaptation baselines (no LLM cost, uses test images)
        try:
            results["zpe"] = self.run_zpe()
        except Exception as e:
            logger.warning(f"ZPE baseline failed: {e}")

        try:
            results["frolic"] = self.run_frolic()
        except Exception as e:
            logger.warning(f"Frolic baseline failed: {e}")

        try:
            results["clip_enhance"] = self.run_clip_enhance()
        except Exception as e:
            logger.warning(f"CLIP-Enhance baseline failed: {e}")

        # LLM-based baselines
        try:
            cupl_result, cupl_e_result = self.run_cupl(llm_model, llm_provider)
            results["cupl"] = cupl_result
            results["cupl_ensemble"] = cupl_e_result
        except Exception as e:
            logger.warning(f"CuPL baseline failed: {e}")

        try:
            results["dclip"] = self.run_dclip(llm_model, llm_provider)
        except Exception as e:
            logger.warning(f"DCLIP baseline failed: {e}")

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
