"""Zero-shot classification task runner using CLIP.

Supports any dataset loadable by torchvision or custom loaders.
Handles prompt ensembling with weighted averaging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from visprompt.agents.base import TaskSpec
from visprompt.tasks.base import BaseTaskRunner
from visprompt.utils.metrics import EvalResult, MetricsComputer

logger = logging.getLogger(__name__)


class CLIPClassificationRunner(BaseTaskRunner):
    """Zero-shot classification using CLIP with prompt ensembling.

    Supports:
    - Multiple prompts per class with weighted ensemble
    - Various CLIP backbones (ViT-B/32, ViT-L/14, etc.)
    - Custom datasets via image_paths + labels
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        image_paths: Optional[list[str | Path]] = None,
        dataset_loader: Optional[Any] = None,
    ):
        """
        Args:
            clip_model_name: CLIP backbone name.
            device: Torch device.
            images: Pre-loaded image array (N, H, W, C) or None.
            labels: Ground-truth label indices (N,) or None.
            image_paths: List of image file paths (alternative to images array).
            dataset_loader: Optional torchvision dataset or custom loader.
        """
        self.clip_model_name = clip_model_name
        self.device = device
        self._images = images
        self._labels = labels
        self._image_paths = image_paths
        self._dataset_loader = dataset_loader
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None

    def load_data(self, split: str = "val") -> None:
        """Load dataset if not already provided."""
        if self._images is not None or self._image_paths is not None:
            return  # Data already provided

        if self._dataset_loader is not None:
            # Assume dataset_loader is a torch Dataset with .data and .targets
            self._images = np.array(self._dataset_loader.data)
            self._labels = np.array(self._dataset_loader.targets)
        else:
            raise ValueError(
                "Provide images/labels, image_paths, or a dataset_loader"
            )

    def _ensure_model(self):
        """Lazy-load CLIP model."""
        if self._clip_model is not None:
            return

        try:
            import clip
            import torch

            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self._tokenizer = clip.tokenize
            self._torch = torch
        except ImportError:
            try:
                import open_clip

                model, _, preprocess = open_clip.create_model_and_transforms(
                    self.clip_model_name, pretrained="openai"
                )
                model = model.to(self.device)
                self._clip_model = model
                self._clip_preprocess = preprocess
                self._tokenizer = open_clip.get_tokenizer(self.clip_model_name)
                import torch
                self._torch = torch
            except ImportError:
                raise ImportError(
                    "Install either 'clip' (pip install git+https://github.com/openai/CLIP.git) "
                    "or 'open_clip' (pip install open-clip-torch)"
                )

    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        """Run zero-shot classification with prompt ensembling."""
        self._ensure_model()
        self.load_data()

        torch = self._torch
        prompts_per_class = prompts.get("prompts_per_class", {})
        ensemble_method = prompts.get("ensemble_method", "weighted_average")

        # ── 1. Encode text prompts ────────────────────────────────────────
        class_features = []
        class_names_ordered = list(task_spec.class_names)

        for cls_name in class_names_ordered:
            cls_info = prompts_per_class.get(cls_name, {
                "prompts": [f"a photo of a {cls_name}"],
                "weights": [1.0],
            })
            cls_prompts = cls_info["prompts"]
            cls_weights = cls_info.get("weights", [1.0] * len(cls_prompts))

            if not cls_prompts:
                cls_prompts = [f"a photo of a {cls_name}"]
                cls_weights = [1.0]

            # Encode all prompts for this class
            text_tokens = self._tokenizer(cls_prompts).to(self.device)
            with torch.no_grad():
                text_features = self._clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Weighted average
            weights = torch.tensor(cls_weights, device=self.device, dtype=text_features.dtype)
            weights = weights / weights.sum()
            weighted = (text_features * weights.unsqueeze(-1)).sum(dim=0)
            weighted = weighted / weighted.norm()
            class_features.append(weighted)

        # (num_classes, embed_dim)
        class_features = torch.stack(class_features)

        # ── 2. Encode images and classify ─────────────────────────────────
        all_preds = []
        all_scores = []
        batch_size = 256

        n_samples = len(self._labels) if self._labels is not None else 0
        if n_samples == 0:
            raise ValueError("No evaluation data loaded")

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            # Get batch of images
            if self._image_paths is not None:
                from PIL import Image
                batch_imgs = []
                for p in self._image_paths[start:end]:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(self._clip_preprocess(img))
                image_input = torch.stack(batch_imgs).to(self.device)
            elif self._images is not None:
                from PIL import Image
                batch_imgs = []
                for img_array in self._images[start:end]:
                    img = Image.fromarray(img_array)
                    batch_imgs.append(self._clip_preprocess(img))
                image_input = torch.stack(batch_imgs).to(self.device)
            else:
                raise ValueError("No image data available")

            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity → softmax
                logits = (100.0 * image_features @ class_features.T)
                probs = logits.softmax(dim=-1)

            preds = probs.argmax(dim=-1).cpu().numpy()
            scores = probs.max(dim=-1).values.cpu().numpy()
            all_preds.append(preds)
            all_scores.append(scores)

        all_preds = np.concatenate(all_preds)
        all_scores = np.concatenate(all_scores)
        targets = self._labels[:n_samples]

        # ── 3. Compute metrics ────────────────────────────────────────────
        result = MetricsComputer.classification_accuracy(
            all_preds, targets, class_names_ordered
        )
        result.confidence_scores = all_scores
        result.metadata["clip_model"] = self.clip_model_name
        result.metadata["n_samples"] = n_samples
        result.metadata["n_prompts_per_class"] = {
            cls: len(prompts_per_class.get(cls, {}).get("prompts", []))
            for cls in class_names_ordered[:5]
        }

        logger.info(f"[CLIPRunner] Accuracy: {result.primary_metric:.4f}")
        return result

    def get_text_embeddings(self, class_names: list[str]) -> np.ndarray:
        """Get CLIP text embeddings for class names (for Analyst's confusion pairs)."""
        self._ensure_model()
        torch = self._torch

        prompts = [f"a photo of a {name}" for name in class_names]
        tokens = self._tokenizer(prompts).to(self.device)
        with torch.no_grad():
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
