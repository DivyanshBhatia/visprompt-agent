"""Zero-shot classification task runner using CLIP.

Supports any dataset loadable by torchvision or custom loaders.
Handles prompt ensembling with weighted averaging.

Architecture upgrades (v2):
- ViT-L/14 backbone by default (+12-14% over ViT-B/32)
- Image feature caching across iterations (10x faster refinement)
- Multi-view TTA: original + flip
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
    - Image feature caching across evaluate() calls
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-L/14",
        pretrained: str = "openai",
        device: str = "cuda",
        images: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        image_paths: Optional[list[str | Path]] = None,
        dataset_loader: Optional[Any] = None,
    ):
        """
        Args:
            clip_model_name: CLIP backbone name. Default ViT-L/14 for best accuracy.
            pretrained: Pretrained weights. 'openai' for OpenAI CLIP, or open_clip pretrained tag.
            device: Torch device.
            images: Pre-loaded image array (N, H, W, C) or None.
            labels: Ground-truth label indices (N,) or None.
            image_paths: List of image file paths (alternative to images array).
            dataset_loader: Optional torchvision dataset or custom loader.
        """
        self.clip_model_name = clip_model_name
        self.pretrained = pretrained
        self.device = device
        self._images = images
        self._labels = labels
        self._image_paths = image_paths
        self._dataset_loader = dataset_loader
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None

        # ── Image feature cache (persists across evaluate() calls) ────────
        # Since images don't change between iterations, we encode them once.
        self._image_features_cache = {}  # key: view_name -> (N, embed_dim)

    def load_data(self, split: str = "val") -> None:
        """Load dataset if not already provided."""
        if self._images is not None or self._image_paths is not None:
            return  # Data already provided

        if self._dataset_loader is not None:
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

        import torch
        self._torch = torch

        # Try OpenAI CLIP first (only works for 'openai' pretrained)
        if self.pretrained == "openai":
            try:
                import clip
                self._clip_model, self._clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
                self._tokenizer = clip.tokenize
                return
            except (ImportError, RuntimeError):
                pass

        # Use open_clip for all other models (EVA-CLIP, MetaCLIP, SigLIP, LAION, etc.)
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model_name, pretrained=self.pretrained
            )
            model = model.to(self.device)
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.clip_model_name)

            # SigLIP and some models don't have logit_scale — create a default
            if not hasattr(self._clip_model, 'logit_scale'):
                self._clip_model.logit_scale = torch.nn.Parameter(
                    torch.tensor(4.6052, device=self.device)  # log(100)
                )
                logger.info(f"[CLIPRunner] Model lacks logit_scale, using default=100.0")

        except ImportError:
            raise ImportError(
                "Install either 'clip' (pip install git+https://github.com/openai/CLIP.git) "
                "or 'open_clip' (pip install open-clip-torch)"
            )

    def _encode_images_cached(self, view_name, transform_fn):
        """Encode images with a given transform, caching results.

        Args:
            view_name: Cache key (e.g., 'orig', 'flip')
            transform_fn: Function that takes a PIL Image and returns preprocessed tensor

        Returns:
            Tensor of shape (N, embed_dim), normalized.
        """
        if view_name in self._image_features_cache:
            logger.info(f"[CLIPRunner] Using cached image features for '{view_name}'")
            return self._image_features_cache[view_name]

        torch = self._torch
        from PIL import Image

        n_samples = len(self._labels) if self._labels is not None else 0
        batch_size = 128
        all_features = []

        logger.info(
            f"[CLIPRunner] Encoding {n_samples} images for view '{view_name}' "
            f"(model={self.clip_model_name})..."
        )

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            batch_imgs = []
            if self._image_paths is not None:
                for p in self._image_paths[start:end]:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(transform_fn(img))
            elif self._images is not None:
                for img_array in self._images[start:end]:
                    img = Image.fromarray(img_array)
                    batch_imgs.append(transform_fn(img))
            else:
                raise ValueError("No image data available")

            image_input = torch.stack(batch_imgs).to(self.device)
            with torch.no_grad():
                features = self._clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        self._image_features_cache[view_name] = all_features
        logger.info(
            f"[CLIPRunner] Cached '{view_name}': {all_features.shape}"
        )
        return all_features

    def evaluate(self, prompts, task_spec, temperature=None):
        """Run zero-shot classification with logit-level ensembling + TTA.

        Architecture:

        1. LOGIT-LEVEL ENSEMBLING: Keep all prompt embeddings separate, compute
           per-prompt cosine similarities, then weighted-average the LOGITS.

        2. MULTI-VIEW TTA: Original + horizontally flipped. Features are cached
           across iterations so only the first call pays encoding cost.

        3. IMAGE FEATURE CACHING: Images don't change between refinement
           iterations. We encode images once and reuse for all subsequent calls.

        4. TEMPERATURE SCALING: Optional softmax temperature for calibration.
           Lower temperature = sharper predictions (more confident).
        """
        self._ensure_model()
        self.load_data()

        torch = self._torch
        prompts_per_class = prompts.get("prompts_per_class", {})
        class_names_ordered = list(task_spec.class_names)
        n_classes = len(class_names_ordered)

        # ── 1. Encode ALL text prompts (keep separate per class) ──────────
        class_text_data = []

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

            text_tokens = self._tokenizer(cls_prompts).to(self.device)
            with torch.no_grad():
                text_features = self._clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            weights = torch.tensor(cls_weights, device=self.device, dtype=text_features.dtype)
            weights = weights / weights.sum()

            class_text_data.append((text_features, weights))

        # ── 2. Get cached image features (TTA views) ─────────────────────
        from PIL import ImageOps

        image_features_orig = self._encode_images_cached(
            "orig",
            lambda img: self._clip_preprocess(img),
        )
        image_features_flip = self._encode_images_cached(
            "flip",
            lambda img: self._clip_preprocess(ImageOps.mirror(img)),
        )

        n_samples = image_features_orig.shape[0]

        # ── 3. Logit-level ensembling over all views ─────────────────────
        with torch.no_grad():
            logit_scale = self._clip_model.logit_scale.exp()

            logits_orig = torch.zeros(n_samples, n_classes, device=self.device)
            logits_flip = torch.zeros(n_samples, n_classes, device=self.device)

            for cls_idx, (text_feats, weights) in enumerate(class_text_data):
                sims_orig = logit_scale * (image_features_orig @ text_feats.T)
                logits_orig[:, cls_idx] = (sims_orig * weights.unsqueeze(0)).sum(dim=-1)

                sims_flip = logit_scale * (image_features_flip @ text_feats.T)
                logits_flip[:, cls_idx] = (sims_flip * weights.unsqueeze(0)).sum(dim=-1)

            # Average TTA logits
            logits = (logits_orig + logits_flip) / 2.0

            # Apply temperature scaling if specified
            if temperature is not None and temperature > 0:
                logits = logits / temperature

            probs = logits.softmax(dim=-1)

        all_preds = probs.argmax(dim=-1).detach().cpu().numpy()
        all_scores = probs.max(dim=-1).values.detach().cpu().numpy()
        targets = self._labels[:n_samples]

        # ── 4. Compute metrics ────────────────────────────────────────────
        result = MetricsComputer.classification_accuracy(
            all_preds, targets, class_names_ordered
        )
        result.confidence_scores = all_scores
        result.metadata["clip_model"] = self.clip_model_name
        result.metadata["n_samples"] = n_samples
        result.metadata["ensemble_method"] = "logit_level"
        result.metadata["tta"] = True
        result.metadata["temperature"] = temperature
        result.metadata["image_cache_hits"] = len(self._image_features_cache)
        result.metadata["n_prompts_per_class"] = {
            cls: len(prompts_per_class.get(cls, {}).get("prompts", []))
            for cls in class_names_ordered[:5]
        }

        logger.info(f"[CLIPRunner] Accuracy: {result.primary_metric:.4f} "
                     f"(logit-ensemble + TTA, model={self.clip_model_name})")
        return result

    def get_text_embeddings(self, class_names):
        """Get CLIP text embeddings for class names (for Analyst's confusion pairs)."""
        self._ensure_model()
        torch = self._torch

        prompts = [f"a photo of a {name}" for name in class_names]
        tokens = self._tokenizer(prompts).to(self.device)
        with torch.no_grad():
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
