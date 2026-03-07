"""Semantic segmentation using CLIPSeg from HuggingFace.

CLIPSeg uses CLIP's text encoder — our prompt ensembling transfers directly.
Multiple text prompts per class → average predicted masks → final segmentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from visprompt.task_spec import TaskSpec
from visprompt.tasks.base import BaseTaskRunner
from visprompt.utils.metrics import EvalResult

logger = logging.getLogger(__name__)


class CLIPSegRunner(BaseTaskRunner):
    """Semantic segmentation using CLIPSeg with prompt ensembling.

    For each class, encodes multiple text prompts and averages the
    predicted heatmaps (logit-level fusion, same as classification).
    """

    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: str = "cuda",
        image_dir: Optional[str | Path] = None,
        mask_dir: Optional[str | Path] = None,
        image_list: Optional[list[dict]] = None,
        class_names: Optional[list[str]] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self._image_list = image_list or []
        self._class_names = class_names or []
        self._model = None
        self._processor = None

    def load_data(self, split: str = "val") -> None:
        if self._image_list:
            return

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        logger.info(f"Loading CLIPSeg model: {self.model_name}...")

        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            self._processor = CLIPSegProcessor.from_pretrained(self.model_name)
            self._model = CLIPSegForImageSegmentation.from_pretrained(self.model_name).to(self.device)
        except (ImportError, Exception):
            from transformers import AutoProcessor, AutoModel
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)

        self._model.eval()
        self._torch = torch
        logger.info("CLIPSeg model loaded")

    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        """Run segmentation with prompt ensembling and compute mIoU.

        prompts format (same as classification):
        {
            "prompts_per_class": {
                "class_name": {
                    "prompts": ["desc1", "desc2", ...],
                    "weights": [w1, w2, ...]
                }
            }
        }
        """
        self._ensure_model()
        torch = self._torch
        from PIL import Image
        import torch.nn.functional as F

        prompts_per_class = prompts.get("prompts_per_class", {})
        class_names = list(task_spec.class_names)
        n_classes = len(class_names)

        # Collect per-class IoU
        class_intersections = {cls: 0.0 for cls in class_names}
        class_unions = {cls: 0.0 for cls in class_names}
        n_images = 0

        logger.info(f"Running CLIPSeg on {len(self._image_list)} images, "
                     f"{n_classes} classes...")

        for img_idx, img_data in enumerate(self._image_list):
            if img_idx % 50 == 0:
                logger.info(f"  Processing image {img_idx}/{len(self._image_list)}...")

            try:
                image = Image.open(img_data["image_path"]).convert("RGB")
                gt_mask = np.array(Image.open(img_data["mask_path"]))

                img_w, img_h = image.size
                mask_h, mask_w = gt_mask.shape[:2]

                # For each class, compute weighted average heatmap
                all_heatmaps = []  # (n_classes, H, W)

                for cls_idx, cls_name in enumerate(class_names):
                    cls_info = prompts_per_class.get(cls_name, {
                        "prompts": [cls_name],
                        "weights": [1.0],
                    })
                    cls_prompts = cls_info.get("prompts", [cls_name])
                    cls_weights = cls_info.get("weights", [1.0] * len(cls_prompts))

                    if not cls_prompts:
                        cls_prompts = [cls_name]
                        cls_weights = [1.0]

                    # Normalize weights
                    total_w = sum(cls_weights)
                    cls_weights = [w / total_w for w in cls_weights]

                    # Accumulate weighted heatmaps
                    heatmap_acc = None

                    for prompt, weight in zip(cls_prompts, cls_weights):
                        inputs = self._processor(
                            text=[prompt],
                            images=image,
                            return_tensors="pt",
                            padding=True,
                        ).to(self.device)

                        with torch.no_grad():
                            outputs = self._model(**inputs)

                        # CLIPSeg outputs logits — handle different output formats
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif hasattr(outputs, 'decoder_output'):
                            logits = outputs.decoder_output
                        else:
                            # Try first tensor output
                            logits = outputs[0]

                        # Ensure shape is (1, H, W) or (H, W)
                        if logits.dim() == 4:
                            logits = logits[0, 0]  # (B, 1, H, W) -> (H, W)
                        elif logits.dim() == 3:
                            logits = logits[0]  # (B, H, W) or (1, H, W) -> (H, W)
                        # else: already (H, W)

                        # Resize to GT mask size
                        logits_resized = F.interpolate(
                            logits.unsqueeze(0).unsqueeze(0),
                            size=(mask_h, mask_w),
                            mode="bilinear",
                            align_corners=False,
                        )[0, 0]  # (mask_h, mask_w)

                        if heatmap_acc is None:
                            heatmap_acc = logits_resized * weight
                        else:
                            heatmap_acc += logits_resized * weight

                    all_heatmaps.append(heatmap_acc)

                # Stack heatmaps: (n_classes, H, W)
                heatmaps = torch.stack(all_heatmaps, dim=0)

                # Predict class per pixel (argmax across classes)
                pred_mask = heatmaps.argmax(dim=0).cpu().numpy()  # (H, W)

                # Compute per-class IoU
                for cls_idx, cls_name in enumerate(class_names):
                    cls_label = img_data.get("class_to_label", {}).get(cls_name, cls_idx + 1)
                    pred_binary = (pred_mask == cls_idx)
                    gt_binary = (gt_mask == cls_label)

                    intersection = np.logical_and(pred_binary, gt_binary).sum()
                    union = np.logical_or(pred_binary, gt_binary).sum()

                    class_intersections[cls_name] += intersection
                    class_unions[cls_name] += union

                n_images += 1

            except Exception as e:
                logger.warning(f"Segmentation failed for {img_data.get('image_path', '?')}: {e}")

        # Compute mIoU
        per_class_iou = {}
        ious = []
        for cls_name in class_names:
            if class_unions[cls_name] > 0:
                iou = class_intersections[cls_name] / class_unions[cls_name]
                per_class_iou[cls_name] = float(iou)
                ious.append(iou)
            else:
                per_class_iou[cls_name] = 0.0

        miou = float(np.mean(ious)) if ious else 0.0

        logger.info(f"[CLIPSeg] mIoU: {miou:.4f} ({n_images} images)")

        return EvalResult(
            primary_metric=miou,
            primary_metric_name="mIoU",
            per_class_metrics=per_class_iou,
            metadata={
                "model": self.model_name,
                "n_images": n_images,
                "n_classes": n_classes,
            },
        )
