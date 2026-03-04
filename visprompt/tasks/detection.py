"""Open-vocabulary detection task runner using GroundingDINO.

Evaluates text prompt quality for object detection by measuring
AP across class descriptions.
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


class GroundingDINODetectionRunner(BaseTaskRunner):
    """Open-vocabulary detection using GroundingDINO with text prompts.

    Evaluates how different text descriptions affect detection quality,
    particularly for rare categories.
    """

    def __init__(
        self,
        model_config: Optional[str] = None,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        image_dir: Optional[str | Path] = None,
        annotation_file: Optional[str | Path] = None,
        dataset_loader: Optional[Any] = None,
    ):
        self.model_config = model_config
        self.checkpoint = checkpoint
        self.device = device
        self.image_dir = Path(image_dir) if image_dir else None
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self._dataset_loader = dataset_loader
        self._model = None
        self._images: list[dict] = []  # [{image, annotations, name}]

    def load_data(self, split: str = "val") -> None:
        if self._images:
            return

        if self._dataset_loader is not None:
            for item in self._dataset_loader:
                self._images.append(item)
        elif self.image_dir and self.annotation_file:
            self._load_coco_format()
        else:
            raise ValueError("Provide dataset_loader or image_dir + annotation_file")

    def _load_coco_format(self):
        """Load COCO/LVIS format annotations."""
        import json
        with open(self.annotation_file) as f:
            coco = json.load(f)

        img_id_to_info = {img["id"]: img for img in coco["images"]}
        img_id_to_anns = {}
        for ann in coco.get("annotations", []):
            img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        cat_id_to_name = {
            cat["id"]: cat["name"] for cat in coco.get("categories", [])
        }

        for img_id, img_info in img_id_to_info.items():
            img_path = self.image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            anns = img_id_to_anns.get(img_id, [])
            gt_boxes = []
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO format: x,y,w,h
                gt_boxes.append({
                    "bbox": [x, y, x + w, y + h],
                    "class": ann["category_id"],
                    "class_name": cat_id_to_name.get(ann["category_id"], "unknown"),
                })

            self._images.append({
                "image_path": str(img_path),
                "annotations": gt_boxes,
                "name": img_info["file_name"],
                "width": img_info.get("width"),
                "height": img_info.get("height"),
            })

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from groundingdino.util.inference import load_model
            self._model = load_model(self.model_config, self.checkpoint, device=self.device)
        except ImportError:
            raise ImportError(
                "Install GroundingDINO: pip install groundingdino "
                "or clone https://github.com/IDEA-Research/GroundingDINO"
            )

    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        self._ensure_model()
        self.load_data()

        class_descriptions = prompts.get("class_descriptions", {})
        conf_threshold = prompts.get("confidence_threshold", 0.3)
        nms_threshold = prompts.get("nms_threshold", 0.5)

        all_pred_boxes = []
        all_gt_boxes = []

        for img_data in self._images:
            # Build text prompt by concatenating class descriptions
            text_prompt = self._build_text_prompt(class_descriptions, task_spec)

            # Run GroundingDINO
            pred_boxes = self._detect(
                img_data["image_path"],
                text_prompt,
                class_descriptions,
                task_spec,
                conf_threshold,
                nms_threshold,
            )

            # Convert GT to standard format
            gt_boxes = []
            for ann in img_data["annotations"]:
                cls_idx = self._class_name_to_idx(ann["class_name"], task_spec)
                if cls_idx >= 0:
                    gt_boxes.append({
                        "bbox": ann["bbox"],
                        "class": cls_idx,
                    })

            all_pred_boxes.append(pred_boxes)
            all_gt_boxes.append(gt_boxes)

        # Compute mAP
        result = MetricsComputer.detection_ap(
            all_pred_boxes, all_gt_boxes,
            iou_threshold=0.5,
            class_names=task_spec.class_names,
        )

        result.metadata["n_images"] = len(self._images)
        result.metadata["conf_threshold"] = conf_threshold
        result.metadata["model"] = "GroundingDINO"

        logger.info(f"[GDINORunner] mAP: {result.primary_metric:.4f}")
        return result

    def _build_text_prompt(
        self, class_descriptions: dict, task_spec: TaskSpec
    ) -> str:
        """Build the text prompt string for GroundingDINO."""
        # GroundingDINO uses ". " separated class descriptions
        descriptions = []
        for cls_name in task_spec.class_names:
            descs = class_descriptions.get(cls_name, [cls_name])
            # Use the most descriptive prompt
            descriptions.append(descs[0] if descs else cls_name)
        return ". ".join(descriptions)

    def _detect(
        self,
        image_path: str,
        text_prompt: str,
        class_descriptions: dict,
        task_spec: TaskSpec,
        conf_threshold: float,
        nms_threshold: float,
    ) -> list[dict]:
        """Run GroundingDINO on a single image."""
        try:
            from groundingdino.util.inference import load_image, predict
            import torch

            image_source, image_tensor = load_image(image_path)
            boxes, logits, phrases = predict(
                model=self._model,
                image=image_tensor,
                caption=text_prompt,
                box_threshold=conf_threshold,
                text_threshold=conf_threshold,
            )

            # NMS
            if len(boxes) > 0:
                keep = self._nms(boxes, logits, nms_threshold)
                boxes = boxes[keep]
                logits = logits[keep]
                phrases = [phrases[i] for i in keep]

            # Map phrases back to class indices
            pred_boxes = []
            h, w = image_source.shape[:2]
            for box, score, phrase in zip(boxes, logits, phrases):
                cls_idx = self._phrase_to_class(phrase, class_descriptions, task_spec)
                if cls_idx >= 0:
                    # Convert from cx,cy,w,h to x1,y1,x2,y2
                    cx, cy, bw, bh = box.tolist()
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    pred_boxes.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_idx,
                        "score": float(score),
                    })

            return pred_boxes

        except Exception as e:
            logger.warning(f"Detection failed for {image_path}: {e}")
            return []

    def _phrase_to_class(
        self, phrase: str, class_descriptions: dict, task_spec: TaskSpec
    ) -> int:
        """Map a detected phrase back to a class index."""
        phrase_lower = phrase.lower().strip()
        for idx, cls_name in enumerate(task_spec.class_names):
            if cls_name.lower() in phrase_lower or phrase_lower in cls_name.lower():
                return idx
            # Check against descriptions
            for desc in class_descriptions.get(cls_name, []):
                if phrase_lower in desc.lower() or desc.lower() in phrase_lower:
                    return idx
        return -1

    def _class_name_to_idx(self, name: str, task_spec: TaskSpec) -> int:
        for idx, cls_name in enumerate(task_spec.class_names):
            if cls_name.lower() == name.lower():
                return idx
        return -1

    @staticmethod
    def _nms(boxes, scores, threshold: float):
        """Simple NMS implementation."""
        import torch
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)

        keep = []
        while len(order) > 0:
            i = order[0].item()
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            mask = iou <= threshold
            order = order[1:][mask]

        return keep
