"""Open-vocabulary detection using OWL-ViT v2 from HuggingFace.

No compilation needed — pure transformers installation.
Evaluates text prompt quality for object detection via mAP.
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


class OWLViTDetectionRunner(BaseTaskRunner):
    """Open-vocabulary detection using OWL-ViT v2.

    Uses HuggingFace transformers — no custom compilation.
    Text prompts directly affect detection quality, making this
    ideal for prompt optimization experiments.
    """

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
        image_dir: Optional[str | Path] = None,
        annotation_file: Optional[str | Path] = None,
        max_images: Optional[int] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.image_dir = Path(image_dir) if image_dir else None
        self.annotation_file = Path(annotation_file) if annotation_file else None
        self.max_images = max_images
        self._model = None
        self._processor = None
        self._images: list[dict] = []
        self._cat_id_to_name: dict[int, str] = {}
        self._cat_id_to_idx: dict[int, int] = {}

    def load_data(self, split: str = "val") -> None:
        if self._images:
            return

        if self.annotation_file and self.image_dir:
            self._load_coco_format()
        else:
            raise ValueError("Provide annotation_file + image_dir for COCO format")

    def _load_coco_format(self):
        """Load COCO format annotations."""
        import json

        logger.info(f"Loading COCO annotations from {self.annotation_file}...")
        with open(self.annotation_file) as f:
            coco = json.load(f)

        # Build category mappings
        categories = sorted(coco.get("categories", []), key=lambda c: c["id"])
        self._cat_id_to_name = {cat["id"]: cat["name"] for cat in categories}
        # Map COCO category IDs (non-contiguous) to 0-indexed
        for idx, cat in enumerate(categories):
            self._cat_id_to_idx[cat["id"]] = idx

        img_id_to_info = {img["id"]: img for img in coco["images"]}
        img_id_to_anns = {}
        for ann in coco.get("annotations", []):
            img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Load images (with optional limit)
        image_ids = list(img_id_to_info.keys())
        if self.max_images:
            rng = np.random.RandomState(42)
            image_ids = rng.choice(image_ids, size=min(self.max_images, len(image_ids)),
                                    replace=False).tolist()

        for img_id in image_ids:
            img_info = img_id_to_info[img_id]
            img_path = self.image_dir / img_info["file_name"]
            if not img_path.exists():
                continue

            anns = img_id_to_anns.get(img_id, [])
            gt_boxes = []
            for ann in anns:
                if ann.get("iscrowd", 0):
                    continue
                x, y, w, h = ann["bbox"]
                cat_id = ann["category_id"]
                if cat_id in self._cat_id_to_idx:
                    gt_boxes.append({
                        "bbox": [x, y, x + w, y + h],
                        "class": self._cat_id_to_idx[cat_id],
                        "class_name": self._cat_id_to_name[cat_id],
                    })

            if gt_boxes:  # Only include images with annotations
                self._images.append({
                    "image_path": str(img_path),
                    "annotations": gt_boxes,
                    "width": img_info.get("width"),
                    "height": img_info.get("height"),
                })

        logger.info(f"Loaded {len(self._images)} images with annotations")

    def _ensure_model(self):
        if self._model is not None:
            return

        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        import torch

        logger.info(f"Loading OWL-ViT model: {self.model_name}...")
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch
        logger.info("OWL-ViT model loaded")

    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        """Run detection and compute mAP.

        prompts should contain:
        - class_descriptions: {class_name: [desc1, desc2, ...]}
        - confidence_threshold: float (default 0.1)
        """
        self._ensure_model()
        self.load_data()

        torch = self._torch
        class_descriptions = prompts.get("class_descriptions", {})
        conf_threshold = prompts.get("confidence_threshold", 0.1)

        # Build text queries — one primary description per class
        text_queries = []
        for cls_name in task_spec.class_names:
            descs = class_descriptions.get(cls_name, [cls_name])
            # Use first description as the query
            text_queries.append(descs[0] if descs else cls_name)

        all_pred_boxes = []
        all_gt_boxes = []

        from PIL import Image

        logger.info(f"Running detection on {len(self._images)} images with "
                     f"{len(text_queries)} class queries...")

        for img_idx, img_data in enumerate(self._images):
            if img_idx % 100 == 0:
                logger.info(f"  Processing image {img_idx}/{len(self._images)}...")

            try:
                image = Image.open(img_data["image_path"]).convert("RGB")
                w, h = image.size

                # Run OWL-ViT
                inputs = self._processor(
                    text=[text_queries],
                    images=image,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Post-process: manual extraction (works across all transformers versions)
                target_sizes = torch.tensor([[h, w]], device=self.device)

                # Try built-in post-processing first
                post_processed = False
                for obj in [
                    getattr(self._processor, 'image_processor', None),
                    self._processor,
                    self._model,
                ]:
                    if obj is not None and hasattr(obj, 'post_process_object_detection'):
                        try:
                            results = obj.post_process_object_detection(
                                outputs, threshold=conf_threshold, target_sizes=target_sizes
                            )[0]
                            post_processed = True
                            break
                        except Exception:
                            continue

                if not post_processed:
                    # Manual post-processing (guaranteed to work)
                    logits = outputs.logits[0]  # (num_queries, num_classes)
                    boxes = outputs.pred_boxes[0]  # (num_queries, 4)

                    # Get max class score per query
                    probs = logits.sigmoid()
                    scores, labels = probs.max(dim=-1)
                    mask = scores > conf_threshold

                    # Convert normalized cx,cy,w,h -> pixel x1,y1,x2,y2
                    filtered_boxes = boxes[mask]
                    if len(filtered_boxes) > 0:
                        cx, cy, bw, bh = filtered_boxes.unbind(-1)
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        converted_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        converted_boxes = torch.zeros((0, 4), device=self.device)

                    results = {
                        "boxes": converted_boxes,
                        "scores": scores[mask],
                        "labels": labels[mask],
                    }

                pred_boxes = []
                for box, score, label in zip(
                    results["boxes"], results["scores"], results["labels"]
                ):
                    x1, y1, x2, y2 = box.tolist()
                    cls_idx = int(label.item())
                    if cls_idx < len(task_spec.class_names):
                        pred_boxes.append({
                            "bbox": [x1, y1, x2, y2],
                            "class": cls_idx,
                            "score": float(score.item()),
                        })

                all_pred_boxes.append(pred_boxes)

            except Exception as e:
                logger.warning(f"Detection failed for {img_data['image_path']}: {e}")
                all_pred_boxes.append([])

            # GT boxes (already 0-indexed from loading)
            gt_boxes = [
                {"bbox": ann["bbox"], "class": ann["class"]}
                for ann in img_data["annotations"]
            ]
            all_gt_boxes.append(gt_boxes)

        # Compute mAP
        result = MetricsComputer.detection_ap(
            all_pred_boxes, all_gt_boxes,
            iou_threshold=0.5,
            class_names=task_spec.class_names,
        )

        result.metadata["n_images"] = len(self._images)
        result.metadata["conf_threshold"] = conf_threshold
        result.metadata["model"] = self.model_name
        result.metadata["n_queries"] = len(text_queries)

        logger.info(f"[OWLViT] mAP@50: {result.primary_metric:.4f}")
        return result
