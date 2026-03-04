"""Task-agnostic metrics computation for evaluation and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation outputs from any task."""

    primary_metric: float  # The main number (accuracy, mIoU, AP, etc.)
    primary_metric_name: str  # e.g. "top1_accuracy", "mIoU", "AP_rare"
    per_class_metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def worst_classes(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the n worst-performing classes."""
        sorted_cls = sorted(self.per_class_metrics.items(), key=lambda x: x[1])
        return sorted_cls[:n]

    def best_classes(self, n: int = 10) -> list[tuple[str, float]]:
        sorted_cls = sorted(self.per_class_metrics.items(), key=lambda x: x[1], reverse=True)
        return sorted_cls[:n]

    def class_accuracy_stats(self) -> dict[str, float]:
        if not self.per_class_metrics:
            return {}
        vals = list(self.per_class_metrics.values())
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    def confusion_pairs(self, n: int = 10) -> list[tuple[str, str, int]]:
        """Top n off-diagonal confusion pairs from the confusion matrix."""
        if self.confusion_matrix is None or not self.per_class_metrics:
            return []
        classes = list(self.per_class_metrics.keys())
        cm = self.confusion_matrix.copy()
        np.fill_diagonal(cm, 0)
        pairs = []
        flat = cm.flatten()
        top_idx = np.argsort(flat)[::-1][:n]
        for idx in top_idx:
            i, j = divmod(idx, cm.shape[1])
            if i < len(classes) and j < len(classes):
                pairs.append((classes[i], classes[j], int(cm[i, j])))
        return pairs


class MetricsComputer:
    """Compute standard metrics for different vision tasks."""

    @staticmethod
    def classification_accuracy(
        preds: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[list[str]] = None,
    ) -> EvalResult:
        """Top-1 classification accuracy with per-class breakdown."""
        correct = preds == targets
        overall = float(correct.mean())

        per_class = {}
        unique_classes = np.unique(targets)
        for c in unique_classes:
            mask = targets == c
            name = class_names[c] if class_names and c < len(class_names) else str(c)
            per_class[name] = float(correct[mask].mean()) if mask.sum() > 0 else 0.0

        # Confusion matrix
        n_classes = len(unique_classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for p, t in zip(preds, targets):
            cm[t, p] += 1

        # Confidence analysis
        return EvalResult(
            primary_metric=overall,
            primary_metric_name="top1_accuracy",
            per_class_metrics=per_class,
            confusion_matrix=cm,
            predictions=preds,
            ground_truth=targets,
        )

    @staticmethod
    def segmentation_iou(
        pred_masks: list[np.ndarray],
        gt_masks: list[np.ndarray],
        class_names: Optional[list[str]] = None,
    ) -> EvalResult:
        """Mean IoU for segmentation tasks."""
        ious = []
        per_class = {}

        for i, (pred, gt) in enumerate(zip(pred_masks, gt_masks)):
            intersection = np.logical_and(pred > 0, gt > 0).sum()
            union = np.logical_or(pred > 0, gt > 0).sum()
            iou = intersection / max(union, 1)
            ious.append(iou)
            name = class_names[i] if class_names and i < len(class_names) else f"sample_{i}"
            per_class[name] = float(iou)

        return EvalResult(
            primary_metric=float(np.mean(ious)),
            primary_metric_name="mIoU",
            per_class_metrics=per_class,
        )

    @staticmethod
    def detection_ap(
        pred_boxes: list[list[dict]],
        gt_boxes: list[list[dict]],
        iou_threshold: float = 0.5,
        class_names: Optional[list[str]] = None,
    ) -> EvalResult:
        """Average Precision for object detection.

        Each box dict: {"bbox": [x1,y1,x2,y2], "class": int, "score": float}
        """
        from collections import defaultdict

        all_scores = defaultdict(list)  # class -> [(score, is_tp)]
        gt_counts = defaultdict(int)

        for img_preds, img_gts in zip(pred_boxes, gt_boxes):
            for g in img_gts:
                gt_counts[g["class"]] += 1

            matched_gt = set()
            sorted_preds = sorted(img_preds, key=lambda x: x["score"], reverse=True)

            for p in sorted_preds:
                best_iou = 0.0
                best_gt_idx = -1
                for gi, g in enumerate(img_gts):
                    if g["class"] != p["class"] or gi in matched_gt:
                        continue
                    iou = MetricsComputer._box_iou(p["bbox"], g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    all_scores[p["class"]].append((p["score"], True))
                    matched_gt.add(best_gt_idx)
                else:
                    all_scores[p["class"]].append((p["score"], False))

        # Compute per-class AP
        per_class = {}
        aps = []
        for cls_id in sorted(set(list(all_scores.keys()) + list(gt_counts.keys()))):
            scores = sorted(all_scores.get(cls_id, []), key=lambda x: x[0], reverse=True)
            n_gt = gt_counts.get(cls_id, 0)
            if n_gt == 0:
                continue
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            for score, is_tp in scores:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
                recalls.append(tp_cumsum / n_gt)

            # AP via 11-point interpolation
            ap = 0.0
            for t in np.arange(0, 1.1, 0.1):
                p_at_r = [p for p, r in zip(precisions, recalls) if r >= t]
                ap += max(p_at_r) / 11 if p_at_r else 0.0

            name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            per_class[name] = float(ap)
            aps.append(ap)

        return EvalResult(
            primary_metric=float(np.mean(aps)) if aps else 0.0,
            primary_metric_name="mAP",
            per_class_metrics=per_class,
        )

    @staticmethod
    def _box_iou(box1: list, box2: list) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / max(union, 1e-6)
