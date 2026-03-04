"""Interactive segmentation task runner using SAM / SAM 2.

Supports point prompts, box prompts, and text prompts (via GroundingDINO + SAM).
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


class SAMSegmentationRunner(BaseTaskRunner):
    """Interactive segmentation using SAM 2 with multi-modal prompts.

    Supports:
    - Point prompts (positive + negative)
    - Box prompts
    - Text-guided (via GroundingDINO → box → SAM)
    - Adaptive prompt selection per image
    """

    def __init__(
        self,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
        image_dir: Optional[str | Path] = None,
        annotation_dir: Optional[str | Path] = None,
        dataset_loader: Optional[Any] = None,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self.image_dir = Path(image_dir) if image_dir else None
        self.annotation_dir = Path(annotation_dir) if annotation_dir else None
        self._dataset_loader = dataset_loader
        self._sam_model = None
        self._predictor = None
        self._images: list[np.ndarray] = []
        self._gt_masks: list[np.ndarray] = []
        self._image_names: list[str] = []

    def load_data(self, split: str = "val") -> None:
        if self._images:
            return

        if self._dataset_loader is not None:
            for item in self._dataset_loader:
                self._images.append(item["image"])
                self._gt_masks.append(item["mask"])
                self._image_names.append(item.get("name", f"img_{len(self._images)}"))
        elif self.image_dir and self.annotation_dir:
            import cv2
            for img_path in sorted(self.image_dir.glob("*.jpg")) + sorted(self.image_dir.glob("*.png")):
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._images.append(img)
                self._image_names.append(img_path.stem)

                mask_path = self.annotation_dir / img_path.with_suffix(".png").name
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    self._gt_masks.append(mask)
                else:
                    self._gt_masks.append(np.zeros(img.shape[:2], dtype=np.uint8))
        else:
            raise ValueError("Provide dataset_loader or image_dir + annotation_dir")

    def _ensure_model(self):
        if self._sam_model is not None:
            return
        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(self.device)
            self._sam_model = sam
            self._predictor = SamPredictor(sam)
        except ImportError:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                sam = build_sam2(self.model_type, self.sam_checkpoint)
                sam.to(self.device)
                self._sam_model = sam
                self._predictor = SAM2ImagePredictor(sam)
            except ImportError:
                raise ImportError(
                    "Install segment-anything or sam2: "
                    "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
                )

    def evaluate(self, prompts: dict[str, Any], task_spec: TaskSpec) -> EvalResult:
        self._ensure_model()
        self.load_data()

        point_strategy = prompts.get("point_strategy", {})
        box_strategy = prompts.get("box_strategy", {})

        pred_masks = []
        gt_masks_list = []

        for idx, (image, gt_mask) in enumerate(zip(self._images, self._gt_masks)):
            self._predictor.set_image(image)

            # Generate prompts based on strategy
            input_points, input_labels = self._generate_points(
                image, gt_mask, point_strategy
            )
            input_box = self._generate_box(image, gt_mask, box_strategy)

            # Run SAM
            kwargs = {}
            if input_points is not None:
                kwargs["point_coords"] = input_points
                kwargs["point_labels"] = input_labels
            if input_box is not None:
                kwargs["box"] = input_box

            if not kwargs:
                # Default: center point
                h, w = image.shape[:2]
                kwargs["point_coords"] = np.array([[w // 2, h // 2]])
                kwargs["point_labels"] = np.array([1])

            masks, scores, _ = self._predictor.predict(
                multimask_output=True, **kwargs
            )

            # Select best mask
            best_idx = scores.argmax()
            pred_mask = masks[best_idx]
            pred_masks.append(pred_mask.astype(np.uint8))
            gt_masks_list.append((gt_mask > 0).astype(np.uint8))

        # Compute metrics
        result = MetricsComputer.segmentation_iou(
            pred_masks, gt_masks_list, self._image_names
        )

        # Add J&F metric for video segmentation datasets
        result.metadata["n_images"] = len(self._images)
        result.metadata["prompt_strategy"] = {
            "points": point_strategy.get("placement", "center"),
            "n_points": point_strategy.get("initial_points", 1),
        }

        logger.info(f"[SAMRunner] mIoU: {result.primary_metric:.4f}")
        return result

    def _generate_points(
        self, image: np.ndarray, gt_mask: np.ndarray, strategy: dict
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate point prompts based on strategy."""
        placement = strategy.get("placement", "center")
        n_points = strategy.get("initial_points", 1)
        use_negative = strategy.get("negative_points", False)

        h, w = image.shape[:2]

        if placement == "center":
            # Center of the object (or image center if no GT)
            if gt_mask.any():
                ys, xs = np.where(gt_mask > 0)
                cx, cy = int(xs.mean()), int(ys.mean())
            else:
                cx, cy = w // 2, h // 2
            points = np.array([[cx, cy]])
            labels = np.array([1])

        elif placement == "grid":
            # Uniform grid
            grid_size = int(np.ceil(np.sqrt(n_points)))
            xs = np.linspace(w * 0.1, w * 0.9, grid_size).astype(int)
            ys = np.linspace(h * 0.1, h * 0.9, grid_size).astype(int)
            grid = np.array([[x, y] for y in ys for x in xs])[:n_points]
            points = grid
            labels = np.ones(len(grid), dtype=int)

        elif placement == "saliency":
            # Use image gradient magnitude as saliency
            import cv2
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
            grad = np.abs(grad)
            grad = grad / max(grad.max(), 1e-8)

            # Sample points from high-saliency regions
            flat = grad.flatten()
            probs = flat / flat.sum()
            indices = np.random.choice(len(flat), size=n_points, replace=False, p=probs)
            ys_sel, xs_sel = np.unravel_index(indices, grad.shape)
            points = np.column_stack([xs_sel, ys_sel])
            labels = np.ones(n_points, dtype=int)

        elif placement == "adaptive":
            # Combine center + saliency
            points_list = []
            labels_list = []

            # Always include center
            if gt_mask.any():
                ys, xs = np.where(gt_mask > 0)
                points_list.append([int(xs.mean()), int(ys.mean())])
            else:
                points_list.append([w // 2, h // 2])
            labels_list.append(1)

            # Add saliency points for remaining budget
            if n_points > 1:
                import cv2
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3))
                flat = grad.flatten()
                probs = flat / max(flat.sum(), 1e-8)
                extra = min(n_points - 1, 4)
                indices = np.random.choice(len(flat), size=extra, replace=False, p=probs)
                ys_sel, xs_sel = np.unravel_index(indices, grad.shape)
                for x, y in zip(xs_sel, ys_sel):
                    points_list.append([int(x), int(y)])
                    labels_list.append(1)

            points = np.array(points_list)
            labels = np.array(labels_list)
        else:
            return None, None

        # Add negative points
        if use_negative and gt_mask.any():
            bg_ys, bg_xs = np.where(gt_mask == 0)
            if len(bg_ys) > 0:
                neg_idx = np.random.choice(len(bg_ys), size=min(2, len(bg_ys)), replace=False)
                neg_points = np.column_stack([bg_xs[neg_idx], bg_ys[neg_idx]])
                neg_labels = np.zeros(len(neg_idx), dtype=int)
                points = np.vstack([points, neg_points])
                labels = np.concatenate([labels, neg_labels])

        return points, labels

    def _generate_box(
        self, image: np.ndarray, gt_mask: np.ndarray, strategy: dict
    ) -> Optional[np.ndarray]:
        """Generate box prompt from strategy."""
        if not strategy:
            return None

        source = strategy.get("source", "ground_truth")
        padding = strategy.get("padding_ratio", 0.1)

        if source == "ground_truth" and gt_mask.any():
            ys, xs = np.where(gt_mask > 0)
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()

            # Add padding
            w_pad = int((x2 - x1) * padding)
            h_pad = int((y2 - y1) * padding)
            h, w = image.shape[:2]
            x1 = max(0, x1 - w_pad)
            y1 = max(0, y1 - h_pad)
            x2 = min(w, x2 + w_pad)
            y2 = min(h, y2 + h_pad)

            return np.array([x1, y1, x2, y2])

        return None
