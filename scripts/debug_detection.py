#!/usr/bin/env python3
"""Debug detection: inspect raw model outputs, box format, and class mapping."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model
    model_name = "google/owlv2-base-patch16-ensemble"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(args.device)
    model.eval()

    # Load annotations
    with open(args.annotation_file) as f:
        coco = json.load(f)

    categories = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    class_names = [c["name"] for c in categories]

    img_id_to_info = {img["id"]: img for img in coco["images"]}
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Pick images with lots of annotations
    img_ids = sorted(img_id_to_anns.keys(), key=lambda x: -len(img_id_to_anns[x]))

    print(f"\nClass names ({len(class_names)}): {class_names[:10]}...")
    print(f"Category ID mapping: {list(cat_id_to_idx.items())[:5]}...")

    text_queries = class_names  # Simple class names

    for img_id in img_ids[:args.n_images]:
        img_info = img_id_to_info[img_id]
        img_path = Path(args.data_dir) / img_info["file_name"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # GT boxes
        anns = img_id_to_anns.get(img_id, [])
        gt_boxes = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            gt_boxes.append({
                "class": cat_id_to_idx[ann["category_id"]],
                "name": cat_id_to_name[ann["category_id"]],
                "bbox": [x, y, x + bw, y + bh],
            })

        print(f"\n{'='*70}")
        print(f"Image: {img_info['file_name']} ({w}x{h})")
        print(f"GT boxes ({len(gt_boxes)}):")
        for gt in gt_boxes[:5]:
            print(f"  {gt['name']:20s} bbox={[f'{v:.0f}' for v in gt['bbox']]}")

        # Run model
        inputs = processor(
            text=[text_queries], images=image,
            return_tensors="pt", padding=True, truncation=True,
        ).to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        boxes = outputs.pred_boxes[0]

        print(f"\nRaw outputs:")
        print(f"  logits shape: {logits.shape}")
        print(f"  boxes shape: {boxes.shape}")
        print(f"  boxes range: [{boxes.min():.3f}, {boxes.max():.3f}]")
        print(f"  boxes[0] sample: {boxes[0].tolist()}")

        # Check box format by looking at value ranges
        col_ranges = []
        for c in range(4):
            col_ranges.append(f"col{c}: [{boxes[:, c].min():.3f}, {boxes[:, c].max():.3f}]")
        print(f"  Per-column ranges: {', '.join(col_ranges)}")

        # Get top detections
        probs = logits.sigmoid()
        scores, labels = probs.max(dim=-1)

        for threshold in [0.3, 0.2, 0.1, 0.05]:
            mask = scores > threshold
            n_det = mask.sum().item()
            print(f"\n  Threshold={threshold}: {n_det} detections")

            if n_det > 0 and n_det < 50:
                det_scores = scores[mask]
                det_labels = labels[mask]
                det_boxes = boxes[mask]

                # Try cx,cy,w,h -> x1,y1,x2,y2 conversion
                cx, cy, bw, bh = det_boxes.unbind(-1)
                x1_cxcy = (cx - bw/2) * w
                y1_cxcy = (cy - bh/2) * h
                x2_cxcy = (cx + bw/2) * w
                y2_cxcy = (cy + bh/2) * h

                # Try direct x1,y1,x2,y2 (already normalized)
                x1_direct = det_boxes[:, 0] * w
                y1_direct = det_boxes[:, 1] * h
                x2_direct = det_boxes[:, 2] * w
                y2_direct = det_boxes[:, 3] * h

                print(f"  Top 5 detections:")
                for i in range(min(5, n_det)):
                    cls_name = class_names[det_labels[i]] if det_labels[i] < len(class_names) else "?"
                    score = det_scores[i].item()
                    box_cxcy = [x1_cxcy[i].item(), y1_cxcy[i].item(), x2_cxcy[i].item(), y2_cxcy[i].item()]
                    box_direct = [x1_direct[i].item(), y1_direct[i].item(), x2_direct[i].item(), y2_direct[i].item()]
                    print(f"    {cls_name:20s} score={score:.3f}")
                    print(f"      cx,cy,w,h->: [{box_cxcy[0]:.0f}, {box_cxcy[1]:.0f}, {box_cxcy[2]:.0f}, {box_cxcy[3]:.0f}]")
                    print(f"      direct->:    [{box_direct[0]:.0f}, {box_direct[1]:.0f}, {box_direct[2]:.0f}, {box_direct[3]:.0f}]")

                # Compute IoU with GT for both formats
                if gt_boxes and n_det > 0:
                    print(f"\n  IoU check (top detection vs first GT '{gt_boxes[0]['name']}' at {[f'{v:.0f}' for v in gt_boxes[0]['bbox']]}):")
                    gt_b = gt_boxes[0]["bbox"]
                    for fmt, x1v, y1v, x2v, y2v in [
                        ("cx,cy,w,h", x1_cxcy[0].item(), y1_cxcy[0].item(), x2_cxcy[0].item(), y2_cxcy[0].item()),
                        ("direct", x1_direct[0].item(), y1_direct[0].item(), x2_direct[0].item(), y2_direct[0].item()),
                    ]:
                        inter_x1 = max(x1v, gt_b[0])
                        inter_y1 = max(y1v, gt_b[1])
                        inter_x2 = min(x2v, gt_b[2])
                        inter_y2 = min(y2v, gt_b[3])
                        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                        area_pred = max(0, x2v - x1v) * max(0, y2v - y1v)
                        area_gt = (gt_b[2] - gt_b[0]) * (gt_b[3] - gt_b[1])
                        union = area_pred + area_gt - inter
                        iou = inter / union if union > 0 else 0
                        print(f"    {fmt}: IoU = {iou:.3f}")


if __name__ == "__main__":
    main()
