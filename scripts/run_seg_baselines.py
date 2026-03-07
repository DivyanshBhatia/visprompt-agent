#!/usr/bin/env python3
"""Run CLIPSeg segmentation baselines on Pascal VOC 2012.

Example:
    python scripts/run_seg_baselines.py --max-images 200
    python scripts/run_seg_baselines.py --max-images 200 --run-ablation
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "dining table", "dog",
    "horse", "motorbike", "person", "potted plant", "sheep",
    "sofa", "train", "tv monitor",
]

# Map class names to VOC label indices (0=background, 1-20=classes, 255=ignore)
VOC_CLASS_TO_LABEL = {cls: idx for idx, cls in enumerate(VOC_CLASSES)}

# 80 ImageNet templates (subset — 10 most relevant for segmentation)
SEG_TEMPLATES = [
    "a photo of a {}.",
    "a photo of a {} in a scene.",
    "a {} in the image.",
    "a photo showing a {}.",
    "an image containing a {}.",
    "a clear photo of a {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of a large {}.",
    "a photo of a small {}.",
]


def load_voc_dataset(data_dir, max_images=None):
    """Load Pascal VOC 2012 val set for segmentation."""
    import torchvision

    data_dir = Path(data_dir) if data_dir else Path("./data")

    logger.info("Downloading/loading Pascal VOC 2012...")
    dataset = torchvision.datasets.VOCSegmentation(
        root=str(data_dir),
        year="2012",
        image_set="val",
        download=True,
    )

    image_list = []
    n_total = len(dataset)
    indices = np.random.RandomState(42).permutation(n_total)
    if max_images:
        indices = indices[:max_images]

    for idx in indices:
        img, mask = dataset[idx]
        # Save temp references — dataset returns PIL images
        # We need paths for CLIPSeg runner
        img_path = dataset.images[idx]
        mask_path = dataset.masks[idx]

        image_list.append({
            "image_path": img_path,
            "mask_path": mask_path,
            "class_to_label": VOC_CLASS_TO_LABEL,
        })

    logger.info(f"Loaded {len(image_list)}/{n_total} VOC val images")
    return image_list


def generate_seg_descriptions(class_names, llm_model, llm_provider):
    """Generate descriptions for segmentation classes."""
    from visprompt.utils.llm import CostTracker, LLMClient

    cost_tracker = CostTracker()
    llm = LLMClient(model=llm_model, provider=llm_provider, cost_tracker=cost_tracker)

    prompt = (
        f"Generate 10 short visual descriptions for each object class below.\n"
        f"These will be used for image segmentation — focus on what the object LOOKS like.\n"
        f"Format: \"a {{class_name}}, {{visual description}}\"\n"
        f"Keep under 15 words. Focus on shape, color, texture, typical appearance.\n\n"
        f"Classes: {', '.join(class_names)}\n\n"
        f'Respond ONLY with JSON: {{"class_name": ["desc1", "desc2", ...]}}\n'
    )

    try:
        result = llm.call_json(
            prompt=prompt,
            system="Generate visual descriptions for image segmentation.",
            agent_name="seg_descriptions",
        )
        descriptions = {}
        for cls in class_names:
            descs = result.get(cls, result.get(cls.replace("_", " "), []))
            if isinstance(descs, list) and descs:
                descriptions[cls] = descs
            else:
                descriptions[cls] = [f"a {cls}"]
        return descriptions, cost_tracker.summary()
    except Exception as e:
        logger.warning(f"Description generation failed: {e}")
        return {cls: [f"a {cls}"] for cls in class_names}, cost_tracker.summary()


def build_prompts(class_names, descriptions, base_weight, desc_weight, templates=None):
    """Build prompt dict with group-normalized weights."""
    if templates is None:
        templates = SEG_TEMPLATES

    prompts_per_class = {}
    for cls_name in class_names:
        base_prompts = [t.format(cls_name) for t in templates]
        desc_prompts = descriptions.get(cls_name, [])

        n_base = len(base_prompts)
        n_desc = len(desc_prompts)

        if n_desc > 0 and desc_weight > 0:
            per_base = base_weight / n_base if n_base > 0 else 0
            per_desc = desc_weight / n_desc if n_desc > 0 else 0
            all_prompts = base_prompts + desc_prompts
            all_weights = [per_base] * n_base + [per_desc] * n_desc
        elif n_base > 0:
            all_prompts = base_prompts
            all_weights = [1.0 / n_base] * n_base
        else:
            all_prompts = [cls_name]
            all_weights = [1.0]

        prompts_per_class[cls_name] = {
            "prompts": all_prompts,
            "weights": all_weights,
        }

    return {"prompts_per_class": prompts_per_class}


def main():
    parser = argparse.ArgumentParser(description="CLIPSeg segmentation baselines")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--max-images", type=int, default=200,
                        help="Max images (200 for fast, None for full val)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--run-ablation", action="store_true",
                        help="Also run weight ablation")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Skip baselines, run only weight ablation")
    parser.add_argument("--output-dir", type=str, default="experiments/segmentation")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Load dataset
    image_list = load_voc_dataset(args.data_dir, args.max_images)

    # Use foreground classes only (skip background)
    fg_classes = VOC_CLASSES[1:]  # 20 foreground classes

    from visprompt.tasks.segmentation_clipseg import CLIPSegRunner
    from visprompt.task_spec import TaskSpec

    task_spec = TaskSpec(
        task_type="segmentation",
        dataset_name="voc2012",
        class_names=fg_classes,
        num_classes=20,
        domain="natural",
        foundation_model="clipseg",
        prompt_modality="text",
        metric_name="mIoU",
    )

    runner = CLIPSegRunner(
        device=args.device,
        image_list=image_list,
        class_names=fg_classes,
    )

    results = {}

    if not args.ablation_only:
        # ── Baseline 1: Single class name ─────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  BASELINE 1: Class name only")
        print(f"{'='*60}")
        t0 = time.time()
        prompts_simple = {"prompts_per_class": {
            cls: {"prompts": [cls], "weights": [1.0]} for cls in fg_classes
        }}
        result = runner.evaluate(prompts_simple, task_spec)
        print(f"  mIoU: {result.primary_metric:.4f} ({time.time()-t0:.0f}s)")
        results["class_name_only"] = result.primary_metric

        # ── Baseline 2: "a photo of a {class}" ────────────────────────────
        print(f"\n{'='*60}")
        print(f"  BASELINE 2: 'a photo of a {{class}}'")
        print(f"{'='*60}")
        t0 = time.time()
        prompts_photo = {"prompts_per_class": {
            cls: {"prompts": [f"a photo of a {cls}"], "weights": [1.0]} for cls in fg_classes
        }}
        result = runner.evaluate(prompts_photo, task_spec)
        print(f"  mIoU: {result.primary_metric:.4f} ({time.time()-t0:.0f}s)")
        results["photo_template"] = result.primary_metric

    # ── Generate descriptions (needed for baselines 4,5 and ablation) ──
    descriptions = None
    if not args.ablation_only or args.run_ablation or args.ablation_only:
        print(f"\n{'='*60}")
        print(f"  Generating LLM descriptions")
        print(f"{'='*60}")
        descriptions, desc_cost = generate_seg_descriptions(
            fg_classes, args.llm, args.llm_provider
        )
        print(f"  Generated descriptions for {len(descriptions)} classes "
              f"(cost: ${desc_cost.get('total_cost_usd', 0):.4f})")

    if not args.ablation_only:
        # ── Baseline 3: 10-template ensemble ──────────────────────────
        print(f"\n{'='*60}")
        print(f"  BASELINE 3: 10-template ensemble")
        print(f"{'='*60}")
        t0 = time.time()
        prompts_tmpl = build_prompts(fg_classes, {}, 1.0, 0.0)
        result = runner.evaluate(prompts_tmpl, task_spec)
        print(f"  mIoU: {result.primary_metric:.4f} ({time.time()-t0:.0f}s)")
        results["template_ensemble"] = result.primary_metric

        # ── Baseline 4: LLM descriptions only ────────────────────────
        print(f"\n{'='*60}")
        print(f"  BASELINE 4: LLM descriptions only")
        print(f"{'='*60}")
        t0 = time.time()
        prompts_desc = build_prompts(fg_classes, descriptions, 0.0, 1.0)
        result = runner.evaluate(prompts_desc, task_spec)
        print(f"  mIoU: {result.primary_metric:.4f} ({time.time()-t0:.0f}s)")
        results["llm_descriptions"] = result.primary_metric

        # ── Baseline 5: Templates + descriptions (equal weight) ──────
        print(f"\n{'='*60}")
        print(f"  BASELINE 5: Templates + descriptions (50/50)")
        print(f"{'='*60}")
        t0 = time.time()
        prompts_combo = build_prompts(fg_classes, descriptions, 0.5, 0.5)
        result = runner.evaluate(prompts_combo, task_spec)
        print(f"  mIoU: {result.primary_metric:.4f} ({time.time()-t0:.0f}s)")
        results["combo_50_50"] = result.primary_metric

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  SEGMENTATION BASELINE COMPARISON (Pascal VOC 2012)")
        print(f"{'='*60}")
        print(f"{'Method':<30} {'mIoU':>10}")
        print(f"{'-'*45}")
        for name, miou in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:<28} {miou:>10.4f}")
        print(f"{'='*60}")

    # ── Weight ablation ───────────────────────────────────────────────
    if args.run_ablation or args.ablation_only:
        print(f"\n{'='*60}")
        print(f"  WEIGHT ABLATION")
        print(f"{'='*60}")
        print(f"{'Config':<30} {'mIoU':>10}")
        print(f"{'-'*45}")

        ablation_results = []

        for base_w, desc_w, label in [
            (1.0, 0.0, "100/0 (templates only)"),
            (0.8, 0.2, "80/20"),
            (0.7, 0.3, "70/30"),
            (0.5, 0.5, "50/50"),
            (0.3, 0.7, "30/70"),
            (0.0, 1.0, "0/100 (descriptions only)"),
        ]:
            t0 = time.time()
            prompts = build_prompts(fg_classes, descriptions, base_w, desc_w)
            result = runner.evaluate(prompts, task_spec)
            miou = result.primary_metric
            print(f"  {label:<28} {miou:>10.4f}  ({time.time()-t0:.0f}s)")
            ablation_results.append({
                "base_weight": base_w, "desc_weight": desc_w,
                "label": label, "mIoU": miou,
            })

        best = max(ablation_results, key=lambda x: x["mIoU"])
        print(f"\n  Best: {best['label']} → {best['mIoU']:.4f}")
        results["ablation"] = ablation_results

    # Save
    output_path = Path(args.output_dir) / "seg_baselines_voc2012.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
