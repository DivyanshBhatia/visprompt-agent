#!/usr/bin/env python3
"""Run only the weight ablation for UCF-101 action recognition (skip baselines)."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_action_recognition import (
    UCF101_CLASSES, UCF101_FOLDER_TO_IDX,
    load_ucf101_frames, build_action_prompts_weighted,
    zero_shot_classify_weighted, generate_action_descriptions,
    ACTION_TEMPLATES,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="UCF-101 ablation only")
    parser.add_argument("--data-dir", type=str, default="/content/ucf101/test")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/action_recognition")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    import torch
    import open_clip
    from PIL import Image
    from visprompt.baselines import IMAGENET_TEMPLATES

    # Load data
    print(f"\n{'='*65}")
    print(f"  UCF-101 ABLATION ONLY")
    print(f"{'='*65}")

    images, labels, class_names = load_ucf101_frames(
        args.data_dir, split_file=None, max_per_class=args.max_per_class
    )

    # Load CLIP
    print(f"\nLoading CLIP: {args.clip_model}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained="openai", device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()

    # Encode images
    print(f"Encoding {len(images)} frames...")
    all_features = []
    for start in range(0, len(images), 128):
        end = min(start + 128, len(images))
        batch = []
        for img_arr in images[start:end]:
            img = Image.fromarray(img_arr)
            batch.append(preprocess(img))
        image_input = torch.stack(batch).to(args.device)
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
    test_features = torch.cat(all_features, dim=0)
    print(f"  Features: {test_features.shape}")

    # Templates baseline
    print(f"\n--- Templates-only ---")
    template_prompts = build_action_prompts_weighted(class_names, {}, 1.0, 0.0)
    templates_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        template_prompts, args.device, class_names
    )
    print(f"  Accuracy: {templates_acc:.4f}")

    # Generate descriptions
    print(f"\n--- Generating action descriptions ({args.llm}) ---")
    descriptions, desc_cost = generate_action_descriptions(
        class_names, args.llm, args.llm_provider
    )

    # Weight ablation
    WEIGHT_CONFIGS = [
        (1.0, 0.0, "100/0"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100"),
    ]

    print(f"\n--- Weight ablation ---")
    print(f"{'Config':<12} {'Accuracy':>10} {'Δ':>10}")
    print(f"{'-'*35}")

    best_acc = 0
    best_config = ""
    weights_results = []

    for base_w, desc_w, label in WEIGHT_CONFIGS:
        prompts = build_action_prompts_weighted(class_names, descriptions, base_w, desc_w)
        acc = zero_shot_classify_weighted(
            model, tokenizer, test_features, labels,
            prompts, args.device, class_names
        )
        delta = acc - templates_acc
        marker = " ← best" if acc > best_acc and desc_w > 0 else ""
        print(f"  {label:<10} {acc:>10.4f} {delta:>+9.4f}{marker}")

        weights_results.append({
            "config": label, "base_weight": base_w, "desc_weight": desc_w,
            "accuracy": float(acc), "delta": float(delta),
        })
        if acc > best_acc:
            best_acc = acc
            best_config = label

    print(f"\n  Best: {best_config} → {best_acc:.4f} (Δ {best_acc - templates_acc:+.4f})")

    # Save
    results = {
        "dataset": "ucf101",
        "clip_model": args.clip_model,
        "n_classes": len(np.unique(labels)),
        "n_samples": len(labels),
        "llm": args.llm,
        "templates_accuracy": float(templates_acc),
        "baselines": {
            "single_template": 0.7248,
            "80_template_ensemble": float(templates_acc),
            "cupl_ensemble": 0.7225,
            "dclip": 0.7178,
            "clip_enhance": 0.7178,
            "class_name_only": 0.6996,
            "cupl_desc_only": 0.6996,
            "waffle_clip": 0.6938,
        },
        "best_accuracy": float(best_acc),
        "best_config": best_config,
        "best_delta": float(best_acc - templates_acc),
        "weights": weights_results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ucf101_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
