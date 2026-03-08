#!/usr/bin/env python3
"""TLAC baseline: Two-stage LMM Augmented CLIP (Munir et al., CVPR 2025 Workshop).

TLAC sends each test image to an LMM (we use GPT-4o-mini for cost), gets a
description, then classifies by matching the description to class names via
CLIP's text encoder.

Two variants:
  SLAC: LMM describes image → CLIP matches description to class names
  TLAC: LMM picks best class from list → CLIP verifies via text similarity

Usage:
    python scripts/run_tlac_baseline.py --dataset flowers102 --clip-model ViT-L/14
    python scripts/run_tlac_baseline.py --dataset dtd --clip-model ViT-L/14 --max-images 500
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def encode_image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    from PIL import Image
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def slac_describe_image(client, image_b64, model="gpt-4o-mini"):
    """SLAC: Ask LMM to describe what's in the image."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What object, animal, texture, or scene is shown in this image? Give a short, specific answer in 1-2 sentences focusing on the main subject."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}},
            ]
        }],
        max_tokens=100,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip(), response


def tlac_classify_image(client, image_b64, class_names, model="gpt-4o-mini"):
    """TLAC: Ask LMM to pick the most relevant class from a list."""
    # For large class lists, batch into groups
    class_str = ", ".join(class_names)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Which of the following categories best describes the main subject in this image?\n\nCategories: {class_str}\n\nRespond with ONLY the exact category name, nothing else."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "low"}},
            ]
        }],
        max_tokens=50,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip(), response

def main():
    parser = argparse.ArgumentParser(description="TLAC/SLAC baseline")
    parser.add_argument("--dataset", type=str, default="flowers102")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lmm-model", type=str, default="gpt-4o",
                        help="Vision LMM model (gpt-4o, gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit images for cost control")
    parser.add_argument("--mode", type=str, default="both", choices=["slac", "tlac", "both"])
    parser.add_argument("--output-dir", type=str, default="experiments/tlac_baseline")
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--batch-delay", type=float, default=0.1,
                        help="Delay between API calls (rate limiting)")
    # Dummy args for compatibility
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--annotation-dir", type=str, default=None)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--sam-checkpoint", type=str, default=None)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str, default=None)
    parser.add_argument("--gdino-checkpoint", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    from openai import OpenAI
    import torch
    import open_clip
    from scripts.run import build_task_spec, _load_pil_dataset

    client = OpenAI()

    print(f"\n{'='*65}")
    print(f"  TLAC BASELINE — {args.dataset.upper()}")
    print(f"  LMM: {args.lmm_model}, CLIP: {args.clip_model}")
    print(f"{'='*65}")

    # Load task spec for class names
    task_spec = build_task_spec(args)
    class_names = task_spec.class_names

    # Load images directly
    if args.dataset == "cifar100":
        import torchvision
        dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir or "./data", train=False, download=True)
        n_val = min(args.val_size or 10000, len(dataset))
        indices = np.random.RandomState(42).permutation(len(dataset))[:n_val]
        images = np.array(dataset.data)[indices]
        labels = np.array(dataset.targets)[indices]
    elif args.dataset == "cifar10":
        import torchvision
        dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir or "./data", train=False, download=True)
        n_val = min(args.val_size or 10000, len(dataset))
        indices = np.random.RandomState(42).permutation(len(dataset))[:n_val]
        images = np.array(dataset.data)[indices]
        labels = np.array(dataset.targets)[indices]
    elif args.dataset == "flowers102":
        images, labels = _load_pil_dataset("Flowers102", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "dtd":
        images, labels = _load_pil_dataset("DTD", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "oxford_pets":
        images, labels = _load_pil_dataset("OxfordIIITPet", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "food101":
        images, labels = _load_pil_dataset("Food101", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "eurosat":
        images, labels = _load_pil_dataset("EuroSAT", args, split=None,
                                           dataset_kwargs={})
    elif args.dataset == "fgvc_aircraft":
        images, labels = _load_pil_dataset("FGVCAircraft", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "caltech101":
        images, labels = _load_pil_dataset("Caltech101", args, split=None,
                                           dataset_kwargs={})
    elif args.dataset == "country211":
        images, labels = _load_pil_dataset("Country211", args, split="test",
                                           dataset_kwargs={"split": "test"})
    elif args.dataset == "ucf101":
        from scripts.run_action_recognition import (
            load_ucf101_frames, UCF101_CLASSES
        )
        if not args.data_dir:
            raise ValueError("UCF-101 requires --data-dir pointing to the test folder")
        images, labels, _ = load_ucf101_frames(
            args.data_dir, split_file=None, max_per_class=None
        )
        # Override class_names from action recognition module
        class_names = UCF101_CLASSES
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_images and args.max_images < len(images):
        # Stratified subsample
        np.random.seed(42)
        indices = []
        for c in range(len(class_names)):
            c_indices = np.where(labels == c)[0]
            n_take = max(1, int(args.max_images * len(c_indices) / len(labels)))
            indices.extend(np.random.choice(c_indices, min(n_take, len(c_indices)), replace=False))
        indices = sorted(indices)[:args.max_images]
        images = [images[i] for i in indices]
        labels = labels[indices]
        print(f"  Subsampled to {len(images)} images")

    n_images = len(images)
    print(f"  Dataset: {args.dataset}, {n_images} images, {len(class_names)} classes")
    
    # Estimate cost
    # GPT-4o pricing per 1M tokens
    INPUT_COST_PER_M = 2.50
    OUTPUT_COST_PER_M = 10.00
    
    def calc_cost(response):
        """Calculate actual cost from API response usage."""
        usage = response.usage
        input_cost = (usage.prompt_tokens / 1_000_000) * INPUT_COST_PER_M
        output_cost = (usage.completion_tokens / 1_000_000) * OUTPUT_COST_PER_M
        return input_cost + output_cost

    print(f"  (Cost will be computed from actual API usage)")
    
    results = {
        "dataset": args.dataset,
        "clip_model": args.clip_model,
        "lmm_model": args.lmm_model,
        "n_images": n_images,
        "n_classes": len(class_names),
    }

    # ── SLAC: Describe then match ────────────────────────────────
    if args.mode in ["slac", "both"]:
        print(f"\n--- SLAC: Describe → CLIP match ---")
        
        # Load CLIP for text matching
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained="openai", device=args.device
        )
        tokenizer = open_clip.get_tokenizer(args.clip_model)
        model.eval()
        
        # Pre-encode class name texts
        class_texts = [f"a photo of a {cn.replace('_', ' ').lower()}" for cn in class_names]
        with torch.no_grad():
            class_tokens = tokenizer(class_texts).to(args.device)
            class_features = model.encode_text(class_tokens)
            class_features = class_features / class_features.norm(dim=-1, keepdim=True)
        
        slac_correct = 0
        slac_total = 0
        slac_cost = 0
        
        for i, img in enumerate(images):
            try:
                img_b64 = encode_image_to_base64(img)
                description, resp = slac_describe_image(client, img_b64, args.lmm_model)
                
                # Encode description with CLIP
                with torch.no_grad():
                    desc_tokens = tokenizer([description]).to(args.device)
                    desc_features = model.encode_text(desc_tokens)
                    desc_features = desc_features / desc_features.norm(dim=-1, keepdim=True)
                
                # Match to class names
                sims = (desc_features @ class_features.T).squeeze(0)
                pred = sims.argmax().item()
                
                if pred == labels[i]:
                    slac_correct += 1
                slac_total += 1
                slac_cost += calc_cost(resp)
                
                if (i + 1) % 100 == 0 or i == n_images - 1:
                    acc = slac_correct / slac_total
                    print(f"  [{i+1}/{n_images}] SLAC acc: {acc:.4f} (cost: ${slac_cost:.2f})")
                
                time.sleep(args.batch_delay)
                
            except Exception as e:
                logger.warning(f"  Image {i} failed: {e}")
                slac_total += 1
                time.sleep(1)
        
        slac_acc = slac_correct / slac_total
        cost_per_img = slac_cost / slac_total if slac_total > 0 else 0
        print(f"\n  SLAC Final: {slac_acc:.4f} ({slac_correct}/{slac_total}), cost: ${slac_cost:.4f} (${cost_per_img:.6f}/image)")
        results["slac_accuracy"] = float(slac_acc)
        results["slac_cost"] = float(slac_cost)
        results["slac_cost_per_image"] = float(cost_per_img)

    # ── TLAC: Direct class selection ──────────────────────────────
    if args.mode in ["tlac", "both"]:
        print(f"\n--- TLAC: Direct class selection ---")
        
        tlac_correct = 0
        tlac_total = 0
        tlac_cost = 0
        
        for i, img in enumerate(images):
            try:
                img_b64 = encode_image_to_base64(img)
                prediction, resp = tlac_classify_image(client, img_b64, class_names, args.lmm_model)
                
                # Match prediction to class name (fuzzy)
                pred_lower = prediction.lower().strip().strip('"').strip("'")
                pred_idx = -1
                for j, cn in enumerate(class_names):
                    cn_lower = cn.replace("_", " ").lower()
                    if cn_lower == pred_lower or cn_lower in pred_lower or pred_lower in cn_lower:
                        pred_idx = j
                        break
                
                if pred_idx == -1:
                    # Try partial matching
                    best_overlap = 0
                    for j, cn in enumerate(class_names):
                        cn_words = set(cn.replace("_", " ").lower().split())
                        pred_words = set(pred_lower.split())
                        overlap = len(cn_words & pred_words)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            pred_idx = j
                
                if pred_idx == labels[i]:
                    tlac_correct += 1
                tlac_total += 1
                tlac_cost += calc_cost(resp)
                
                if (i + 1) % 100 == 0 or i == n_images - 1:
                    acc = tlac_correct / tlac_total
                    print(f"  [{i+1}/{n_images}] TLAC acc: {acc:.4f} (cost: ${tlac_cost:.2f})")
                
                time.sleep(args.batch_delay)
                
            except Exception as e:
                logger.warning(f"  Image {i} failed: {e}")
                tlac_total += 1
                time.sleep(1)
        
        tlac_acc = tlac_correct / tlac_total
        cost_per_img = tlac_cost / tlac_total if tlac_total > 0 else 0
        print(f"\n  TLAC Final: {tlac_acc:.4f} ({tlac_correct}/{tlac_total}), cost: ${tlac_cost:.4f} (${cost_per_img:.6f}/image)")
        results["tlac_accuracy"] = float(tlac_acc)
        results["tlac_cost"] = float(tlac_cost)
        results["tlac_cost_per_image"] = float(cost_per_img)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tlac_{args.dataset}_{args.lmm_model}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Summary comparison
    print(f"\n{'='*50}")
    print(f"  COMPARISON ({args.dataset})")
    print(f"{'='*50}")
    if "slac_accuracy" in results:
        print(f"  SLAC ({args.lmm_model}):     {results['slac_accuracy']:.4f}  (${results['slac_cost']:.2f})")
    if "tlac_accuracy" in results:
        print(f"  TLAC ({args.lmm_model}):     {results['tlac_accuracy']:.4f}  (${results['tlac_cost']:.2f})")
    print(f"  Note: Per-image LMM cost. NETRA cost is per-class (one-time).")


if __name__ == "__main__":
    main()
