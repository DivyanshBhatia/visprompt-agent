#!/usr/bin/env python3
"""Per-class action recognition analysis: equipment vs motion classes.

Computes per-class accuracy delta (NETRA vs templates) and groups by action type
to test whether equipment-specific actions benefit more than motion-specific ones.

Usage:
    python scripts/run_action_perclass.py \
        --dataset hmdb51 --data-dir /content/hmdb51/HMDB51 \
        --clip-model ViT-L/14 --llm gpt-4o
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.utils.llm import LLMClient

ACTION_TEMPLATES = [
    "a photo of a person {}",
    "a video frame of a person {}",
    "a photo of someone {}",
    "an image of a person {}",
    "a still frame showing {}",
    "a person is seen {}",
    "a photo showing the action of {}",
    "a video screenshot of {}",
    "an example of {}",
    "a demonstration of {}",
]

# Action type categories (manually curated for UCF-101, HMDB-51, K400)
EQUIPMENT_ACTIONS = {
    # UCF-101
    "playing sitar", "playing guitar", "playing piano", "playing cello",
    "playing violin", "playing flute", "playing tabla", "playing dhol",
    "archery", "bowling", "billiards", "fencing", "rowing",
    "biking", "horse riding", "skiing", "surfing", "kayaking",
    "drumming", "typing", "knitting", "writing on board",
    "baseball pitch", "basketball", "cricket bowling", "cricket shot",
    "tennis swing", "golf swing", "table tennis shot",
    "hammering", "cutting in kitchen", "mopping floor",
    # HMDB-51
    "shoot gun", "draw sword", "sword exercise", "shoot bow",
    "ride bike", "ride horse", "swing baseball",
    "brush hair", "pour", "eat", "drink", "smoke",
    # K400
    "playing accordion", "playing bagpipes", "playing bass guitar",
    "playing drums", "playing harp", "playing keyboard",
    "playing organ", "playing trumpet", "playing ukulele",
    "playing xylophone", "bowling", "dribbling basketball",
    "dunking basketball", "golf driving", "ice skating",
    "juggling balls", "skateboarding", "skiing crosscountry",
    "skiing slalom", "snowboarding", "surfing water",
}

MOTION_ACTIONS = {
    # UCF-101
    "walking", "running", "jogging", "push ups", "pull ups",
    "sit up", "lunges", "jumping jack", "handstand pushups",
    "hand stand walking", "cartwheel", "backflip",
    "front crawl", "breast stroke", "body weight squats",
    "floor gymnastics", "balance beam", "uneven bars",
    "still rings", "pommel horse", "parallel bars",
    "high jump", "long jump", "pole vault",
    "salsa spin", "swing", "tai chi",
    # HMDB-51
    "walk", "run", "climb", "climb stairs", "jump",
    "fall floor", "handstand", "cartwheel", "backhand flip",
    "pushup", "pullup", "situp", "somersault",
    "kick", "punch", "clap", "wave",
    # K400
    "running on treadmill", "walking the dog", "jogging",
    "crawling baby", "dancing ballet", "dancing gangnam style",
    "doing aerobics", "exercising arm", "high kick",
    "jumping into pool", "lunge", "pull ups", "push up",
    "squat", "stretching arm", "stretching leg",
    "tai chi", "triple jump", "vault",
}


def classify_action(class_name):
    """Classify an action as equipment, motion, or other."""
    cn_lower = class_name.lower().replace("_", " ")
    for eq in EQUIPMENT_ACTIONS:
        if eq in cn_lower or cn_lower in eq:
            return "equipment"
    for mo in MOTION_ACTIONS:
        if mo in cn_lower or cn_lower in mo:
            return "motion"
    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ucf101", "hmdb51", "kinetics400"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--beta", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"\n{'='*70}")
    print(f"  PER-CLASS ACTION ANALYSIS — {args.dataset.upper()}")
    print(f"  Equipment vs Motion vs Other")
    print(f"{'='*70}\n")

    # Import video loader
    from scripts.run_video_benchmarks import (
        load_video_dataset, load_clip_model, encode_images,
        build_action_templates, build_netra_prompts,
        generate_action_descriptions, encode_text_prompts,
    )

    # Load data
    images, labels, classnames = load_video_dataset(
        args.data_dir, args.dataset, args.val_size
    )
    print(f"  {len(classnames)} classes, {len(images)} frames\n")

    # Load CLIP
    model, preprocess, tokenizer = load_clip_model(args.clip_model, args.device)

    # Encode images
    print("Encoding frames...")
    image_features = encode_images(model, preprocess, images, args.device)

    # Generate descriptions
    cache = Path(args.output_dir) / f"action_descriptions_{args.dataset}_{args.llm}.json"
    descriptions = generate_action_descriptions(
        classnames, args.llm, args.llm_provider, cache
    )

    # --- Templates baseline: per-class accuracy ---
    ppc_tmpl = build_action_templates(classnames)
    emb_tmpl = encode_text_prompts(model, tokenizer, ppc_tmpl, classnames, args.device)
    
    sims_tmpl = image_features.to(emb_tmpl.device) @ emb_tmpl.T
    preds_tmpl = sims_tmpl.argmax(dim=1).cpu().numpy()
    labels_arr = np.array(labels)

    tmpl_perclass = {}
    for c_idx, cn in enumerate(classnames):
        mask = labels_arr == c_idx
        if mask.sum() == 0:
            continue
        acc = (preds_tmpl[mask] == c_idx).mean()
        tmpl_perclass[cn] = float(acc)

    # --- NETRA: per-class accuracy ---
    ppc_netra = build_netra_prompts(classnames, descriptions, args.alpha, args.beta)
    emb_netra = encode_text_prompts(model, tokenizer, ppc_netra, classnames, args.device)

    sims_netra = image_features.to(emb_netra.device) @ emb_netra.T
    preds_netra = sims_netra.argmax(dim=1).cpu().numpy()

    netra_perclass = {}
    for c_idx, cn in enumerate(classnames):
        mask = labels_arr == c_idx
        if mask.sum() == 0:
            continue
        acc = (preds_netra[mask] == c_idx).mean()
        netra_perclass[cn] = float(acc)

    # --- Compute deltas and group by type ---
    results = []
    for cn in classnames:
        if cn not in tmpl_perclass or cn not in netra_perclass:
            continue
        action_type = classify_action(cn)
        delta = netra_perclass[cn] - tmpl_perclass[cn]
        results.append({
            "class": cn,
            "type": action_type,
            "template_acc": tmpl_perclass[cn],
            "netra_acc": netra_perclass[cn],
            "delta": delta,
        })

    # Sort by delta
    results.sort(key=lambda x: x["delta"], reverse=True)

    # Group stats
    groups = {"equipment": [], "motion": [], "other": []}
    for r in results:
        groups[r["type"]].append(r["delta"])

    print(f"\n{'='*70}")
    print(f"  GROUP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Type':<12} {'Count':>6} {'Mean Δ':>10} {'Median Δ':>10} {'Positive':>10}")
    print(f"  {'-'*50}")
    for gtype in ["equipment", "motion", "other"]:
        deltas = groups[gtype]
        if len(deltas) == 0:
            print(f"  {gtype:<12} {'0':>6}")
            continue
        mean_d = np.mean(deltas) * 100
        median_d = np.median(deltas) * 100
        n_pos = sum(1 for d in deltas if d > 0)
        print(f"  {gtype:<12} {len(deltas):>6} {mean_d:>+9.2f}% {median_d:>+9.2f}% {n_pos:>5}/{len(deltas)}")

    # Top gainers and losers
    print(f"\n{'='*70}")
    print(f"  TOP 10 GAINERS")
    print(f"{'='*70}")
    print(f"  {'Class':<30} {'Type':<12} {'Tmpl':>8} {'NETRA':>8} {'Δ':>8}")
    print(f"  {'-'*68}")
    for r in results[:10]:
        print(f"  {r['class']:<30} {r['type']:<12} {r['template_acc']*100:>7.1f}% {r['netra_acc']*100:>7.1f}% {r['delta']*100:>+7.1f}%")

    print(f"\n{'='*70}")
    print(f"  TOP 10 LOSERS")
    print(f"{'='*70}")
    print(f"  {'Class':<30} {'Type':<12} {'Tmpl':>8} {'NETRA':>8} {'Δ':>8}")
    print(f"  {'-'*68}")
    for r in results[-10:]:
        print(f"  {r['class']:<30} {r['type']:<12} {r['template_acc']*100:>7.1f}% {r['netra_acc']*100:>7.1f}% {r['delta']*100:>+7.1f}%")

    # Save
    output = {
        "dataset": args.dataset,
        "alpha": args.alpha,
        "beta": args.beta,
        "group_summary": {
            gtype: {
                "count": len(deltas),
                "mean_delta": float(np.mean(deltas)) if deltas else 0,
                "median_delta": float(np.median(deltas)) if deltas else 0,
                "std_delta": float(np.std(deltas)) if deltas else 0,
                "n_positive": sum(1 for d in deltas if d > 0),
            }
            for gtype, deltas in groups.items()
        },
        "per_class": results,
    }
    out_path = Path(args.output_dir) / f"action_perclass_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
