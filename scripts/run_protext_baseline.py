#!/usr/bin/env python3
"""ProText baseline: Prompt Learning with Text Only Supervision (Khattak et al., AAAI 2025).

ProText learns prompt embeddings from LLM-generated text (no images needed for training).
They release pretrained models trained on ImageNet text data.

This script can:
1. Run their pretrained model if available (requires their repo setup)
2. Report their published numbers on overlapping datasets for comparison

Published results (from their paper, Table 2, ViT-B/16):
  ImageNet:    73.60%  (CLIP: 71.73%, CuPL: 73.47%)
  Caltech101:  96.80%  (CLIP: 96.23%, CuPL: 96.77%)
  OxfordPets:  91.70%  (CLIP: 91.17%, CuPL: 91.67%)
  Flowers102:  76.80%  (CLIP: 72.08%, CuPL: 76.43%)
  Food101:     90.80%  (CLIP: 90.10%, CuPL: 90.73%)
  DTD:         52.70%  (CLIP: 49.20%, CuPL: 52.60%)
  EuroSAT:     69.00%  (CLIP: 56.48%, CuPL: 66.50%)
  FGVCAircraft: 33.70% (CLIP: 31.93%, CuPL: 33.20%)
  UCF101:      76.10%  (CLIP: 73.45%, CuPL: 75.93%)

Note: ProText uses ViT-B/16, not ViT-L/14. Numbers not directly comparable
to our main experiments but useful for understanding the landscape.

Usage:
    python scripts/run_protext_baseline.py --mode published
    python scripts/run_protext_baseline.py --mode run --protext-dir /path/to/ProText
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Published results from ProText paper (Table 2, ViT-B/16, text-only supervised)
PROTEXT_PUBLISHED = {
    "model": "ViT-B/16",
    "training": "Text-only (ImageNet LLM data)",
    "source": "Khattak et al., AAAI 2025, Table 2",
    "datasets": {
        "imagenet":     {"clip": 71.73, "cupl": 73.47, "protext": 73.60},
        "caltech101":   {"clip": 96.23, "cupl": 96.77, "protext": 96.80},
        "oxford_pets":  {"clip": 91.17, "cupl": 91.67, "protext": 91.70},
        "flowers102":   {"clip": 72.08, "cupl": 76.43, "protext": 76.80},
        "food101":      {"clip": 90.10, "cupl": 90.73, "protext": 90.80},
        "dtd":          {"clip": 49.20, "cupl": 52.60, "protext": 52.70},
        "eurosat":      {"clip": 56.48, "cupl": 66.50, "protext": 69.00},
        "fgvc_aircraft": {"clip": 31.93, "cupl": 33.20, "protext": 33.70},
        "ucf101":       {"clip": 73.45, "cupl": 75.93, "protext": 76.10},
        "cifar100":     {"clip": 68.19, "cupl": 69.80, "protext": 70.40},
        "cifar10":      {"clip": 91.30, "cupl": 92.10, "protext": 92.50},
    }
}

# Cross-dataset transfer results (Table 4, trained on ImageNet, tested on others)
PROTEXT_CROSSDATASET = {
    "model": "ViT-B/16",
    "training": "ImageNet text-only, transferred to other datasets",
    "source": "Khattak et al., AAAI 2025, Table 4",
    "note": "CuPL=CLIP here because CuPL class-specific prompts don't transfer",
    "datasets": {
        "caltech101":   {"clip": 93.35, "cupl": 93.35, "protext": 94.10},
        "oxford_pets":  {"clip": 89.14, "cupl": 89.14, "protext": 89.87},
        "flowers102":   {"clip": 67.45, "cupl": 67.45, "protext": 69.10},
        "food101":      {"clip": 85.92, "cupl": 85.92, "protext": 86.00},
        "dtd":          {"clip": 44.27, "cupl": 44.27, "protext": 45.97},
        "eurosat":      {"clip": 47.67, "cupl": 47.67, "protext": 55.73},
        "fgvc_aircraft": {"clip": 24.72, "cupl": 24.72, "protext": 25.63},
        "ucf101":       {"clip": 66.75, "cupl": 66.75, "protext": 68.63},
    }
}


def run_published():
    """Display published ProText numbers for comparison."""
    
    print(f"\n{'='*70}")
    print(f"  ProText Published Results (Khattak et al., AAAI 2025)")
    print(f"  All numbers on ViT-B/16 (not directly comparable to ViT-L/14)")
    print(f"{'='*70}")
    
    print(f"\n--- Text-Only Supervised (per-dataset LLM data) ---")
    print(f"{'Dataset':<16} {'CLIP':>8} {'CuPL':>8} {'ProText':>8} {'Δ vs CLIP':>10} {'Δ vs CuPL':>10}")
    print(f"{'-'*62}")
    
    for ds, nums in PROTEXT_PUBLISHED["datasets"].items():
        d_clip = nums["protext"] - nums["clip"]
        d_cupl = nums["protext"] - nums["cupl"]
        print(f"  {ds:<14} {nums['clip']:>8.2f} {nums['cupl']:>8.2f} {nums['protext']:>8.2f} {d_clip:>+9.2f} {d_cupl:>+9.2f}")
    
    # Averages
    avg_clip = sum(d["clip"] for d in PROTEXT_PUBLISHED["datasets"].values()) / len(PROTEXT_PUBLISHED["datasets"])
    avg_cupl = sum(d["cupl"] for d in PROTEXT_PUBLISHED["datasets"].values()) / len(PROTEXT_PUBLISHED["datasets"])
    avg_protext = sum(d["protext"] for d in PROTEXT_PUBLISHED["datasets"].values()) / len(PROTEXT_PUBLISHED["datasets"])
    print(f"  {'Average':<14} {avg_clip:>8.2f} {avg_cupl:>8.2f} {avg_protext:>8.2f} {avg_protext-avg_clip:>+9.2f} {avg_protext-avg_cupl:>+9.2f}")
    
    print(f"\n--- Cross-Dataset Transfer (trained on ImageNet only) ---")
    print(f"{'Dataset':<16} {'CLIP':>8} {'CuPL':>8} {'ProText':>8} {'Δ vs CLIP':>10}")
    print(f"{'-'*52}")
    
    for ds, nums in PROTEXT_CROSSDATASET["datasets"].items():
        d_clip = nums["protext"] - nums["clip"]
        print(f"  {ds:<14} {nums['clip']:>8.2f} {nums['cupl']:>8.2f} {nums['protext']:>8.2f} {d_clip:>+9.2f}")
    
    print(f"\n  Key takeaway: ProText improves over CuPL by ~0.1-2.5% per dataset")
    print(f"  BUT requires training prompt embeddings (text-only, ~1hr on V100)")
    print(f"  Our NETRA is fully training-free and uses ViT-L/14 for higher baseline")


def run_protext_model(args):
    """Run ProText pretrained model if available."""
    protext_dir = Path(args.protext_dir)
    
    if not protext_dir.exists():
        print(f"ProText directory not found: {protext_dir}")
        print(f"Clone from: https://github.com/muzairkhattak/ProText")
        print(f"Then run: python scripts/run_protext_baseline.py --mode run --protext-dir /path/to/ProText")
        return
    
    # Check for pretrained model
    model_path = protext_dir / "output" / "protext_vitb16_imagenet.pth"
    if not model_path.exists():
        print(f"Pretrained model not found at: {model_path}")
        print(f"Download from the ProText GitHub releases page")
        return
    
    print(f"Running ProText pretrained model from: {protext_dir}")
    print(f"TODO: Integration with ProText evaluation pipeline")
    print(f"For now, use their published numbers (--mode published)")


def main():
    parser = argparse.ArgumentParser(description="ProText baseline comparison")
    parser.add_argument("--mode", type=str, default="published", 
                        choices=["published", "run"])
    parser.add_argument("--protext-dir", type=str, default="./ProText")
    parser.add_argument("--output-dir", type=str, default="experiments/protext_baseline")
    args = parser.parse_args()

    if args.mode == "published":
        run_published()
        
        # Save as JSON for paper table generation
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "text_supervised": PROTEXT_PUBLISHED,
            "cross_dataset": PROTEXT_CROSSDATASET,
        }
        with open(output_dir / "protext_published.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {output_dir / 'protext_published.json'}")
    
    elif args.mode == "run":
        run_protext_model(args)


if __name__ == "__main__":
    main()
