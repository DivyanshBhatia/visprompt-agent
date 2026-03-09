#!/usr/bin/env python3
"""Unsupervised weight selection for NETRA.

Tests whether the cosine divergence between template and description
embeddings predicts the optimal α/β weight ratio per dataset.

Hypothesis: When template and description centroids are similar (high cosine),
descriptions add little → use more templates (high α). When they diverge
(low cosine), descriptions add new information → use more descriptions (low α).

Usage:
    python scripts/run_unsupervised_weight.py \
        --datasets cifar100 flowers102 dtd oxford_pets food101 \
        --clip-model ViT-L/14 --llm gpt-4o
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_spec, build_task_runner
from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.utils.llm import LLMClient

logger = logging.getLogger(__name__)


def compute_group_divergence(task_runner, class_names, descriptions):
    """Compute mean cosine divergence between template and description centroids.
    
    For each class:
      1. Encode all templates → average → normalize = template centroid
      2. Encode all descriptions → average → normalize = description centroid
      3. Cosine similarity between the two centroids
    
    Returns:
      - per_class_cosine: dict of {class_name: cosine_sim}
      - mean_cosine: average across all classes
      - divergence: 1 - mean_cosine (higher = more different)
    """
    device = next(task_runner._clip_model.parameters()).device
    per_class_cosine = {}
    
    for cls in class_names:
        # Template centroid
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        tmpl_tokens = task_runner._tokenize(templates)
        with torch.no_grad():
            tmpl_emb = task_runner._clip_model.encode_text(tmpl_tokens.to(device))
            tmpl_emb = tmpl_emb / tmpl_emb.norm(dim=-1, keepdim=True)
            tmpl_centroid = tmpl_emb.mean(dim=0)
            tmpl_centroid = tmpl_centroid / tmpl_centroid.norm()
        
        # Description centroid
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        if len(descs) == 0:
            descs = [f"a photo of a {cls}"]
        desc_tokens = task_runner._tokenize(descs)
        with torch.no_grad():
            desc_emb = task_runner._clip_model.encode_text(desc_tokens.to(device))
            desc_emb = desc_emb / desc_emb.norm(dim=-1, keepdim=True)
            desc_centroid = desc_emb.mean(dim=0)
            desc_centroid = desc_centroid / desc_centroid.norm()
        
        # Cosine similarity
        cos_sim = (tmpl_centroid @ desc_centroid).item()
        per_class_cosine[cls] = cos_sim
    
    cosines = list(per_class_cosine.values())
    mean_cos = np.mean(cosines)
    divergence = 1.0 - mean_cos
    
    return per_class_cosine, mean_cos, divergence


def predict_alpha(divergence, min_alpha=0.0, max_alpha=1.0):
    """Simple linear heuristic: more divergence → lower alpha (more description weight).
    
    divergence ~ 0 → alpha ~ max_alpha (templates dominate)
    divergence ~ 0.3+ → alpha ~ min_alpha (descriptions dominate)
    """
    # Linear mapping: divergence 0.05 → alpha 0.85, divergence 0.25 → alpha 0.0
    alpha = max_alpha - (divergence - 0.05) * (max_alpha - min_alpha) / 0.20
    alpha = max(min_alpha, min(max_alpha, alpha))
    # Round to nearest 0.05
    alpha = round(alpha * 20) / 20
    return alpha


def main():
    parser = argparse.ArgumentParser(description="Unsupervised weight selection for NETRA")
    parser.add_argument("--datasets", nargs="+", 
                       default=["cifar100", "flowers102", "dtd", "oxford_pets", "food101",
                                "cifar10", "caltech101", "eurosat", "fgvc_aircraft", "country211"])
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n{'='*70}")
    print(f"  UNSUPERVISED WEIGHT SELECTION — NETRA")
    print(f"  Testing cosine divergence as predictor of optimal α")
    print(f"{'='*70}\n")
    
    results = []
    
    # Known optimal weights from our experiments
    known_optimal = {
        "cifar100": 0.55, "cifar10": 0.70, "flowers102": 0.0,
        "dtd": 0.40, "oxford_pets": 0.0, "food101": 0.55,
        "caltech101": 1.0, "eurosat": 0.0, "fgvc_aircraft": 0.55,
        "country211": 0.0,
    }
    
    for dataset_name in args.datasets:
        print(f"\n--- {dataset_name} ---")
        
        try:
            # Build task spec and runner
            args.dataset = dataset_name
            task_spec = build_task_spec(args)
            task_runner = build_task_runner(args, task_spec)
            class_names = task_spec.class_names
            
            # Generate or load descriptions
            from scripts.run import generate_descriptions
            descriptions = generate_descriptions(task_spec, args.llm, args.llm_provider)
            if isinstance(descriptions, tuple):
                descriptions = descriptions[0]
            
            # Compute divergence
            _, mean_cos, divergence = compute_group_divergence(
                task_runner, class_names, descriptions
            )
            
            # Predict alpha
            predicted_alpha = predict_alpha(divergence)
            actual_alpha = known_optimal.get(dataset_name, None)
            
            print(f"  Mean template-desc cosine: {mean_cos:.4f}")
            print(f"  Divergence (1-cos):        {divergence:.4f}")
            print(f"  Predicted α:               {predicted_alpha:.2f}")
            if actual_alpha is not None:
                print(f"  Actual optimal α:          {actual_alpha:.2f}")
                print(f"  Error:                     {abs(predicted_alpha - actual_alpha):.2f}")
            
            results.append({
                "dataset": dataset_name,
                "n_classes": len(class_names),
                "mean_cosine": mean_cos,
                "divergence": divergence,
                "predicted_alpha": predicted_alpha,
                "actual_optimal_alpha": actual_alpha,
                "error": abs(predicted_alpha - actual_alpha) if actual_alpha is not None else None,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY — Unsupervised Weight Prediction")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Divergence':>12} {'Predicted α':>13} {'Actual α':>10} {'Error':>8}")
    print(f"{'-'*60}")
    
    errors = []
    for r in results:
        actual = f"{r['actual_optimal_alpha']:.2f}" if r['actual_optimal_alpha'] is not None else "?"
        error = f"{r['error']:.2f}" if r['error'] is not None else "?"
        print(f"  {r['dataset']:<13} {r['divergence']:>12.4f} {r['predicted_alpha']:>13.2f} {actual:>10} {error:>8}")
        if r['error'] is not None:
            errors.append(r['error'])
    
    if errors:
        print(f"\n  Mean absolute error: {np.mean(errors):.3f}")
        print(f"  Median absolute error: {np.median(errors):.3f}")
        # Correlation
        divs = [r['divergence'] for r in results if r['actual_optimal_alpha'] is not None]
        actuals = [r['actual_optimal_alpha'] for r in results if r['actual_optimal_alpha'] is not None]
        if len(divs) >= 3:
            corr = np.corrcoef(divs, actuals)[0, 1]
            print(f"  Correlation (divergence vs actual α): {corr:.3f}")
    
    print(f"{'='*70}\n")
    
    # Save
    output_path = Path(args.output_dir) / "unsupervised_weight_selection.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
