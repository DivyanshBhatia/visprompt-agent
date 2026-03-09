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

from scripts.run import build_task_spec
from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.utils.llm import LLMClient

logger = logging.getLogger(__name__)


def compute_group_divergence(model, tokenizer, class_names, descriptions, device):
    """Compute mean cosine divergence between template and description centroids."""
    per_class_cosine = {}
    
    for cls in class_names:
        # Template centroid
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        tmpl_tokens = tokenizer(templates).to(device)
        with torch.no_grad():
            tmpl_emb = model.encode_text(tmpl_tokens)
            tmpl_emb = tmpl_emb / tmpl_emb.norm(dim=-1, keepdim=True)
            tmpl_centroid = tmpl_emb.mean(dim=0)
            tmpl_centroid = tmpl_centroid / tmpl_centroid.norm()
        
        # Description centroid
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        if len(descs) == 0:
            descs = [f"a photo of a {cls}"]
        desc_tokens = tokenizer(descs).to(device)
        with torch.no_grad():
            desc_emb = model.encode_text(desc_tokens)
            desc_emb = desc_emb / desc_emb.norm(dim=-1, keepdim=True)
            desc_centroid = desc_emb.mean(dim=0)
            desc_centroid = desc_centroid / desc_centroid.norm()
        
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
    
    # Load CLIP model once (shared across datasets)
    import open_clip
    model_name = args.clip_model or "ViT-L/14"
    model_name_clean = model_name.replace('/', '-')
    print(f"Loading CLIP model {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name_clean, pretrained='openai', device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name_clean)
    print(f"  Done.\n")
    
    for dataset_name in args.datasets:
        print(f"\n--- {dataset_name} ---")
        
        try:
            # Get class names
            args.dataset = dataset_name
            task_spec = build_task_spec(args)
            class_names = task_spec.class_names
            
            # Generate or load descriptions
            desc_cache = Path(args.output_dir) / f"descriptions_{dataset_name}_{args.llm}.json"
            if desc_cache.exists():
                with open(desc_cache) as f:
                    descriptions = json.load(f)
                print(f"  Loaded cached descriptions from {desc_cache}")
            else:
                client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
                descriptions = {}
                for i in range(0, len(class_names), 10):
                    batch = class_names[i:i+10]
                    prompt = (
                        f"For each category below, provide 10-15 short visual descriptions "
                        f"suitable for CLIP-based image classification. Focus on distinctive "
                        f"visual attributes: color, shape, texture, size, parts, habitat/setting.\n\n"
                        f"Categories: {', '.join(batch)}\n\n"
                        f'Return ONLY valid JSON: {{"category": ["description1", ...]}}\n'
                    )
                    try:
                        resp = client.call(prompt, json_mode=True)
                        descs = json.loads(resp)
                        for cn in batch:
                            found = None
                            for key in descs:
                                if cn.lower().strip() in key.lower():
                                    found = descs[key]
                                    break
                            descriptions[cn] = found if found and isinstance(found, list) else [f"a photo of a {cn}"]
                    except Exception as e:
                        for cn in batch:
                            descriptions[cn] = [f"a photo of a {cn}"]
                desc_cache.parent.mkdir(parents=True, exist_ok=True)
                with open(desc_cache, 'w') as f:
                    json.dump(descriptions, f, indent=1)
                print(f"  Generated and cached descriptions")
            
            # Compute divergence
            _, mean_cos, divergence = compute_group_divergence(
                model, tokenizer, class_names, descriptions, args.device
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
