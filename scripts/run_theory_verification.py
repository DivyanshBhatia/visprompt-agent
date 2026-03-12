#!/usr/bin/env python3
"""Empirical verification of Appendix A.13 theoretical propositions.

Three experiments:
  1. Equal-norm assumption: CV of pre-normalization class embedding norms
  2. Margin decomposition: template vs description discriminativeness per dataset
  3. Suppression test: vary M (template count), show CuPL+e degrades but NETRA doesn't

Usage:
    python scripts/run_theory_verification.py \
        --datasets cifar100 flowers102 dtd oxford_pets eurosat food101 \
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

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(clip_model, device):
    """Load CLIP model once."""
    import open_clip
    model_name = clip_model.replace('/', '-')
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai', device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device):
    """Encode list of texts, return L2-normalized embeddings."""
    tokens = tokenizer(texts).to(device)
    emb = model.encode_text(tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


@torch.no_grad()
def encode_images_from_dataset(model, preprocess, images, device, batch_size=256):
    """Encode PIL images."""
    all_features = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(img) for img in images[start:end]]).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


# =====================================================================
# Experiment 1: Equal-Norm Assumption
# =====================================================================
@torch.no_grad()
def experiment1_equal_norms(model, tokenizer, classnames, descriptions, device):
    """Verify that pre-normalization class embedding norms are approximately equal."""
    
    configs = [
        (1.0, 0.0, "100/0"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100"),
    ]
    
    results = []
    
    for alpha, beta, label in configs:
        norms = []
        for cls in classnames:
            # Template embeddings
            if alpha > 0:
                templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
                tmpl_emb = encode_texts(model, tokenizer, templates, device)
                tmpl_centroid = tmpl_emb.mean(dim=0)  # NOT normalized yet
            
            # Description embeddings
            descs = descriptions.get(cls, [f"a photo of a {cls}"])
            if beta > 0:
                desc_emb = encode_texts(model, tokenizer, descs, device)
                desc_centroid = desc_emb.mean(dim=0)  # NOT normalized yet
            
            # Fused embedding (pre-normalization)
            if alpha > 0 and beta > 0:
                t_c = alpha * tmpl_centroid + beta * desc_centroid
            elif alpha > 0:
                t_c = tmpl_centroid
            else:
                t_c = desc_centroid
            
            norm = t_c.norm().item()
            norms.append(norm)
        
        norms = np.array(norms)
        mean_norm = norms.mean()
        std_norm = norms.std()
        cv = std_norm / mean_norm if mean_norm > 0 else 0
        
        results.append({
            "config": label,
            "alpha": alpha,
            "beta": beta,
            "mean_norm": float(mean_norm),
            "std_norm": float(std_norm),
            "cv": float(cv),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
        })
    
    return results


# =====================================================================
# Experiment 2: Margin Decomposition
# =====================================================================
@torch.no_grad()
def experiment2_margin_decomposition(model, tokenizer, preprocess, 
                                      classnames, descriptions, 
                                      images, labels, device):
    """Measure template-disc vs description-disc per correct classification."""
    
    # Compute centroids for all classes
    tmpl_centroids = {}
    desc_centroids = {}
    
    for cls in classnames:
        # Template centroid
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        tmpl_emb = encode_texts(model, tokenizer, templates, device)
        tmpl_centroids[cls] = tmpl_emb.mean(dim=0)
        
        # Description centroid
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        desc_emb = encode_texts(model, tokenizer, descs, device)
        desc_centroids[cls] = desc_emb.mean(dim=0)
    
    # Stack centroids
    mu_T = torch.stack([tmpl_centroids[c] for c in classnames]).to(device)  # [C, D]
    mu_D = torch.stack([desc_centroids[c] for c in classnames]).to(device)  # [C, D]
    
    # Encode images
    print("    Encoding images for margin decomposition...")
    image_features = encode_images_from_dataset(model, preprocess, images, device)
    image_features = image_features.to(device)
    labels_arr = np.array(labels)
    
    # For each correctly classified image (using NETRA 55/45), compute margins
    # First get NETRA predictions
    netra_emb = 0.55 * mu_T + 0.45 * mu_D
    netra_emb = netra_emb / netra_emb.norm(dim=-1, keepdim=True)
    sims = image_features @ netra_emb.T
    preds = sims.argmax(dim=1).cpu().numpy()
    
    correct_mask = preds == labels_arr
    
    template_discs = []
    desc_discs = []
    
    n_sample = min(correct_mask.sum(), 2000)  # Cap for speed
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > n_sample:
        correct_indices = np.random.RandomState(42).choice(
            correct_indices, n_sample, replace=False
        )
    
    for idx in correct_indices:
        x = image_features[idx]  # [D]
        c_star = labels_arr[idx]
        
        # Find top competitor (highest sim among wrong classes)
        all_sims = sims[idx].cpu().numpy()
        all_sims[c_star] = -np.inf
        c_prime = all_sims.argmax()
        
        # Template discrimination: x^T (mu_T^c* - mu_T^c')
        tmpl_diff = mu_T[c_star] - mu_T[c_prime]
        t_disc = (x @ tmpl_diff).item()
        
        # Description discrimination: x^T (mu_D^c* - mu_D^c')
        desc_diff = mu_D[c_star] - mu_D[c_prime]
        d_disc = (x @ desc_diff).item()
        
        template_discs.append(t_disc)
        desc_discs.append(d_disc)
    
    template_discs = np.array(template_discs)
    desc_discs = np.array(desc_discs)
    
    return {
        "n_samples": len(template_discs),
        "mean_template_disc": float(template_discs.mean()),
        "mean_desc_disc": float(desc_discs.mean()),
        "std_template_disc": float(template_discs.std()),
        "std_desc_disc": float(desc_discs.std()),
        "ratio": float(desc_discs.mean() / template_discs.mean()) if template_discs.mean() != 0 else float('inf'),
        "desc_positive_frac": float((desc_discs > 0).mean()),
        "template_positive_frac": float((template_discs > 0).mean()),
    }


# =====================================================================
# Experiment 3: Suppression Test (vary M)
# =====================================================================
@torch.no_grad()
def experiment3_suppression(model, tokenizer, preprocess,
                             classnames, descriptions,
                             images, labels, device):
    """Vary M (number of templates), show CuPL+e degrades but NETRA doesn't."""
    
    # Pre-encode all template and description embeddings
    all_tmpl_embs = {}  # cls -> [80, D] tensor
    all_desc_embs = {}  # cls -> [N, D] tensor
    
    for cls in classnames:
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        all_tmpl_embs[cls] = encode_texts(model, tokenizer, templates, device)
        
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        all_desc_embs[cls] = encode_texts(model, tokenizer, descs, device)
    
    # Encode images
    print("    Encoding images for suppression test...")
    image_features = encode_images_from_dataset(model, preprocess, images, device)
    image_features = image_features.to(device)
    labels_t = torch.tensor(labels, device=device)
    
    N = 10  # Fixed description count (use first 10 or all if < 10)
    M_values = [1, 5, 10, 20, 40, 80]
    beta_netra = 0.45
    
    results = []
    
    for M in M_values:
        # Select M templates (first M of the 80)
        
        # --- CuPL+e: uniform weighting ---
        cupl_class_embs = []
        for cls in classnames:
            tmpl = all_tmpl_embs[cls][:M]  # [M, D]
            desc = all_desc_embs[cls][:N]  # [N, D]
            all_emb = torch.cat([tmpl, desc], dim=0)  # [M+N, D]
            centroid = all_emb.mean(dim=0)
            centroid = centroid / centroid.norm()
            cupl_class_embs.append(centroid)
        cupl_class_embs = torch.stack(cupl_class_embs)
        
        cupl_sims = image_features @ cupl_class_embs.T
        cupl_preds = cupl_sims.argmax(dim=1)
        cupl_acc = (cupl_preds == labels_t).float().mean().item()
        
        # Effective description weight under uniform
        desc_weight_uniform = N / (M + N)
        
        # --- NETRA: group-normalized ---
        alpha_netra = 1 - beta_netra
        netra_class_embs = []
        for cls in classnames:
            tmpl = all_tmpl_embs[cls][:M]  # [M, D]
            desc = all_desc_embs[cls][:N]  # [N, D]
            tmpl_centroid = tmpl.mean(dim=0)
            desc_centroid = desc.mean(dim=0)
            fused = alpha_netra * tmpl_centroid + beta_netra * desc_centroid
            fused = fused / fused.norm()
            netra_class_embs.append(fused)
        netra_class_embs = torch.stack(netra_class_embs)
        
        netra_sims = image_features @ netra_class_embs.T
        netra_preds = netra_sims.argmax(dim=1)
        netra_acc = (netra_preds == labels_t).float().mean().item()
        
        results.append({
            "M": M,
            "N": N,
            "desc_weight_uniform": desc_weight_uniform,
            "desc_weight_netra": beta_netra,
            "cupl_acc": float(cupl_acc),
            "netra_acc": float(netra_acc),
        })
    
    return results


def load_dataset_images(dataset_name, args):
    """Load images and labels for a dataset."""
    from PIL import Image
    
    if dataset_name in ["cifar100", "cifar10"]:
        import torchvision
        if dataset_name == "cifar100":
            ds = torchvision.datasets.CIFAR100(root=args.data_dir or "./data", train=False, download=True)
        else:
            ds = torchvision.datasets.CIFAR10(root=args.data_dir or "./data", train=False, download=True)
        
        n = min(args.val_size, len(ds))
        indices = np.random.RandomState(42).permutation(len(ds))[:n]
        images = [Image.fromarray(ds.data[i]) for i in indices]
        labels = [ds.targets[i] for i in indices]
        return images, labels
    
    elif dataset_name == "flowers102":
        import torchvision
        ds = torchvision.datasets.Flowers102(root=args.data_dir or "./data", split="test", download=True)
        n = min(args.val_size, len(ds))
        indices = np.random.RandomState(42).permutation(len(ds))[:n]
        images = [ds[i][0] for i in indices]
        labels = [ds[i][1] for i in indices]
        return images, labels
    
    elif dataset_name == "dtd":
        import torchvision
        ds = torchvision.datasets.DTD(root=args.data_dir or "./data", split="test", download=True)
        n = min(args.val_size, len(ds))
        indices = np.random.RandomState(42).permutation(len(ds))[:n]
        images = [ds[i][0] for i in indices]
        labels = [ds[i][1] for i in indices]
        return images, labels
    
    elif dataset_name == "oxford_pets":
        import torchvision
        ds = torchvision.datasets.OxfordIIITPet(root=args.data_dir or "./data", split="test", download=True)
        n = min(args.val_size, len(ds))
        indices = np.random.RandomState(42).permutation(len(ds))[:n]
        images = [ds[i][0] for i in indices]
        labels = [ds[i][1] for i in indices]
        return images, labels
    
    elif dataset_name == "food101":
        import torchvision
        ds = torchvision.datasets.Food101(root=args.data_dir or "./data", split="test", download=True)
        n = min(args.val_size, len(ds))
        indices = np.random.RandomState(42).permutation(len(ds))[:n]
        images = [ds[i][0] for i in indices]
        labels = [ds[i][1] for i in indices]
        return images, labels
    
    elif dataset_name == "eurosat":
        import torchvision
        ds = torchvision.datasets.EuroSAT(root=args.data_dir or "./data", download=True)
        # EuroSAT has no split, use last 5000 as test
        n_total = len(ds)
        test_indices = list(range(n_total - 5000, n_total))
        n = min(args.val_size, len(test_indices))
        indices = np.random.RandomState(42).permutation(len(test_indices))[:n]
        indices = [test_indices[i] for i in indices]
        images = [ds[i][0] for i in indices]
        labels = [ds[i][1] for i in indices]
        return images, labels
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for image loading")


def main():
    parser = argparse.ArgumentParser(description="Verify A.13 theoretical propositions")
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar100", "flowers102", "dtd", "oxford_pets", "eurosat"])
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--experiments", nargs="+", default=["1", "2", "3"],
                        help="Which experiments to run: 1=norms, 2=margins, 3=suppression")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    print(f"\n{'='*70}")
    print(f"  THEORY VERIFICATION — Appendix A.13")
    print(f"  CLIP: {args.clip_model}")
    print(f"{'='*70}\n")
    
    # Load model once
    model, preprocess, tokenizer = load_model_and_tokenizer(args.clip_model, args.device)
    
    all_results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"  {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Get class names
        args.dataset = dataset_name
        task_spec = build_task_spec(args)
        classnames = task_spec.class_names
        
        # Load descriptions
        # Load or generate descriptions
        desc_cache = Path(args.output_dir) / f"descriptions_{dataset_name}_{args.llm}.json"
        descriptions = None
        # Try multiple patterns
        for pat in [f"descriptions_{dataset_name}_{args.llm}.json",
                    f"desc_{dataset_name}_{args.llm}.json"]:
            p = Path(args.output_dir) / pat
            if p.exists():
                with open(p) as f:
                    descriptions = json.load(f)
                print(f"  Loaded cached descriptions from {p.name}")
                break
        # Search recursively
        if descriptions is None:
            for p in Path(args.output_dir).rglob(f"*{dataset_name}*desc*.json"):
                with open(p) as f:
                    descriptions = json.load(f)
                print(f"  Found descriptions at {p}")
                break
        # Generate if still not found
        if descriptions is None:
            print(f"  Generating descriptions for {dataset_name}...")
            from scripts.run_weight_ablation import generate_descriptions
            descriptions, cost = generate_descriptions(task_spec, args.llm, "openai")
            with open(desc_cache, 'w') as f:
                json.dump(descriptions, f, indent=1)
            print(f"  Generated and cached ({cost})")
        
        dataset_results = {"dataset": dataset_name, "n_classes": len(classnames)}
        
        # ── Experiment 1: Equal-Norm Assumption ──
        if "1" in args.experiments:
            print(f"\n  --- Experiment 1: Equal-Norm Assumption ---")
            exp1 = experiment1_equal_norms(model, tokenizer, classnames, descriptions, args.device)
            dataset_results["exp1_norms"] = exp1
            
            print(f"  {'Config':<10} {'Mean ∥t∥':>10} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
            print(f"  {'-'*54}")
            for r in exp1:
                print(f"  {r['config']:<10} {r['mean_norm']:>10.4f} {r['std_norm']:>8.4f} {r['cv']:>8.4f} {r['min_norm']:>8.4f} {r['max_norm']:>8.4f}")
        
        # ── Experiment 2: Margin Decomposition ──
        if "2" in args.experiments:
            print(f"\n  --- Experiment 2: Margin Decomposition ---")
            try:
                images, labels = load_dataset_images(dataset_name, args)
                exp2 = experiment2_margin_decomposition(
                    model, tokenizer, preprocess, classnames, descriptions,
                    images, labels, args.device
                )
                dataset_results["exp2_margins"] = exp2
                
                print(f"  Samples: {exp2['n_samples']}")
                print(f"  Mean template disc: {exp2['mean_template_disc']:.4f} ± {exp2['std_template_disc']:.4f}")
                print(f"  Mean desc disc:     {exp2['mean_desc_disc']:.4f} ± {exp2['std_desc_disc']:.4f}")
                print(f"  Ratio (desc/tmpl):  {exp2['ratio']:.2f}×")
                print(f"  Desc positive:      {exp2['desc_positive_frac']*100:.1f}%")
            except Exception as e:
                print(f"  Skipped: {e}")
        
        # ── Experiment 3: Suppression Test ──
        if "3" in args.experiments:
            print(f"\n  --- Experiment 3: Suppression (vary M) ---")
            try:
                if "2" not in args.experiments:
                    images, labels = load_dataset_images(dataset_name, args)
                exp3 = experiment3_suppression(
                    model, tokenizer, preprocess, classnames, descriptions,
                    images, labels, args.device
                )
                dataset_results["exp3_suppression"] = exp3
                
                print(f"  {'M':>4} {'Desc wt (unif)':>15} {'CuPL+e':>10} {'NETRA':>10} {'Δ CuPL+e':>10} {'Δ NETRA':>10}")
                print(f"  {'-'*61}")
                base_cupl = exp3[-1]['cupl_acc']  # M=80 as baseline
                base_netra = exp3[-1]['netra_acc']
                for r in exp3:
                    d_cupl = (r['cupl_acc'] - base_cupl) * 100
                    d_netra = (r['netra_acc'] - base_netra) * 100
                    print(f"  {r['M']:>4} {r['desc_weight_uniform']:>15.2%} {r['cupl_acc']*100:>9.2f}% {r['netra_acc']*100:>9.2f}% {d_cupl:>+9.2f}% {d_netra:>+9.2f}%")
            except Exception as e:
                print(f"  Skipped: {e}")
        
        all_results[dataset_name] = dataset_results
    
    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    
    if "1" in args.experiments:
        print(f"\n  Exp 1 — Equal-Norm CV at 55/45:")
        for ds in args.datasets:
            if ds in all_results and "exp1_norms" in all_results[ds]:
                for r in all_results[ds]["exp1_norms"]:
                    if r["config"] == "55/45":
                        status = "✓" if r["cv"] < 0.1 else "⚠"
                        print(f"  {status} {ds:<15} CV = {r['cv']:.4f}")
    
    if "2" in args.experiments:
        print(f"\n  Exp 2 — Desc/Template Discrimination Ratio:")
        for ds in args.datasets:
            if ds in all_results and "exp2_margins" in all_results[ds]:
                r = all_results[ds]["exp2_margins"]
                print(f"  {ds:<15} ratio = {r['ratio']:.2f}× (desc positive: {r['desc_positive_frac']*100:.0f}%)")
    
    if "3" in args.experiments:
        print(f"\n  Exp 3 — Suppression (CuPL+e acc change M=1 vs M=80):")
        for ds in args.datasets:
            if ds in all_results and "exp3_suppression" in all_results[ds]:
                exp3 = all_results[ds]["exp3_suppression"]
                cupl_change = (exp3[0]['cupl_acc'] - exp3[-1]['cupl_acc']) * 100
                netra_change = (exp3[0]['netra_acc'] - exp3[-1]['netra_acc']) * 100
                print(f"  {ds:<15} CuPL+e: {cupl_change:+.2f}%  NETRA: {netra_change:+.2f}%")
    
    # Save
    out_path = Path(args.output_dir) / "theory_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
