#!/usr/bin/env python3
"""NETRA on applied domains: product categorization, medical, grocery.

Tests whether NETRA's gains on fine-grained benchmarks transfer to
practical applied domains where LLM descriptions are naturally rich.

Supported datasets:
  - fgvc_food (Food-101 subset with fine-grained cuisine types)
  - stanford_cars (196 car models)
  - freiburg_groceries (25 grocery categories)
  - fashion_mnist (10 clothing types — simple baseline)
  - custom (any ImageFolder dataset)

Usage:
    # Stanford Cars (196 fine-grained car models)
    python scripts/run_applied_domains.py \
        --dataset stanford_cars --data-dir /path/to/cars \
        --clip-model ViT-L/14 --llm gpt-4o

    # Any ImageFolder dataset
    python scripts/run_applied_domains.py \
        --dataset custom --data-dir /path/to/dataset \
        --clip-model ViT-L/14 --llm gpt-4o --domain-type "product"
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.utils.llm import LLMClient

logger = logging.getLogger(__name__)

# Domain-specific prompt templates
PRODUCT_TEMPLATES = [
    "a photo of a {}",
    "a product photo of a {}",
    "an image of a {} for sale",
    "a {} on a white background",
    "a close-up photo of a {}",
    "an online listing photo of a {}",
    "a {} product image",
    "a catalog photo of a {}",
    "a studio photo of a {}",
    "a retail display of a {}",
]

MEDICAL_TEMPLATES = [
    "a medical image of {}",
    "a clinical photo showing {}",
    "a pathology image of {}",
    "a diagnostic image of {}",
    "a radiograph showing {}",
    "a microscopy image of {}",
    "a histological image of {}",
    "a medical scan showing {}",
    "a clinical presentation of {}",
    "an example of {} in medical imaging",
]

DOMAIN_TEMPLATES = {
    "product": PRODUCT_TEMPLATES,
    "medical": MEDICAL_TEMPLATES,
    "general": [],  # Use only ImageNet templates
}

# Description prompts per domain
DOMAIN_DESC_PROMPTS = {
    "product": (
        "For each product category below, provide 10-15 short visual descriptions "
        "suitable for CLIP-based product image classification. Focus on: shape, color, "
        "material, texture, typical packaging, distinctive design elements, size relative "
        "to common objects, and typical setting where the product appears.\n\n"
        "Categories: {classes}\n\n"
        'Return ONLY valid JSON: {{"category": ["description1", ...]}}\n'
    ),
    "medical": (
        "For each medical condition/finding below, provide 10-15 short visual descriptions "
        "suitable for CLIP-based medical image classification. Focus on: shape, color, "
        "texture, borders, location, size, contrast with surrounding tissue, and "
        "distinctive morphological features.\n\n"
        "Categories: {classes}\n\n"
        'Return ONLY valid JSON: {{"category": ["description1", ...]}}\n'
    ),
    "general": (
        "For each category below, provide 10-15 short visual descriptions "
        "suitable for CLIP-based image classification. Focus on distinctive "
        "visual attributes: color, shape, texture, size, parts, habitat/setting.\n\n"
        "Categories: {classes}\n\n"
        'Return ONLY valid JSON: {{"category": ["description1", ...]}}\n'
    ),
}


def load_imagefolder_dataset(data_dir, val_size=None):
    """Load any ImageFolder-style dataset."""
    from PIL import Image
    data_path = Path(data_dir)
    
    # Find class folders
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    # Try common subfolder names
    if len(class_dirs) == 0 or all(d.name in ['train', 'val', 'test', 'images'] for d in class_dirs):
        for split in ['test', 'val', 'validation', 'images']:
            sp = data_path / split
            if sp.exists():
                class_dirs = sorted([d for d in sp.iterdir() if d.is_dir()])
                if len(class_dirs) > 2:
                    data_path = sp
                    break
    
    if len(class_dirs) == 0:
        raise RuntimeError(f"No class folders found in {data_dir}")
    
    classnames = [d.name.replace('_', ' ') for d in class_dirs]
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    all_images = []
    for cls_dir in class_dirs:
        for f in cls_dir.rglob('*'):
            if f.suffix.lower() in image_extensions:
                all_images.append((f, class_to_idx[cls_dir.name]))
    
    print(f"  Found {len(all_images)} images across {len(classnames)} classes")
    
    if val_size and val_size < len(all_images):
        random.seed(42)
        all_images = random.sample(all_images, val_size)
        print(f"  Subsampled to {val_size}")
    
    images = []
    labels = []
    skipped = 0
    for fpath, label in all_images:
        try:
            img = Image.open(fpath).convert('RGB')
            images.append(img)
            labels.append(label)
        except Exception:
            skipped += 1
    
    print(f"  Loaded {len(images)} images ({skipped} skipped)")
    return images, labels, classnames


def load_stanford_cars(data_dir, val_size=None):
    """Load Stanford Cars dataset."""
    try:
        import torchvision
        dataset = torchvision.datasets.StanfordCars(
            root=data_dir or "./data", split="test", download=True
        )
        classnames = [c.replace('_', ' ') for c in dataset.classes]
        
        n = min(val_size or len(dataset), len(dataset))
        indices = np.random.RandomState(42).permutation(len(dataset))[:n]
        
        images = []
        labels = []
        for idx in indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        print(f"  Loaded {len(images)} images, {len(classnames)} classes")
        return images, labels, classnames
    except Exception as e:
        print(f"  torchvision StanfordCars failed: {e}")
        print(f"  Trying ImageFolder from {data_dir}...")
        return load_imagefolder_dataset(data_dir, val_size)


def load_fashion_mnist(data_dir, val_size=None):
    """Load Fashion-MNIST as a simple product baseline."""
    import torchvision
    from PIL import Image
    
    dataset = torchvision.datasets.FashionMNIST(
        root=data_dir or "./data", train=False, download=True
    )
    classnames = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot"
    ]
    
    n = min(val_size or len(dataset), len(dataset))
    indices = np.random.RandomState(42).permutation(len(dataset))[:n]
    
    images = []
    labels = []
    for idx in indices:
        img_tensor, label = dataset[idx]
        # Convert grayscale to RGB PIL
        img = img_tensor.convert('RGB')
        images.append(img)
        labels.append(label)
    
    print(f"  Loaded {len(images)} images, {len(classnames)} classes")
    return images, labels, classnames


def load_dataset(dataset_name, data_dir, val_size):
    """Load dataset by name."""
    if dataset_name == "stanford_cars":
        return load_stanford_cars(data_dir, val_size)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(data_dir, val_size)
    elif dataset_name == "custom":
        return load_imagefolder_dataset(data_dir, val_size)
    else:
        # Try ImageFolder
        return load_imagefolder_dataset(data_dir, val_size)


def generate_descriptions(classnames, llm_model, llm_provider, domain_type, cache_path):
    """Generate domain-specific descriptions."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    
    client = LLMClient(model=llm_model, provider=llm_provider, temperature=0.7)
    prompt_template = DOMAIN_DESC_PROMPTS.get(domain_type, DOMAIN_DESC_PROMPTS["general"])
    descriptions = {}
    
    for i in range(0, len(classnames), 10):
        batch = classnames[i:i+10]
        prompt = prompt_template.format(classes=", ".join(batch))
        try:
            resp = client.call(prompt, json_mode=True)
            descs = json.loads(resp)
            for cn in batch:
                found = None
                for key in descs:
                    if cn.lower().strip() in key.lower() or key.lower() in cn.lower():
                        found = descs[key]
                        break
                descriptions[cn] = found if found and isinstance(found, list) else [f"a photo of a {cn}"]
        except Exception as e:
            logger.warning(f"  LLM batch failed: {e}")
            for cn in batch:
                descriptions[cn] = [f"a photo of a {cn}"]
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(descriptions, f, indent=1)
    return descriptions


@torch.no_grad()
def run_evaluation(model, preprocess, tokenizer, images, labels, classnames,
                   descriptions, domain_type, device, batch_size=256):
    """Run full evaluation: baselines + NETRA sweep."""
    import open_clip
    
    # Encode images
    print("  Encoding images...")
    all_features = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(img) for img in images[start:end]]).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
    image_features = torch.cat(all_features, dim=0)
    labels_t = torch.tensor(labels)
    
    def encode_and_eval(prompts_per_class, weights_per_class=None):
        """Encode text prompts and evaluate."""
        class_embs = []
        for cls in classnames:
            prompts = prompts_per_class[cls]
            tokens = tokenizer(prompts).to(device)
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            if weights_per_class and cls in weights_per_class:
                w = torch.tensor(weights_per_class[cls], dtype=torch.float32, device=device)
                w = w / w.sum()
                cls_emb = (emb * w.unsqueeze(1)).sum(dim=0)
            else:
                cls_emb = emb.mean(dim=0)
            cls_emb = cls_emb / cls_emb.norm()
            class_embs.append(cls_emb)
        
        class_embs = torch.stack(class_embs).to(device)
        sims = image_features.to(device) @ class_embs.T
        preds = sims.argmax(dim=1).cpu()
        top1 = (preds == labels_t).float().mean().item()
        top5_preds = sims.topk(min(5, sims.shape[1]), dim=1).indices.cpu()
        top5 = (top5_preds == labels_t.unsqueeze(1)).any(dim=1).float().mean().item()
        return top1, top5
    
    # Get domain-specific templates
    domain_tmpls = DOMAIN_TEMPLATES.get(domain_type, [])
    all_templates = domain_tmpls + IMAGENET_TEMPLATES
    M = len(all_templates)
    
    results = {}
    
    # Class name only
    ppc = {c: [c] for c in classnames}
    t1, t5 = encode_and_eval(ppc)
    results["class_name"] = {"top1": t1, "top5": t5}
    print(f"  Class name only:    {t1*100:.2f}%")
    
    # Single template
    ppc = {c: [f"a photo of a {c}"] for c in classnames}
    t1, t5 = encode_and_eval(ppc)
    results["single_template"] = {"top1": t1, "top5": t5}
    print(f"  Single template:    {t1*100:.2f}%")
    
    # All templates (domain + ImageNet)
    ppc = {c: [t.format(c) for t in all_templates] for c in classnames}
    t1, t5 = encode_and_eval(ppc)
    baseline = t1
    results["templates"] = {"top1": t1, "top5": t5}
    print(f"  {M} templates:      {t1*100:.2f}%")
    
    # CuPL (desc only)
    ppc = {c: descriptions.get(c, [f"a photo of a {c}"]) for c in classnames}
    t1, t5 = encode_and_eval(ppc)
    results["cupl_desc"] = {"top1": t1, "top5": t5}
    print(f"  CuPL (desc only):   {t1*100:.2f}%  Δ: {(t1-baseline)*100:+.2f}%")
    
    # CuPL+e (uniform)
    ppc = {}
    for c in classnames:
        tmpl = [t.format(c) for t in all_templates]
        descs = descriptions.get(c, [])
        ppc[c] = tmpl + descs
    t1, t5 = encode_and_eval(ppc)
    results["cupl_e"] = {"top1": t1, "top5": t5}
    print(f"  CuPL+e (uniform):   {t1*100:.2f}%  Δ: {(t1-baseline)*100:+.2f}%")
    
    # NETRA sweep
    print(f"\n  NETRA Weight Sweep:")
    print(f"  {'Config':<20} {'Top-1':>8} {'Top-5':>8} {'Δ':>8}")
    print(f"  {'-'*46}")
    
    sweep = []
    for alpha, beta, label in [
        (1.0, 0.0, "100/0 (templates)"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45 (default)"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100 (desc only)"),
    ]:
        ppc = {}
        wpc = {}
        for c in classnames:
            tmpl = [t.format(c) for t in all_templates] if alpha > 0 else []
            descs = descriptions.get(c, [f"a photo of a {c}"]) if beta > 0 else []
            N = len(descs)
            t_w = [alpha / M] * M if alpha > 0 else []
            d_w = [beta / N] * N if beta > 0 else []
            prompts = (tmpl if alpha > 0 else []) + (descs if beta > 0 else [])
            weights = t_w + d_w
            if not prompts:
                prompts, weights = [f"a photo of a {c}"], [1.0]
            ppc[c] = prompts
            wpc[c] = weights
        
        t1, t5 = encode_and_eval(ppc, wpc)
        delta = t1 - baseline
        print(f"  {label:<20} {t1*100:>7.2f}% {t5*100:>7.2f}% {delta*100:>+7.2f}%")
        sweep.append({"alpha": alpha, "beta": beta, "label": label,
                       "top1": t1, "top5": t5, "delta": delta})
    
    best = max(sweep, key=lambda x: x["top1"])
    default = [s for s in sweep if s["alpha"] == 0.55][0]
    
    results["netra_sweep"] = sweep
    results["netra_best"] = best
    results["netra_default"] = default
    results["baseline_top1"] = baseline
    
    return results


def main():
    parser = argparse.ArgumentParser(description="NETRA on applied domains")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name or 'custom' for ImageFolder")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--domain-type", default="general",
                        choices=["product", "medical", "general"])
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Auto-detect domain type
    if args.domain_type == "general":
        if "car" in args.dataset.lower() or "product" in args.dataset.lower():
            args.domain_type = "product"
        elif "medical" in args.dataset.lower() or "path" in args.dataset.lower():
            args.domain_type = "medical"
    
    print(f"\n{'='*65}")
    print(f"  NETRA — APPLIED DOMAIN: {args.dataset.upper()}")
    print(f"  Domain: {args.domain_type} | CLIP: {args.clip_model} | LLM: {args.llm}")
    print(f"{'='*65}\n")
    
    # Load dataset
    print("Loading dataset...")
    images, labels, classnames = load_dataset(args.dataset, args.data_dir, args.val_size)
    print(f"  {len(classnames)} classes, {len(images)} images\n")
    
    # Load CLIP
    import open_clip
    model_name = args.clip_model.replace('/', '-')
    print(f"Loading CLIP {args.clip_model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai', device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    
    # Generate descriptions
    print("\nGenerating descriptions...")
    cache = Path(args.output_dir) / f"desc_{args.dataset}_{args.domain_type}_{args.llm}.json"
    descriptions = generate_descriptions(
        classnames, args.llm, args.llm_provider, args.domain_type, cache
    )
    print(f"  {sum(1 for c in classnames if c in descriptions)}/{len(classnames)} classes\n")
    
    # Run evaluation
    results = run_evaluation(
        model, preprocess, tokenizer, images, labels, classnames,
        descriptions, args.domain_type, args.device, args.batch_size
    )
    
    # Summary
    best = results["netra_best"]
    default = results["netra_default"]
    baseline = results["baseline_top1"]
    
    print(f"\n{'='*65}")
    print(f"  SUMMARY — {args.dataset} ({args.domain_type})")
    print(f"{'='*65}")
    print(f"  Templates baseline: {baseline*100:.2f}%")
    print(f"  CuPL+e (uniform):   {results['cupl_e']['top1']*100:.2f}%  ({(results['cupl_e']['top1']-baseline)*100:+.2f}%)")
    print(f"  NETRA (55/45):       {default['top1']*100:.2f}%  ({default['delta']*100:+.2f}%)")
    print(f"  NETRA (best):        {best['top1']*100:.2f}%  ({best['delta']*100:+.2f}%)  @ {best['label']}")
    print(f"{'='*65}\n")
    
    # Save
    output = {
        "dataset": args.dataset,
        "domain_type": args.domain_type,
        "clip_model": args.clip_model,
        "llm": args.llm,
        "n_images": len(images),
        "n_classes": len(classnames),
        **results,
    }
    out_path = Path(args.output_dir) / f"applied_{args.dataset}_{args.domain_type}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
