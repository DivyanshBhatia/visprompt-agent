#!/usr/bin/env python3
"""NETRA evaluation on ImageNet, ImageNet-V2, and DomainNet.

These are the standard large-scale zero-shot benchmarks.
ImageNet: 1000 classes, 50K val images
ImageNet-V2: 1000 classes, 10K images (distribution shift)
DomainNet: 345 classes, 6 domains (clipart, infograph, painting, quickdraw, real, sketch)

Usage:
    # ImageNet (requires val set at --data-dir)
    python scripts/run_imagenet.py --dataset imagenet --data-dir /path/to/imagenet/val \
        --clip-model ViT-L/14 --llm gpt-4o

    # ImageNet-V2 (auto-downloads)
    python scripts/run_imagenet.py --dataset imagenet-v2 --clip-model ViT-L/14

    # DomainNet (requires download)
    python scripts/run_imagenet.py --dataset domainnet --data-dir /path/to/domainnet \
        --domain real --clip-model ViT-L/14

    # Quick test (subset of 5000 images)
    python scripts/run_imagenet.py --dataset imagenet --data-dir /path/to/imagenet/val \
        --clip-model ViT-L/14 --val-size 5000
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from visprompt.baselines import IMAGENET_TEMPLATES
from visprompt.task_spec import TaskSpec
from visprompt.utils.llm import LLMClient

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# ImageNet class names (1000 classes, standard ordering)
# ══════════════════════════════════════════════════════════════════════

def get_imagenet_classnames():
    """Get ImageNet-1K class names in standard CLIP ordering.
    
    These are the human-readable class names used by CLIP and OpenAI.
    We use the standard openai/CLIP class names from their repo.
    """
    try:
        # Try to load from open_clip or clip
        import open_clip
        # open_clip ships with imagenet classnames
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        # Fall through to manual list if not available
    except Exception:
        pass
    
    # Standard ImageNet class names (will be loaded from file or generated)
    # For now, use torchvision's built-in
    try:
        import torchvision
        # ImageNet class names from torchvision
        # These match the folder names (wnids), we need human-readable names
        pass
    except Exception:
        pass
    
    # Use the canonical list from CLIP paper / openai github
    # This file should be at data/imagenet_classnames.json
    classname_file = Path(__file__).parent.parent / "data" / "imagenet_classnames.json"
    if classname_file.exists():
        with open(classname_file) as f:
            return json.load(f)
    
    # Generate from folder names if ImageNet val is available
    return None


def download_imagenet_classnames():
    """Download ImageNet class names from reliable sources."""
    import urllib.request
    classnames_dir = Path(__file__).parent.parent / "data"
    classnames_dir.mkdir(exist_ok=True)
    classnames_file = classnames_dir / "imagenet_classnames.json"
    
    if classnames_file.exists():
        with open(classnames_file) as f:
            return json.load(f)
    
    # Try multiple sources
    urls = [
        "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
        "https://raw.githubusercontent.com/openai/CLIP/main/notebooks/imagenet_classes.txt",
        "https://raw.githubusercontent.com/xmartlabs/caffeflow/master/examples/imagenet/imagenet-classes.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt",
    ]
    
    for url in urls:
        try:
            print(f"  Downloading ImageNet class names from {url.split('/')[2]}...")
            response = urllib.request.urlopen(url, timeout=10)
            text = response.read().decode('utf-8')
            
            if url.endswith('.json'):
                classnames = json.loads(text)
            elif url.endswith('.txt'):
                lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
                # TF labels file has "background" as first entry
                if lines[0].lower() == 'background':
                    lines = lines[1:]
                classnames = lines[:1000]
            
            if len(classnames) == 1000:
                with open(classnames_file, 'w') as f:
                    json.dump(classnames, f, indent=2)
                print(f"  Saved {len(classnames)} class names")
                return classnames
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("  WARNING: Could not download ImageNet class names from any source.")
    print("  Using numeric labels (0-999). Results will be invalid.")
    return None


# ══════════════════════════════════════════════════════════════════════
# Dataset loading
# ══════════════════════════════════════════════════════════════════════

def load_imagenet(data_dir, val_size=None):
    """Load ImageNet validation set using ImageFolder."""
    from torchvision.datasets import ImageFolder
    
    dataset = ImageFolder(data_dir)
    
    # Get class names from folder structure
    # ImageNet folders are wnids (n01440764), we need human-readable names
    classnames = download_imagenet_classnames()
    if classnames is None:
        # Fallback: use folder names cleaned up
        classnames = [name.replace('_', ' ') for name in dataset.classes]
    
    print(f"  Loaded ImageNet val: {len(dataset)} images, {len(classnames)} classes")
    
    # Subsample if requested
    if val_size and val_size < len(dataset):
        indices = np.random.RandomState(42).permutation(len(dataset))[:val_size]
    else:
        indices = np.arange(len(dataset))
    
    return dataset, classnames, indices


def load_imagenet_v2(data_dir=None, val_size=None):
    """Load ImageNet-V2 (matched frequency variant).
    
    Three ways to load:
    1. --data-dir pointing to extracted folder (ImageFolder format)
    2. Auto-download tar.gz from HuggingFace via huggingface_hub
    3. imagenetv2_pytorch package
    """
    from torchvision.datasets import ImageFolder
    
    # Option 1: User provides data dir
    if data_dir and os.path.exists(data_dir):
        dataset = ImageFolder(data_dir)
        classnames = download_imagenet_classnames()
        if classnames is None:
            classnames = [str(i) for i in range(1000)]
        print(f"  Loaded ImageNet-V2 from {data_dir}: {len(dataset)} images")
        
        if val_size and val_size < len(dataset):
            indices = np.random.RandomState(42).permutation(len(dataset))[:val_size]
        else:
            indices = np.arange(len(dataset))
        return dataset, classnames, indices
    
    # Option 2: Download tar.gz from HuggingFace repo
    download_dir = data_dir or "./data/imagenet-v2"
    extracted_dir = os.path.join(download_dir, "imagenetv2-matched-frequency-format-val")
    
    if not os.path.exists(extracted_dir):
        os.makedirs(download_dir, exist_ok=True)
        tar_path = os.path.join(download_dir, "imagenetv2-matched-frequency.tar.gz")
        
        if not os.path.exists(tar_path):
            try:
                from huggingface_hub import hf_hub_download
                print("  Downloading ImageNet-V2 from HuggingFace (~1.26GB)...")
                tar_path = hf_hub_download(
                    repo_id="vaishaal/ImageNetV2",
                    filename="imagenetv2-matched-frequency.tar.gz",
                    repo_type="dataset",
                    local_dir=download_dir,
                )
            except ImportError:
                raise RuntimeError(
                    "Please install huggingface_hub: pip install huggingface_hub\n"
                    "Or download manually:\n"
                    "  1. Go to https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main\n"
                    "  2. Download imagenetv2-matched-frequency.tar.gz\n"
                    "  3. Extract and pass --data-dir to the extracted folder"
                )
        
        print(f"  Extracting...")
        import tarfile, zipfile
        # Auto-detect format (HF xet storage may strip gzip)
        if tarfile.is_tarfile(tar_path):
            with tarfile.open(tar_path) as tar:
                tar.extractall(download_dir)
        elif zipfile.is_zipfile(tar_path):
            with zipfile.ZipFile(tar_path, 'r') as zf:
                zf.extractall(download_dir)
        else:
            raise RuntimeError(f"Cannot extract {tar_path} — unknown format")
        
        # Find the extracted folder
        if not os.path.exists(extracted_dir):
            for d in os.listdir(download_dir):
                full = os.path.join(download_dir, d)
                if os.path.isdir(full) and "imagenetv2" in d.lower():
                    extracted_dir = full
                    break
        
        print(f"  Extracted to {extracted_dir}")
    
    dataset = ImageFolder(extracted_dir)
    classnames = download_imagenet_classnames()
    if classnames is None:
        classnames = [str(i) for i in range(1000)]
    
    print(f"  Loaded ImageNet-V2: {len(dataset)} images, {len(classnames)} classes")
    
    if val_size and val_size < len(dataset):
        indices = np.random.RandomState(42).permutation(len(dataset))[:val_size]
    else:
        indices = np.arange(len(dataset))
    
    return dataset, classnames, indices


def load_domainnet(data_dir, domain="real", val_size=None):
    """Load DomainNet dataset for a specific domain."""
    from torchvision.datasets import ImageFolder
    
    domain_dir = os.path.join(data_dir, domain)
    if not os.path.exists(domain_dir):
        raise RuntimeError(
            f"DomainNet domain '{domain}' not found at {domain_dir}.\n"
            "Please download DomainNet from http://ai.bu.edu/M3SDA/ "
            "and extract to --data-dir."
        )
    
    dataset = ImageFolder(domain_dir)
    classnames = [name.replace('_', ' ') for name in dataset.classes]
    
    print(f"  Loaded DomainNet/{domain}: {len(dataset)} images, {len(classnames)} classes")
    
    if val_size and val_size < len(dataset):
        indices = np.random.RandomState(42).permutation(len(dataset))[:val_size]
    else:
        indices = np.arange(len(dataset))
    
    return dataset, classnames, indices


# ══════════════════════════════════════════════════════════════════════
# CLIP encoding
# ══════════════════════════════════════════════════════════════════════

def load_clip_model(model_name, device):
    """Load CLIP model via open_clip."""
    import open_clip
    
    # Parse model name
    model_map = {
        "ViT-L/14": ("ViT-L-14", "openai"),
        "ViT-B/32": ("ViT-B-32", "openai"),
        "ViT-B/16": ("ViT-B-16", "openai"),
    }
    
    if model_name in model_map:
        model_str, pretrained = model_map[model_name]
    else:
        model_str = model_name
        pretrained = "openai"
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_str, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_str)
    model.eval()
    
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, dataset, indices, device, batch_size=256):
    """Encode images with CLIP, return normalized features."""
    from torch.utils.data import DataLoader, Subset
    
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    all_features = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(loader):
        # images are already PIL → preprocessed by DataLoader
        # But ImageFolder returns PIL, we need to preprocess
        if not isinstance(images, torch.Tensor):
            images = torch.stack([preprocess(img) for img in images])
        else:
            # Re-preprocess if needed
            pass
        
        images = images.to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
        all_labels.append(labels)
        
        if (batch_idx + 1) % 20 == 0:
            n_done = min((batch_idx + 1) * batch_size, len(indices))
            print(f"  Encoded {n_done}/{len(indices)} images...")
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


@torch.no_grad()
def encode_images_with_preprocess(model, preprocess, dataset, indices, device, batch_size=256):
    """Encode images with explicit preprocessing (for ImageFolder datasets)."""
    all_features = []
    all_labels = []
    
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]
        
        images = []
        labels = []
        for idx in batch_indices:
            img, label = dataset[idx]
            images.append(preprocess(img))
            labels.append(label)
        
        images = torch.stack(images).to(device)
        features = model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
        all_labels.append(torch.tensor(labels))
        
        if (start // batch_size + 1) % 20 == 0:
            print(f"  Encoded {end}/{len(indices)} images...")
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return features, labels


@torch.no_grad()
def encode_text_prompts(model, tokenizer, prompts_per_class, classnames, device):
    """Encode text prompts and return normalized class embeddings."""
    import open_clip
    
    class_embeddings = []
    for cls in classnames:
        info = prompts_per_class[cls]
        prompts = info["prompts"]
        weights = info["weights"]
        
        tokens = tokenizer(prompts).to(device)
        embeddings = model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Weighted average
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_t = weights_t / weights_t.sum()
        class_emb = (embeddings * weights_t.unsqueeze(1)).sum(dim=0)
        class_emb = class_emb / class_emb.norm()
        
        class_embeddings.append(class_emb)
    
    return torch.stack(class_embeddings)  # (n_classes, dim)


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_accuracy(image_features, labels, class_embeddings):
    """Compute top-1 and top-5 accuracy."""
    # image_features: (n_images, dim), class_embeddings: (n_classes, dim)
    sims = (image_features @ class_embeddings.T)  # (n_images, n_classes)
    
    # Top-1
    preds = sims.argmax(dim=1)
    top1 = (preds == labels).float().mean().item()
    
    # Top-5
    top5_preds = sims.topk(5, dim=1).indices
    top5 = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    return {"top1": top1, "top5": top5}


# ══════════════════════════════════════════════════════════════════════
# LLM description generation (batched for 1000 classes)
# ══════════════════════════════════════════════════════════════════════

def generate_descriptions(classnames, llm_model, llm_provider, cache_dir="./data/descriptions"):
    """Generate LLM descriptions for all classes, with caching."""
    cache_path = Path(cache_dir) / f"descriptions_{len(classnames)}classes_{llm_model.replace('/', '_')}.json"
    
    if cache_path.exists():
        print(f"  Loading cached descriptions from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Verify all classes present
        missing = [c for c in classnames if c not in cached]
        if not missing:
            return cached, {"total_cost_usd": 0, "cached": True}
        print(f"  {len(missing)} classes missing from cache, regenerating those...")
    else:
        cached = {}
        missing = classnames
    
    client = LLMClient(model=llm_model, provider=llm_provider, temperature=0.7)
    total_cost = 0
    
    # Batch by 10 classes
    for i in range(0, len(missing), 10):
        batch = missing[i:i+10]
        batch_str = ", ".join(batch)
        
        prompt = (
            f"For each category below, provide 10-15 short visual descriptions "
            f"suitable for CLIP-based image classification. Focus on distinctive "
            f"visual attributes: color, shape, texture, size, parts, habitat/setting. "
            f"Each description should be a complete sentence that could caption a photo.\n\n"
            f"Categories: {batch_str}\n\n"
            f'Return ONLY valid JSON: {{"category": ["description1", "description2", ...]}}\n'
        )
        
        try:
            response = client.call(prompt, json_mode=True)
            descs = json.loads(response)
            
            for cn in batch:
                found = None
                for key in descs:
                    if cn.lower().strip() in key.lower():
                        found = descs[key]
                        break
                if found and isinstance(found, list):
                    cached[cn] = found
                else:
                    cached[cn] = [f"a photo of a {cn}"]
        except Exception as e:
            logger.warning(f"  Batch {i//10 + 1} failed: {e}")
            for cn in batch:
                cached[cn] = [f"a photo of a {cn}"]
        
        if (i // 10 + 1) % 10 == 0:
            print(f"  Generated descriptions for {min(i+10, len(missing))}/{len(missing)} classes...")
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(cached, f, indent=1)
    print(f"  Cached descriptions to {cache_path}")
    
    cost = client.total_cost if hasattr(client, 'total_cost') else 0
    return cached, {"total_cost_usd": cost}


# ══════════════════════════════════════════════════════════════════════
# Build prompt configs
# ══════════════════════════════════════════════════════════════════════

def build_template_prompts(classnames):
    """80-template ensemble baseline."""
    return {
        cls: {
            "prompts": [t.format(cls) for t in IMAGENET_TEMPLATES],
            "weights": [1.0] * len(IMAGENET_TEMPLATES),
        }
        for cls in classnames
    }


def build_netra_prompts(classnames, descriptions, alpha, beta):
    """NETRA group-normalized fusion."""
    ppc = {}
    for cls in classnames:
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        
        M = len(templates)
        N = len(descs)
        
        template_weights = [alpha / M] * M if M > 0 and alpha > 0 else []
        desc_weights = [beta / N] * N if N > 0 and beta > 0 else []
        
        prompts = (templates if alpha > 0 else []) + (descs if beta > 0 else [])
        weights = template_weights + desc_weights
        
        if not prompts:
            prompts = [f"a photo of a {cls}"]
            weights = [1.0]
        
        ppc[cls] = {"prompts": prompts, "weights": weights}
    
    return ppc


def build_cupl_prompts(classnames, descriptions):
    """CuPL+e: uniform weight over templates + descriptions."""
    ppc = {}
    for cls in classnames:
        templates = [t.format(cls) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(cls, [])
        all_p = templates + descs
        ppc[cls] = {"prompts": all_p, "weights": [1.0] * len(all_p)}
    return ppc


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NETRA on ImageNet/V2/DomainNet")
    parser.add_argument("--dataset", required=True, choices=["imagenet", "imagenet-v2", "domainnet"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--domain", type=str, default="real", help="DomainNet domain")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--val-size", type=int, default=None, help="Subsample for quick testing")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip baselines, only run NETRA sweep")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    print(f"\n{'='*70}")
    print(f"  NETRA — {args.dataset.upper()} Evaluation")
    print(f"  CLIP: {args.clip_model} | LLM: {args.llm} | Device: {args.device}")
    print(f"{'='*70}\n")
    
    # ── Load dataset ──────────────────────────────────────────────────
    print("Loading dataset...")
    if args.dataset == "imagenet":
        if not args.data_dir:
            raise ValueError("ImageNet requires --data-dir pointing to val set (ImageFolder format)")
        dataset, classnames, indices = load_imagenet(args.data_dir, args.val_size)
    elif args.dataset == "imagenet-v2":
        dataset, classnames, indices = load_imagenet_v2(args.data_dir, args.val_size)
    elif args.dataset == "domainnet":
        if not args.data_dir:
            raise ValueError("DomainNet requires --data-dir")
        dataset, classnames, indices = load_domainnet(args.data_dir, args.domain, args.val_size)
    
    n_classes = len(classnames)
    n_images = len(indices)
    print(f"  {n_classes} classes, {n_images} images\n")
    
    # ── Load CLIP ─────────────────────────────────────────────────────
    print("Loading CLIP model...")
    model, preprocess, tokenizer = load_clip_model(args.clip_model, args.device)
    
    # ── Encode images ─────────────────────────────────────────────────
    print("Encoding images...")
    t0 = time.time()
    image_features, labels = encode_images_with_preprocess(
        model, preprocess, dataset, indices, args.device, args.batch_size
    )
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {image_features.shape}\n")
    
    results = {}
    
    # ── Baseline 1: Class name only ───────────────────────────────────
    if not args.skip_baselines:
        print("--- Baseline 1: Class name only ---")
        ppc = {cls: {"prompts": [cls], "weights": [1.0]} for cls in classnames}
        class_emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
        metrics = evaluate_accuracy(image_features, labels, class_emb)
        print(f"  Top-1: {metrics['top1']*100:.2f}%  Top-5: {metrics['top5']*100:.2f}%\n")
        results["class_name_only"] = metrics
    
    # ── Baseline 2: "a photo of a {class}" ────────────────────────────
    if not args.skip_baselines:
        print('--- Baseline 2: "a photo of a {class}" ---')
        ppc = {cls: {"prompts": [f"a photo of a {cls}"], "weights": [1.0]} for cls in classnames}
        class_emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
        metrics = evaluate_accuracy(image_features, labels, class_emb)
        print(f"  Top-1: {metrics['top1']*100:.2f}%  Top-5: {metrics['top5']*100:.2f}%\n")
        results["single_template"] = metrics
    
    # ── Baseline 3: 80-template ensemble ──────────────────────────────
    print("--- Baseline 3: 80-template ensemble ---")
    ppc = build_template_prompts(classnames)
    class_emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    metrics = evaluate_accuracy(image_features, labels, class_emb)
    baseline_top1 = metrics["top1"]
    print(f"  Top-1: {metrics['top1']*100:.2f}%  Top-5: {metrics['top5']*100:.2f}%\n")
    results["80_templates"] = metrics
    
    # ── Generate descriptions ─────────────────────────────────────────
    print("--- Generating LLM descriptions ---")
    descriptions, desc_cost = generate_descriptions(classnames, args.llm, args.llm_provider)
    n_with_desc = sum(1 for c in classnames if c in descriptions and len(descriptions[c]) > 1)
    print(f"  {n_with_desc}/{n_classes} classes with descriptions "
          f"(cost: ${desc_cost.get('total_cost_usd', 0):.4f})\n")
    
    # ── Baseline 4: CuPL+e (uniform weight) ──────────────────────────
    if not args.skip_baselines:
        print("--- Baseline 4: CuPL+e (uniform weight) ---")
        ppc = build_cupl_prompts(classnames, descriptions)
        class_emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
        metrics = evaluate_accuracy(image_features, labels, class_emb)
        print(f"  Top-1: {metrics['top1']*100:.2f}%  Top-5: {metrics['top5']*100:.2f}%  "
              f"Δ: {(metrics['top1']-baseline_top1)*100:+.2f}%\n")
        results["cupl_ensemble"] = metrics
    
    # ── NETRA weight sweep ────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  NETRA WEIGHT SWEEP")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Top-1':>8} {'Top-5':>8} {'Δ Top-1':>10}")
    print(f"{'-'*55}")
    
    sweep_configs = [
        (1.0, 0.0, "100/0 (templates only)"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45 (default)"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100 (desc only)"),
    ]
    
    sweep_results = []
    for alpha, beta, label in sweep_configs:
        ppc = build_netra_prompts(classnames, descriptions, alpha, beta)
        class_emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
        metrics = evaluate_accuracy(image_features, labels, class_emb)
        delta = metrics["top1"] - baseline_top1
        
        print(f"  {label:<23} {metrics['top1']*100:>7.2f}% {metrics['top5']*100:>7.2f}% {delta*100:>+9.2f}%")
        
        sweep_results.append({
            "alpha": alpha, "beta": beta, "label": label,
            **metrics, "delta_top1": delta,
        })
    
    best = max(sweep_results, key=lambda x: x["top1"])
    print(f"\n  Best: {best['label']} → Top-1={best['top1']*100:.2f}% "
          f"(Δ={best['delta_top1']*100:+.2f}% vs templates)")
    results["netra_sweep"] = sweep_results
    results["netra_best"] = best
    
    # Fixed default (55/45)
    default = [r for r in sweep_results if r["alpha"] == 0.55][0]
    results["netra_default"] = default
    print(f"  Default (55/45): Top-1={default['top1']*100:.2f}% "
          f"(Δ={default['delta_top1']*100:+.2f}%)\n")
    
    # ── Summary ───────────────────────────────────────────────────────
    dataset_name = args.dataset
    if args.dataset == "domainnet":
        dataset_name = f"domainnet_{args.domain}"
    
    print(f"{'='*60}")
    print(f"  SUMMARY — {dataset_name}")
    print(f"{'='*60}")
    print(f"  80 templates:     {results['80_templates']['top1']*100:.2f}%")
    if "cupl_ensemble" in results:
        print(f"  CuPL+e:           {results['cupl_ensemble']['top1']*100:.2f}%")
    print(f"  NETRA (default):  {default['top1']*100:.2f}% (Δ={default['delta_top1']*100:+.2f}%)")
    print(f"  NETRA (best):     {best['top1']*100:.2f}% (Δ={best['delta_top1']*100:+.2f}%) @ {best['label']}")
    print(f"{'='*60}\n")
    
    # ── Save ──────────────────────────────────────────────────────────
    save_data = {
        "dataset": dataset_name,
        "clip_model": args.clip_model,
        "llm": args.llm,
        "n_images": n_images,
        "n_classes": n_classes,
        "results": {}
    }
    
    for k, v in results.items():
        if isinstance(v, dict):
            save_data["results"][k] = v
        elif isinstance(v, list):
            save_data["results"][k] = v
    
    output_path = Path(args.output_dir) / f"imagenet_{dataset_name}_{args.clip_model.replace('/', '_')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
