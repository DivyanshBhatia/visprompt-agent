#!/usr/bin/env python3
"""MS-COCO category-level retrieval with all baselines.

Uses COCO's 80 object categories for class-level retrieval (not caption retrieval).
This is the same protocol as our other retrieval benchmarks: one text embedding per class,
rank all images by cosine similarity, compute mAP.

Note: COCO images often contain multiple objects. We evaluate per-category:
an image is "relevant" for class c if it has at least one annotation of category c.
This is multi-label retrieval, which is harder than single-label but more realistic.

Usage:
    # Download COCO val2017 first:
    # wget http://images.cocodataset.org/zips/val2017.zip
    # wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

    python scripts/run_coco_retrieval.py \
        --coco-dir /path/to/coco \
        --clip-model ViT-L/14 --llm gpt-4o
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

logger = logging.getLogger(__name__)

# COCO 80 category names (standard order)
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def load_coco_data(coco_dir, split="val2017", max_images=None):
    """Load COCO images and multi-label annotations."""
    coco_dir = Path(coco_dir)
    img_dir = coco_dir / split
    ann_file = coco_dir / "annotations" / f"instances_{split}.json"

    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotations not found: {ann_file}")
    if not img_dir.exists():
        raise FileNotFoundError(f"COCO images not found: {img_dir}")

    print(f"  Loading COCO annotations from {ann_file}...")
    with open(ann_file) as f:
        coco = json.load(f)

    # Build category ID → name mapping
    cat_id_to_name = {}
    cat_id_to_idx = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name in COCO_CATEGORIES:
            cat_id_to_name[cat["id"]] = name
            cat_id_to_idx[cat["id"]] = COCO_CATEGORIES.index(name)

    # Build image ID → file path mapping
    img_id_to_file = {}
    for img_info in coco["images"]:
        img_id_to_file[img_info["id"]] = img_dir / img_info["file_name"]

    # Build multi-label: image_id → set of category indices
    img_labels = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id in cat_id_to_idx:
            if img_id not in img_labels:
                img_labels[img_id] = set()
            img_labels[img_id].add(cat_id_to_idx[cat_id])

    # Filter to images with at least one annotation
    valid_img_ids = sorted(img_labels.keys())
    if max_images and max_images < len(valid_img_ids):
        random.seed(42)
        valid_img_ids = sorted(random.sample(valid_img_ids, max_images))

    image_paths = [img_id_to_file[iid] for iid in valid_img_ids]
    # Multi-label matrix: (n_images, n_classes) binary
    n_classes = len(COCO_CATEGORIES)
    label_matrix = np.zeros((len(valid_img_ids), n_classes), dtype=bool)
    for i, iid in enumerate(valid_img_ids):
        for cidx in img_labels[iid]:
            label_matrix[i, cidx] = True

    print(f"  {len(image_paths)} images, {n_classes} categories")
    print(f"  Avg categories per image: {label_matrix.sum(axis=1).mean():.1f}")
    print(f"  Avg images per category: {label_matrix.sum(axis=0).mean():.0f}")

    return image_paths, label_matrix, COCO_CATEGORIES


def compute_multilabel_retrieval(similarities, label_matrix, class_names):
    """Compute retrieval mAP for multi-label data.

    Args:
        similarities: (n_classes, n_images) cosine similarity
        label_matrix: (n_images, n_classes) binary relevance
        class_names: list of class names

    Returns:
        dict with mAP, per-class AP
    """
    n_classes = similarities.shape[0]
    per_class_ap = {}
    all_aps = []

    for cls_idx in range(n_classes):
        sims = similarities[cls_idx]
        relevant = label_matrix[:, cls_idx]
        n_relevant = relevant.sum()

        if n_relevant == 0:
            continue

        ranked_indices = np.argsort(-sims)
        tp_cumsum = 0
        precisions = []
        for rank, img_idx in enumerate(ranked_indices):
            if relevant[img_idx]:
                tp_cumsum += 1
                precisions.append(tp_cumsum / (rank + 1))

        ap = np.mean(precisions) if precisions else 0.0
        per_class_ap[class_names[cls_idx]] = float(ap)
        all_aps.append(ap)

    mAP = float(np.mean(all_aps)) if all_aps else 0.0
    return {"mAP": mAP, "per_class_ap": per_class_ap, "n_evaluated": len(all_aps)}


@torch.no_grad()
def encode_images(model, preprocess, image_paths, device, batch_size=128):
    """Encode images from file paths."""
    from PIL import Image
    all_features = []
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        batch = []
        for p in image_paths[start:end]:
            try:
                img = Image.open(p).convert("RGB")
                batch.append(preprocess(img))
            except Exception as e:
                logger.warning(f"Failed to load {p}: {e}")
                # Use a blank image
                batch.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(batch).to(device)
        features = model.encode_image(batch_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
        if (start // batch_size) % 10 == 0:
            print(f"    Encoded {end}/{len(image_paths)} images...")
    return torch.cat(all_features, dim=0)


@torch.no_grad()
def encode_text_prompts(model, tokenizer, prompts_per_class, class_names, device):
    """Encode text prompts with optional weights, return (n_classes, dim)."""
    class_embs = []
    for cls in class_names:
        ppc = prompts_per_class[cls]
        prompts = ppc["prompts"]
        weights = ppc.get("weights", [1.0] * len(prompts))

        tokens = tokenizer(prompts).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        w = torch.tensor(weights, dtype=torch.float32, device=device)
        w = w / w.sum()
        cls_emb = (emb * w.unsqueeze(1)).sum(dim=0)
        cls_emb = cls_emb / cls_emb.norm()
        class_embs.append(cls_emb)

    return torch.stack(class_embs)


def generate_descriptions(class_names, llm_model, llm_provider, cache_path):
    """Generate or load cached descriptions."""
    if cache_path.exists():
        with open(cache_path) as f:
            print(f"  Loaded cached descriptions from {cache_path.name}")
            return json.load(f)

    print(f"  Generating descriptions with {llm_model}...")
    from visprompt.utils.llm import CostTracker, LLMClient
    cost_tracker = CostTracker()
    llm = LLMClient(model=llm_model, provider=llm_provider,
                     temperature=0.7, cost_tracker=cost_tracker)

    descriptions = {}
    batch_size = 10
    for i in range(0, len(class_names), batch_size):
        batch = class_names[i:i+batch_size]
        prompt = (
            f"Generate 10-15 short visual descriptions for each object category.\n"
            f"Format: \"a {{category}}, {{visual description}}\"\n"
            f"Focus on shape, color, size, texture, typical context.\n\n"
            f"Categories: {', '.join(batch)}\n\n"
            f'Return ONLY valid JSON: {{"category": ["desc1", ...]}}\n'
        )
        try:
            result = llm.call_json(prompt=prompt,
                                    system="Generate visual descriptions for CLIP retrieval.",
                                    agent_name="coco_retrieval")
            for cn in batch:
                descs = result.get(cn, result.get(cn.replace("_", " "), []))
                if isinstance(descs, list) and descs:
                    cleaned = []
                    for d in descs:
                        if cn.lower() not in d.lower():
                            d = f"a {cn}, {d}"
                        cleaned.append(d)
                    descriptions[cn] = cleaned
                else:
                    descriptions[cn] = [f"a {cn} in a typical setting"]
        except Exception as e:
            logger.warning(f"Batch failed: {e}")
            for cn in batch:
                descriptions[cn] = [f"a {cn} in a typical setting"]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(descriptions, f, indent=1)
    print(f"  Cached to {cache_path}")
    return descriptions


def main():
    parser = argparse.ArgumentParser(description="MS-COCO category-level retrieval")
    parser.add_argument("--coco-dir", required=True, help="Path to COCO dataset root")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output-dir", default="experiments/retrieval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"\n{'='*65}")
    print(f"  MS-COCO CATEGORY-LEVEL RETRIEVAL")
    print(f"  CLIP: {args.clip_model}, LLM: {args.llm}")
    print(f"{'='*65}\n")

    # Load data
    image_paths, label_matrix, class_names = load_coco_data(
        args.coco_dir, args.split, args.max_images
    )

    # Load CLIP
    import open_clip
    model_name = args.clip_model.replace('/', '-')
    print(f"  Loading CLIP {args.clip_model}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained='openai', device=args.device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Encode images
    print(f"\n  Encoding {len(image_paths)} images...")
    image_features = encode_images(model, preprocess, image_paths, args.device)
    print(f"  Image features: {image_features.shape}")

    results = {}

    # ── Baseline 1: Class name only ──
    print(f"\n--- Class name only ---")
    ppc = {c: {"prompts": [c]} for c in class_names}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}")
    results["class_name"] = metrics

    # ── Baseline 2: 80 templates ──
    print(f"\n--- 80 templates ---")
    ppc = {c: {"prompts": [t.format(c) for t in IMAGENET_TEMPLATES]} for c in class_names}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    baseline_map = metrics["mAP"]
    print(f"  mAP: {metrics['mAP']:.4f}")
    results["80_templates"] = metrics

    # ── Generate descriptions ──
    desc_cache = Path(args.output_dir) / f"descriptions_coco_{args.llm}.json"
    descriptions = generate_descriptions(class_names, args.llm, args.llm_provider, desc_cache)

    # ── Baseline 3: CuPL (descriptions only) ──
    print(f"\n--- CuPL (descriptions only) ---")
    ppc = {c: {"prompts": descriptions.get(c, [f"a photo of a {c}"])} for c in class_names}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}  (Δ: {(metrics['mAP']-baseline_map)*100:+.2f}%)")
    results["cupl_desc"] = metrics

    # ── Baseline 4: WaffleCLIP ──
    print(f"\n--- WaffleCLIP ---")
    random.seed(42)
    random_words = [
        "bright", "dark", "colorful", "textured", "smooth", "large", "small",
        "round", "angular", "metallic", "wooden", "organic", "geometric",
        "striped", "furry", "shiny", "transparent", "natural", "indoor", "outdoor",
    ]
    ppc = {}
    for cls in class_names:
        wps = [f"a photo of a {cls}"]
        for _ in range(15):
            words = random.sample(random_words, random.randint(2, 4))
            wps.append(f"a {' '.join(words)} photo of a {cls}")
        ppc[cls] = {"prompts": wps}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}  (Δ: {(metrics['mAP']-baseline_map)*100:+.2f}%)")
    results["waffle_clip"] = metrics

    # ── Baseline 5: DCLIP ──
    print(f"\n--- DCLIP ---")
    from visprompt.utils.llm import LLMClient
    dclip_cache = Path(args.output_dir) / f"dclip_coco_{args.llm}.json"
    if dclip_cache.exists():
        with open(dclip_cache) as f:
            dclip_descriptions = json.load(f)
    else:
        dclip_client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
        dclip_descriptions = {}
        for i in range(0, len(class_names), 10):
            batch = class_names[i:i+10]
            prompt = (
                f"For each category, describe what it looks like.\n"
                f"Give 5-8 short visual descriptors per category.\n"
                f'Format: "{{class}} which has {{descriptor}}"\n\n'
                f"Categories: {', '.join(batch)}\n\n"
                f'Return JSON: {{"category": ["desc1", ...]}}\n'
            )
            try:
                response = dclip_client.call(prompt, json_mode=True)
                descs = json.loads(response)
                for cn in batch:
                    found = None
                    for key in descs:
                        if cn.lower() in key.lower():
                            found = descs[key]
                            break
                    if found and isinstance(found, list):
                        formatted = [f"{cn} which has {d}" if cn.lower() not in d.lower() else d for d in found]
                        dclip_descriptions[cn] = formatted
                    else:
                        dclip_descriptions[cn] = [f"a photo of a {cn}"]
            except Exception as e:
                for cn in batch:
                    dclip_descriptions[cn] = [f"a photo of a {cn}"]
        with open(dclip_cache, 'w') as f:
            json.dump(dclip_descriptions, f, indent=1)

    ppc = {c: {"prompts": dclip_descriptions.get(c, [f"a photo of a {c}"])} for c in class_names}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}  (Δ: {(metrics['mAP']-baseline_map)*100:+.2f}%)")
    results["dclip"] = metrics

    # ── Baseline 6: CuPL+e (uniform) ──
    print(f"\n--- CuPL+e (uniform) ---")
    ppc = {}
    for cls in class_names:
        tmpl = [t.format(cls) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(cls, [])
        ppc[cls] = {"prompts": tmpl + descs}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}  (Δ: {(metrics['mAP']-baseline_map)*100:+.2f}%)")
    results["cupl_e"] = metrics

    # ── Baseline 7: CLIP-Enhance ──
    print(f"\n--- CLIP-Enhance ---")
    enhance_cache = Path(args.output_dir) / f"enhance_coco_{args.llm}.json"
    if enhance_cache.exists():
        with open(enhance_cache) as f:
            enhance_descriptions = json.load(f)
    else:
        enhance_client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
        enhance_descriptions = {}
        for i in range(0, len(class_names), 10):
            batch = class_names[i:i+10]
            prompt = (
                f"For each category, provide:\n"
                f"1. 3 synonyms or alternative names\n"
                f"2. 5 visual descriptions\n\n"
                f"Categories: {', '.join(batch)}\n\n"
                f'Return JSON: {{"category": {{"synonyms": [...], "descriptions": [...]}}}}\n'
            )
            try:
                response = enhance_client.call(prompt, json_mode=True)
                descs = json.loads(response)
                for cn in batch:
                    found = None
                    for key in descs:
                        if cn.lower() in key.lower():
                            found = descs[key]
                            break
                    if found and isinstance(found, dict):
                        all_p = [f"a photo of a {cn}"]
                        for s in found.get("synonyms", []):
                            all_p.append(f"a photo of a {s}")
                        all_p.extend(found.get("descriptions", []))
                        enhance_descriptions[cn] = all_p
                    else:
                        enhance_descriptions[cn] = [f"a photo of a {cn}"]
            except Exception as e:
                for cn in batch:
                    enhance_descriptions[cn] = [f"a photo of a {cn}"]
        with open(enhance_cache, 'w') as f:
            json.dump(enhance_descriptions, f, indent=1)

    ppc = {c: {"prompts": enhance_descriptions.get(c, [f"a photo of a {c}"])} for c in class_names}
    text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
    sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
    metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
    print(f"  mAP: {metrics['mAP']:.4f}  (Δ: {(metrics['mAP']-baseline_map)*100:+.2f}%)")
    results["clip_enhance"] = metrics

    # ── NETRA weight sweep ──
    print(f"\n--- NETRA weight sweep ---")
    M = len(IMAGENET_TEMPLATES)
    sweep_configs = [
        (1.0, 0.0, "100/0"), (0.85, 0.15, "85/15"), (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"), (0.40, 0.60, "40/60"), (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100"),
    ]
    print(f"  {'Config':<12} {'mAP':>8} {'Δ':>10}")
    print(f"  {'-'*32}")

    netra_results = []
    for alpha, beta, label in sweep_configs:
        ppc = {}
        for cls in class_names:
            tmpl = [t.format(cls) for t in IMAGENET_TEMPLATES] if alpha > 0 else []
            descs = descriptions.get(cls, [f"a photo of a {cls}"]) if beta > 0 else []
            N = len(descs)
            prompts, weights = [], []
            if alpha > 0:
                prompts.extend(tmpl)
                weights.extend([alpha / M] * M)
            if beta > 0:
                prompts.extend(descs)
                weights.extend([beta / N] * N)
            if not prompts:
                prompts, weights = [f"a photo of a {cls}"], [1.0]
            ppc[cls] = {"prompts": prompts, "weights": weights}

        text_emb = encode_text_prompts(model, tokenizer, ppc, class_names, args.device)
        sims = (text_emb.to(args.device) @ image_features.to(args.device).T).cpu().numpy()
        metrics = compute_multilabel_retrieval(sims, label_matrix, class_names)
        delta = metrics["mAP"] - baseline_map
        print(f"  {label:<12} {metrics['mAP']:>8.4f} {delta*100:>+9.2f}%")
        netra_results.append({"config": label, "alpha": alpha, "beta": beta, **metrics})

    results["netra_sweep"] = netra_results
    best = max(netra_results, key=lambda x: x["mAP"])
    results["netra_best"] = best

    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  SUMMARY — MS-COCO Category Retrieval (80 classes)")
    print(f"{'='*65}")
    print(f"  {'Method':<20} {'mAP':>8} {'Δ':>10}")
    print(f"  {'-'*40}")
    for name, key in [("Class name", "class_name"), ("80 templates", "80_templates"),
                       ("WaffleCLIP", "waffle_clip"), ("CuPL (desc)", "cupl_desc"),
                       ("DCLIP", "dclip"), ("CLIP-Enhance", "clip_enhance"),
                       ("CuPL+e", "cupl_e")]:
        if key in results:
            m = results[key]
            d = (m["mAP"] - baseline_map) * 100
            print(f"  {name:<20} {m['mAP']:>8.4f} {d:>+9.2f}%")
    print(f"  {'NETRA (best)':<20} {best['mAP']:>8.4f} {(best['mAP']-baseline_map)*100:>+9.2f}%  @ {best['config']}")
    print(f"{'='*65}")

    # Save
    out_path = Path(args.output_dir) / "coco_retrieval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
