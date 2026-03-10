#!/usr/bin/env python3
"""NETRA on Kinetics-400 and HMDB-51 (middle-frame extraction).

Same single-frame protocol as UCF-101: extract middle frame, classify with CLIP.
This is NOT a temporal understanding evaluation — see paper Section 4.4 for discussion.

Usage:
    # HMDB-51 (small, ~1.5K test videos, quick)
    python scripts/run_video_benchmarks.py --dataset hmdb51 \
        --data-dir /path/to/hmdb51 --clip-model ViT-L/14 --llm gpt-4o

    # Kinetics-400 (large, use --val-size for subset)
    python scripts/run_video_benchmarks.py --dataset kinetics400 \
        --data-dir /path/to/kinetics400/val \
        --clip-model ViT-L/14 --llm gpt-4o --val-size 10000

Data format:
    HMDB-51: hmdb51/brush_hair/*.avi, hmdb51/cartwheel/*.avi, ...
    Kinetics-400: kinetics400/val/abseiling/*.mp4, kinetics400/val/air_drumming/*.mp4, ...
    Both should be in ImageFolder-like structure with class subfolders.
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

# Action-specific templates (same 10 as UCF-101 + 80 ImageNet)
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


def extract_middle_frame(video_path):
    """Extract the middle frame from a video file."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # Try reading frames manually
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            if len(frames) == 0:
                return None
            middle = frames[len(frames) // 2]
            from PIL import Image
            return Image.fromarray(cv2.cvtColor(middle, cv2.COLOR_BGR2RGB))

        mid_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        from PIL import Image
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as e:
        logger.warning(f"Failed to read {video_path}: {e}")
        return None


def load_video_dataset(data_dir, dataset_name, val_size=None, split="test"):
    """Load video dataset by extracting middle frames.

    Supports two formats:
    1. Video files: data_dir/class_name/video.avi → extract middle frame
    2. Frame folders: data_dir/class_name/video_clip_folder/frame_001.jpg → pick middle frame
    """
    from PIL import Image
    data_path = Path(data_dir)

    # Find class folders
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if len(class_dirs) == 0:
        for s in [split, "val", "test", "testing"]:
            sp = data_path / s
            if sp.exists():
                class_dirs = sorted([d for d in sp.iterdir() if d.is_dir()])
                if len(class_dirs) > 0:
                    data_path = sp
                    break

    if len(class_dirs) == 0:
        raise RuntimeError(f"No class folders found in {data_dir}")

    classnames = [d.name.replace('_', ' ') for d in class_dirs]
    class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

    video_extensions = {'.avi', '.mp4', '.mkv', '.mov', '.webm'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Detect format: check first class folder
    first_class = class_dirs[0]
    first_children = list(first_class.iterdir())

    # Check if children are subdirectories (frame folders) or files
    has_subdirs = any(c.is_dir() for c in first_children)
    has_videos = any(c.suffix.lower() in video_extensions for c in first_children if c.is_file())
    has_images = any(c.suffix.lower() in image_extensions for c in first_children if c.is_file())

    if has_subdirs:
        # Format: class_dir/clip_folder/frame_001.jpg → pick middle frame per clip
        print(f"  Detected frame-folder format (e.g., HMDB-51 extracted frames)")
        all_clips = []
        for cls_dir in class_dirs:
            for clip_dir in sorted(cls_dir.iterdir()):
                if clip_dir.is_dir():
                    all_clips.append((clip_dir, class_to_idx[cls_dir.name]))

        print(f"  Found {len(all_clips)} clips across {len(classnames)} classes")

        if val_size and val_size < len(all_clips):
            random.seed(42)
            all_clips = random.sample(all_clips, val_size)
            print(f"  Subsampled to {val_size} clips")

        print(f"  Picking middle frame from each clip...")
        images = []
        labels = []
        skipped = 0
        for i, (clip_dir, label) in enumerate(all_clips):
            frames = sorted([f for f in clip_dir.iterdir()
                           if f.suffix.lower() in image_extensions])
            if len(frames) == 0:
                skipped += 1
                continue
            mid_frame = frames[len(frames) // 2]
            try:
                img = Image.open(mid_frame).convert('RGB')
                images.append(img)
                labels.append(label)
            except Exception:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(all_clips)} clips processed")

        print(f"  Loaded {len(images)} frames ({skipped} clips skipped)")
        return images, labels, classnames

    elif has_videos:
        # Format: class_dir/video.avi → extract middle frame
        print(f"  Detected video file format")
        all_videos = []
        for cls_dir in class_dirs:
            for f in cls_dir.iterdir():
                if f.suffix.lower() in video_extensions:
                    all_videos.append((f, class_to_idx[cls_dir.name]))

        print(f"  Found {len(all_videos)} videos across {len(classnames)} classes")

        if val_size and val_size < len(all_videos):
            random.seed(42)
            all_videos = random.sample(all_videos, val_size)
            print(f"  Subsampled to {val_size} videos")

        print(f"  Extracting middle frames...")
        images = []
        labels = []
        skipped = 0
        for i, (vpath, label) in enumerate(all_videos):
            frame = extract_middle_frame(vpath)
            if frame is not None:
                images.append(frame)
                labels.append(label)
            else:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(all_videos)} videos processed ({skipped} skipped)")

        print(f"  Extracted {len(images)} frames ({skipped} videos skipped)")
        return images, labels, classnames

    elif has_images:
        # Format: class_dir/image.jpg → just load images directly (already frames)
        print(f"  Detected image file format (pre-extracted single frames)")
        all_images = []
        for cls_dir in class_dirs:
            for f in cls_dir.iterdir():
                if f.suffix.lower() in image_extensions:
                    all_images.append((f, class_to_idx[cls_dir.name]))

        print(f"  Found {len(all_images)} images across {len(classnames)} classes")

        if val_size and val_size < len(all_images):
            random.seed(42)
            all_images = random.sample(all_images, val_size)
            print(f"  Subsampled to {val_size} images")

        images = []
        labels = []
        skipped = 0
        for i, (fpath, label) in enumerate(all_images):
            try:
                img = Image.open(fpath).convert('RGB')
                images.append(img)
                labels.append(label)
            except Exception:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(all_images)} images loaded")

        print(f"  Loaded {len(images)} images ({skipped} skipped)")
        return images, labels, classnames

    else:
        raise RuntimeError(
            f"Could not detect format in {data_dir}. Expected one of:\n"
            f"  1. class_dir/clip_folder/frame.jpg (HMDB-51 extracted)\n"
            f"  2. class_dir/video.avi (video files)\n"
            f"  3. class_dir/image.jpg (pre-extracted frames)"
        )


def load_clip_model(model_name, device):
    """Load CLIP model."""
    import open_clip
    model_map = {
        "ViT-L/14": ("ViT-L-14", "openai"),
        "ViT-B/32": ("ViT-B-32", "openai"),
        "ViT-B/16": ("ViT-B-16", "openai"),
    }
    if model_name in model_map:
        model_str, pretrained = model_map[model_name]
    else:
        model_str, pretrained = model_name, "openai"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_str, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_str)
    model.eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_images(model, preprocess, images, device, batch_size=256):
    """Encode PIL images with CLIP."""
    all_features = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = torch.stack([preprocess(img) for img in images[start:end]]).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
        if (start // batch_size + 1) % 10 == 0:
            print(f"    Encoded {end}/{len(images)} images...")
    return torch.cat(all_features, dim=0)


@torch.no_grad()
def encode_text_prompts(model, tokenizer, prompts_per_class, classnames, device):
    """Encode prompts and return class embeddings."""
    class_embeddings = []
    for cls in classnames:
        info = prompts_per_class[cls]
        tokens = tokenizer(info["prompts"]).to(device)
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        weights = torch.tensor(info["weights"], dtype=torch.float32, device=device)
        weights = weights / weights.sum()
        cls_emb = (emb * weights.unsqueeze(1)).sum(dim=0)
        cls_emb = cls_emb / cls_emb.norm()
        class_embeddings.append(cls_emb)
    return torch.stack(class_embeddings)


def evaluate(image_features, labels, class_embeddings):
    """Compute top-1 and top-5 accuracy."""
    device = class_embeddings.device
    image_features = image_features.to(device)
    labels_t = torch.tensor(labels, device=device)
    sims = image_features @ class_embeddings.T
    preds = sims.argmax(dim=1)
    top1 = (preds == labels_t).float().mean().item()
    top5_preds = sims.topk(min(5, sims.shape[1]), dim=1).indices
    top5 = (top5_preds == labels_t.unsqueeze(1)).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": top5}


def build_action_templates(classnames):
    """Build action templates (10 action-specific + 80 ImageNet)."""
    ppc = {}
    for cls in classnames:
        action = [t.format(cls) for t in ACTION_TEMPLATES]
        imagenet = [t.format(cls) for t in IMAGENET_TEMPLATES]
        all_t = action + imagenet
        ppc[cls] = {"prompts": all_t, "weights": [1.0] * len(all_t)}
    return ppc


def build_netra_prompts(classnames, descriptions, alpha, beta):
    """NETRA group-normalized fusion with action templates."""
    ppc = {}
    for cls in classnames:
        action = [t.format(cls) for t in ACTION_TEMPLATES]
        imagenet = [t.format(cls) for t in IMAGENET_TEMPLATES]
        templates = action + imagenet
        descs = descriptions.get(cls, [f"a photo of a {cls}"])
        M, N = len(templates), len(descs)
        t_w = [alpha / M] * M if alpha > 0 else []
        d_w = [beta / N] * N if beta > 0 else []
        prompts = (templates if alpha > 0 else []) + (descs if beta > 0 else [])
        weights = t_w + d_w
        if not prompts:
            prompts, weights = [f"a photo of a {cls}"], [1.0]
        ppc[cls] = {"prompts": prompts, "weights": weights}
    return ppc


def generate_action_descriptions(classnames, llm_model, llm_provider, cache_path):
    """Generate action descriptions."""
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    client = LLMClient(model=llm_model, provider=llm_provider, temperature=0.7)
    descriptions = {}

    for i in range(0, len(classnames), 10):
        batch = classnames[i:i+10]
        prompt = (
            f"For each action/activity below, provide 10-15 short visual descriptions "
            f"of what a person doing this action looks like in a photo. "
            f"Focus on: body pose, objects held, clothing, setting, motion cues.\n\n"
            f"Actions: {', '.join(batch)}\n\n"
            f'Return ONLY valid JSON: {{"action": ["description1", ...]}}\n'
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
                descriptions[cn] = found if found and isinstance(found, list) else [f"a photo of a person {cn}"]
        except Exception:
            for cn in batch:
                descriptions[cn] = [f"a photo of a person {cn}"]
        if (i // 10 + 1) % 10 == 0:
            print(f"    Generated for {min(i+10, len(classnames))}/{len(classnames)} classes...")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(descriptions, f, indent=1)
    return descriptions


def main():
    parser = argparse.ArgumentParser(description="NETRA on Kinetics-400 / HMDB-51")
    parser.add_argument("--dataset", required=True, choices=["kinetics400", "hmdb51"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--clip-model", default="ViT-L/14")
    parser.add_argument("--llm", default="gpt-4o")
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="experiments")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print(f"\n{'='*70}")
    print(f"  NETRA — {args.dataset.upper()} (Middle-Frame Protocol)")
    print(f"  CLIP: {args.clip_model} | LLM: {args.llm}")
    print(f"{'='*70}\n")

    # Load dataset
    print("Loading dataset...")
    images, labels, classnames = load_video_dataset(
        args.data_dir, args.dataset, args.val_size
    )
    n_classes = len(classnames)
    n_images = len(images)
    print(f"  {n_classes} classes, {n_images} frames\n")

    # Load CLIP
    print("Loading CLIP model...")
    model, preprocess, tokenizer = load_clip_model(args.clip_model, args.device)

    # Encode images
    print("Encoding frames...")
    t0 = time.time()
    image_features = encode_images(model, preprocess, images, args.device, args.batch_size)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    results = {}

    # --- Baselines ---
    print("--- Class name only ---")
    ppc = {c: {"prompts": [c], "weights": [1.0]} for c in classnames}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%\n")
    results["class_name_only"] = m

    print("--- Single template ---")
    ppc = {c: {"prompts": [f"a photo of a person {c}"], "weights": [1.0]} for c in classnames}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%\n")
    results["single_template"] = m

    print("--- 90 templates (10 action + 80 ImageNet) ---")
    ppc = build_action_templates(classnames)
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    baseline_top1 = m["top1"]
    print(f"  Top-1: {m['top1']*100:.2f}%\n")
    results["90_templates"] = m

    # WaffleCLIP
    print("--- WaffleCLIP ---")
    random.seed(42)
    rw = ["bright","dark","colorful","textured","smooth","rough","large","small",
          "round","angular","metallic","wooden","organic","geometric","striped",
          "spotted","furry","shiny","matte","indoor","outdoor","urban","rural"]
    ppc = {}
    for c in classnames:
        wps = [f"a photo of a person {c}"]
        for _ in range(15):
            words = random.sample(rw, random.randint(2, 4))
            wps.append(f"a {' '.join(words)} photo of a person {c}")
        ppc[c] = {"prompts": wps, "weights": [1.0] * len(wps)}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["waffle_clip"] = m

    # Generate descriptions
    print("--- Generating LLM action descriptions ---")
    cache = Path(args.output_dir) / f"action_descriptions_{args.dataset}_{args.llm}.json"
    descriptions = generate_action_descriptions(classnames, args.llm, args.llm_provider, cache)
    print(f"  {sum(1 for c in classnames if c in descriptions)}/{n_classes} classes\n")

    # CuPL (desc only)
    print("--- CuPL (descriptions only) ---")
    ppc = {c: {"prompts": descriptions.get(c, [f"a photo of {c}"]),
               "weights": [1.0] * len(descriptions.get(c, [f"a photo of {c}"]))}
           for c in classnames}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["cupl_desc_only"] = m

    # CuPL+e
    print("--- CuPL+e (uniform) ---")
    ppc = {}
    for c in classnames:
        tmpl = [t.format(c) for t in ACTION_TEMPLATES] + [t.format(c) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(c, [])
        all_p = tmpl + descs
        ppc[c] = {"prompts": all_p, "weights": [1.0] * len(all_p)}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["cupl_ensemble"] = m

    # DCLIP (attribute descriptors)
    print("--- DCLIP ---")
    dclip_cache = Path(args.output_dir) / f"dclip_action_{args.dataset}_{args.llm}.json"
    if dclip_cache.exists():
        with open(dclip_cache) as f:
            dclip_descriptions = json.load(f)
    else:
        dclip_client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
        dclip_descriptions = {}
        for i in range(0, len(classnames), 10):
            batch = classnames[i:i+10]
            prompt = (
                f"For each action, describe what a person doing it looks like.\n"
                f"Give 5-8 short visual descriptors per action.\n"
                f'Format each as: "a person {{action}} which involves {{descriptor}}"\n\n'
                f"Actions: {', '.join(batch)}\n\n"
                f'Return ONLY valid JSON: {{"action": ["desc1", ...]}}\n'
            )
            try:
                resp = dclip_client.call(prompt, json_mode=True)
                descs = json.loads(resp)
                for cn in batch:
                    found = None
                    for key in descs:
                        if cn.lower().strip() in key.lower():
                            found = descs[key]
                            break
                    if found and isinstance(found, list):
                        formatted = []
                        for d in found:
                            if cn.lower() not in d.lower():
                                d = f"a person {cn} which involves {d}"
                            formatted.append(d)
                        dclip_descriptions[cn] = formatted
                    else:
                        dclip_descriptions[cn] = [f"a photo of a person {cn}"]
            except Exception:
                for cn in batch:
                    dclip_descriptions[cn] = [f"a photo of a person {cn}"]
        dclip_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(dclip_cache, 'w') as f:
            json.dump(dclip_descriptions, f, indent=1)

    ppc = {c: {"prompts": dclip_descriptions.get(c, [f"a photo of a person {c}"]),
               "weights": [1.0] * len(dclip_descriptions.get(c, [f"a photo of a person {c}"]))}
           for c in classnames}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["dclip"] = m

    # CLIP-Enhance (synonyms + descriptions)
    print("--- CLIP-Enhance ---")
    enhance_cache = Path(args.output_dir) / f"enhance_action_{args.dataset}_{args.llm}.json"
    if enhance_cache.exists():
        with open(enhance_cache) as f:
            enhance_descriptions = json.load(f)
    else:
        enhance_client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
        enhance_descriptions = {}
        for i in range(0, len(classnames), 10):
            batch = classnames[i:i+10]
            prompt = (
                f"For each action, provide:\n"
                f"1. 3 synonyms or alternative names for this action\n"
                f"2. 5 visual descriptions of what someone doing it looks like\n\n"
                f"Actions: {', '.join(batch)}\n\n"
                f'Return JSON: {{"action": {{"synonyms": ["syn1", ...], "descriptions": ["desc1", ...]}}}}\n'
                f"Return ONLY valid JSON."
            )
            try:
                resp = enhance_client.call(prompt, json_mode=True)
                descs = json.loads(resp)
                for cn in batch:
                    found = None
                    for key in descs:
                        if cn.lower().strip() in key.lower():
                            found = descs[key]
                            break
                    if found and isinstance(found, dict):
                        syns = found.get("synonyms", [])
                        vis = found.get("descriptions", [])
                        all_p = [f"a photo of a person {cn}"]
                        for s in syns:
                            all_p.append(f"a photo of a person {s}")
                        all_p.extend(vis)
                        enhance_descriptions[cn] = all_p
                    else:
                        enhance_descriptions[cn] = [f"a photo of a person {cn}"]
            except Exception:
                for cn in batch:
                    enhance_descriptions[cn] = [f"a photo of a person {cn}"]
        enhance_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(enhance_cache, 'w') as f:
            json.dump(enhance_descriptions, f, indent=1)

    ppc = {c: {"prompts": enhance_descriptions.get(c, [f"a photo of a person {c}"]),
               "weights": [1.0] * len(enhance_descriptions.get(c, [f"a photo of a person {c}"]))}
           for c in classnames}
    emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
    m = evaluate(image_features, labels, emb)
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["clip_enhance"] = m

    # Frolic (CuPL+e + logit correction)
    print("--- Frolic ---")
    tmpl_emb = encode_text_prompts(model, tokenizer, build_action_templates(classnames), classnames, args.device)
    tmpl_logits = image_features.to(tmpl_emb.device) @ tmpl_emb.T
    class_prior = tmpl_logits.softmax(dim=1).mean(dim=0)
    log_prior = class_prior.log()

    cupl_ppc = {}
    for c in classnames:
        tmpl = [t.format(c) for t in ACTION_TEMPLATES] + [t.format(c) for t in IMAGENET_TEMPLATES]
        descs = descriptions.get(c, [])
        all_p = tmpl + descs
        cupl_ppc[c] = {"prompts": all_p, "weights": [1.0] * len(all_p)}
    cupl_emb = encode_text_prompts(model, tokenizer, cupl_ppc, classnames, args.device)
    cupl_logits = image_features.to(cupl_emb.device) @ cupl_emb.T
    frolic_logits = cupl_logits + 0.5 * log_prior.unsqueeze(0)

    labels_t = torch.tensor(labels, device=frolic_logits.device)
    preds = frolic_logits.argmax(dim=1)
    top1 = (preds == labels_t).float().mean().item()
    top5_preds = frolic_logits.topk(min(5, frolic_logits.shape[1]), dim=1).indices
    top5 = (top5_preds == labels_t.unsqueeze(1)).any(dim=1).float().mean().item()
    m = {"top1": top1, "top5": top5}
    print(f"  Top-1: {m['top1']*100:.2f}%  Δ: {(m['top1']-baseline_top1)*100:+.2f}%\n")
    results["frolic"] = m

    # --- NETRA sweep ---
    print(f"{'='*60}")
    print(f"  NETRA WEIGHT SWEEP — {args.dataset}")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Top-1':>8} {'Top-5':>8} {'Δ Top-1':>10}")
    print(f"{'-'*55}")

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
        ppc = build_netra_prompts(classnames, descriptions, alpha, beta)
        emb = encode_text_prompts(model, tokenizer, ppc, classnames, args.device)
        m = evaluate(image_features, labels, emb)
        delta = m["top1"] - baseline_top1
        print(f"  {label:<23} {m['top1']*100:>7.2f}% {m['top5']*100:>7.2f}% {delta*100:>+9.2f}%")
        sweep.append({"alpha": alpha, "beta": beta, "label": label, **m, "delta": delta})

    best = max(sweep, key=lambda x: x["top1"])
    default = [s for s in sweep if s["alpha"] == 0.55][0]

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {args.dataset}")
    print(f"{'='*60}")
    print(f"  {'Method':<25} {'Top-1':>8} {'Δ':>8}")
    print(f"  {'-'*43}")
    print(f"  {'90 templates':<25} {baseline_top1*100:>7.2f}%  {'---':>7}")
    for key, name in [("waffle_clip","WaffleCLIP"), ("cupl_desc_only","CuPL (desc)"),
                       ("dclip","DCLIP"), ("clip_enhance","CLIP-Enhance"),
                       ("cupl_ensemble","CuPL+e"), ("frolic","Frolic")]:
        if key in results:
            d = (results[key]['top1'] - baseline_top1) * 100
            print(f"  {name:<25} {results[key]['top1']*100:>7.2f}% {d:>+7.2f}%")
    print(f"  {'NETRA (55/45)':<25} {default['top1']*100:>7.2f}% {default['delta']*100:>+7.2f}%")
    print(f"  {'NETRA (best)':<25} {best['top1']*100:>7.2f}% {best['delta']*100:>+7.2f}%  @ {best['label']}")
    print(f"{'='*60}\n")

    # Save
    save = {
        "dataset": args.dataset, "clip_model": args.clip_model, "llm": args.llm,
        "n_images": n_images, "n_classes": n_classes,
        "results": results, "netra_sweep": sweep,
        "netra_best": best, "netra_default": default,
    }
    out = Path(args.output_dir) / f"video_{args.dataset}_{args.clip_model.replace('/','_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
