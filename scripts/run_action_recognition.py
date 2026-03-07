#!/usr/bin/env python3
"""Zero-shot action recognition on UCF-101 using CLIP.

Extracts middle frame from each video, classifies with CLIP.
Compares: templates-only vs our method (templates + LLM descriptions).

Setup:
    1. Download UCF-101 from https://www.crcv.ucf.edu/data/UCF101.php
       or: pip install huggingface_hub && python -c "
       from huggingface_hub import snapshot_download
       snapshot_download('sayakpaul/ucf101-subset', repo_type='dataset', local_dir='./data/ucf101')
       "
    2. Download train/test splits from https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

Usage:
    python scripts/run_action_recognition.py --data-dir ./data/UCF-101 --split-dir ./data/ucfTrainTestlist
    
    # Or with HuggingFace subset:
    python scripts/run_action_recognition.py --use-hf-subset
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# UCF-101 class names (CamelCase -> readable)
UCF101_CLASSES = [
    "Apply Eye Makeup", "Apply Lipstick", "Archery", "Baby Crawling",
    "Balance Beam", "Band Marching", "Baseball Pitch", "Basketball",
    "Basketball Dunk", "Bench Press", "Biking", "Billiards",
    "Blow Dry Hair", "Blowing Candles", "Body Weight Squats", "Bowling",
    "Boxing Punching Bag", "Boxing Speed Bag", "Breast Stroke", "Brushing Teeth",
    "Clean And Jerk", "Cliff Diving", "Cricket Bowling", "Cricket Shot",
    "Cutting In Kitchen", "Diving", "Drumming", "Fencing",
    "Field Hockey Penalty", "Floor Gymnastics", "Frisbee Catch", "Front Crawl",
    "Golf Swing", "Haircut", "Hammer Throw", "Hammering",
    "Handstand Pushups", "Handstand Walking", "Head Massage", "High Jump",
    "Horse Race", "Horse Riding", "Hula Hoop", "Ice Dancing",
    "Javelin Throw", "Juggling Balls", "Jump Rope", "Jumping Jack",
    "Kayaking", "Knitting", "Long Jump", "Lunges",
    "Military Parade", "Mixing", "Mopping Floor", "Nunchucks",
    "Parallel Bars", "Pizza Tossing", "Playing Cello", "Playing Daf",
    "Playing Dhol", "Playing Flute", "Playing Guitar", "Playing Piano",
    "Playing Sitar", "Playing Tabla", "Playing Violin", "Pole Vault",
    "Pommel Horse", "Pull Ups", "Punch", "Push Ups",
    "Rafting", "Rock Climbing Indoor", "Rope Climbing", "Rowing",
    "Salsa Spin", "Shaving Beard", "Shotput", "Skate Boarding",
    "Skiing", "Skijet", "Sky Diving", "Soccer Juggling",
    "Soccer Penalty", "Still Rings", "Sumo Wrestling", "Surfing",
    "Swing", "Table Tennis Shot", "Tai Chi", "Tennis Swing",
    "Throw Discus", "Trampoline Jumping", "Typing", "Uneven Bars",
    "Volleyball Spiking", "Walking With Dog", "Wall Pushups", "Writing On Board",
    "Yo Yo",
]

# Map folder names to class indices
UCF101_FOLDER_TO_IDX = {}
UCF101_FOLDERS = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling",
    "BalanceBeam", "BandMarching", "BaseballPitch", "Basketball",
    "BasketballDunk", "BenchPress", "Biking", "Billiards",
    "BlowDryHair", "BlowingCandles", "BodyWeightSquats", "Bowling",
    "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot",
    "CuttingInKitchen", "Diving", "Drumming", "Fencing",
    "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch", "FrontCrawl",
    "GolfSwing", "Haircut", "HammerThrow", "Hammering",
    "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
    "HorseRace", "HorseRiding", "HulaHoop", "IceDancing",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack",
    "Kayaking", "Knitting", "LongJump", "Lunges",
    "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks",
    "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
    "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano",
    "PlayingSitar", "PlayingTabla", "PlayingViolin", "PoleVault",
    "PommelHorse", "PullUps", "Punch", "PushUps",
    "Rafting", "RockClimbingIndoor", "RopeClimbing", "Rowing",
    "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoardin",
    "Skiing", "Skijet", "SkyDiving", "SoccerJuggling",
    "SoccerPenalty", "StillRings", "SumoWrestling", "Surfing",
    "Swing", "TableTennisShot", "TaiChi", "TennisSwing",
    "ThrowDiscus", "TrampolineJumping", "Typing", "UnevenBars",
    "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
    "YoYo",
]
for i, folder in enumerate(UCF101_FOLDERS):
    UCF101_FOLDER_TO_IDX[folder] = i


# Action-specific templates (from prior work + custom)
ACTION_TEMPLATES = [
    "a photo of a person {}.",
    "a photo of someone {}.",
    "a video frame showing a person {}.",
    "a still image of a person {}.",
    "an image of someone performing {}.",
    "a photo showing the action of {}.",
    "a person is {} in this photo.",
    "someone is {} in this image.",
    "a video screenshot of a person {}.",
    "an action shot of someone {}.",
]


def extract_middle_frame(video_path):
    """Extract the middle frame from a video file."""
    import cv2
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def load_ucf101_frames(data_dir, split_file=None, max_per_class=None):
    """Load middle frames from UCF-101 videos.
    
    Args:
        data_dir: Path to UCF-101 directory containing class folders
        split_file: Path to test split file (e.g., testlist01.txt)
        max_per_class: Max videos per class (for quick testing)
    
    Returns:
        images: list of numpy arrays (H, W, 3)
        labels: numpy array of class indices
        class_names: list of readable class names
    """
    data_dir = Path(data_dir)
    
    # If split file provided, use it
    if split_file and Path(split_file).exists():
        test_videos = []
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Format: "ClassName/v_ClassName_g01_c01.avi" or with label
                parts = line.split()
                video_path = parts[0]
                test_videos.append(video_path)
        
        print(f"  Loading {len(test_videos)} test videos from split file...")
        images = []
        labels = []
        skipped = 0
        
        for vpath in test_videos:
            full_path = data_dir / vpath
            if not full_path.exists():
                skipped += 1
                continue
            
            # Extract class from path
            class_folder = vpath.split("/")[0]
            if class_folder not in UCF101_FOLDER_TO_IDX:
                skipped += 1
                continue
            
            frame = extract_middle_frame(full_path)
            if frame is None:
                skipped += 1
                continue
            
            images.append(frame)
            labels.append(UCF101_FOLDER_TO_IDX[class_folder])
        
        if skipped > 0:
            print(f"  Skipped {skipped} videos (missing/unreadable)")
    else:
        # Load from directory structure
        print(f"  Loading from directory: {data_dir}")
        images = []
        labels = []
        
        for class_folder in sorted(data_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            folder_name = class_folder.name
            if folder_name not in UCF101_FOLDER_TO_IDX:
                continue
            
            class_idx = UCF101_FOLDER_TO_IDX[folder_name]
            video_files = sorted(class_folder.glob("*.avi")) + sorted(class_folder.glob("*.mp4"))
            
            if max_per_class:
                video_files = video_files[:max_per_class]
            
            for vf in video_files:
                frame = extract_middle_frame(vf)
                if frame is not None:
                    images.append(frame)
                    labels.append(class_idx)
        
    # Convert to arrays
    try:
        images_arr = np.stack(images)
    except ValueError:
        images_arr = np.empty(len(images), dtype=object)
        for i, img in enumerate(images):
            images_arr[i] = img
    
    labels_arr = np.array(labels)
    n_classes = len(np.unique(labels_arr))
    print(f"  Loaded {len(images_arr)} frames, {n_classes} classes")
    
    return images_arr, labels_arr, UCF101_CLASSES


def load_ucf101_from_hf():
    """Load UCF-101 subset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        sys.exit(1)
    
    print("  Loading UCF-101 from HuggingFace...")
    ds = load_dataset("sayakpaul/ucf101-subset", split="test")
    
    images = []
    labels = []
    
    for item in ds:
        img = np.array(item["image"].convert("RGB"))
        images.append(img)
        labels.append(item["label"])
    
    try:
        images_arr = np.stack(images)
    except ValueError:
        images_arr = np.empty(len(images), dtype=object)
        for i, img in enumerate(images):
            images_arr[i] = img
    
    labels_arr = np.array(labels)
    class_names = ds.features["label"].names
    # Convert CamelCase to readable
    readable_names = []
    for name in class_names:
        if name in UCF101_FOLDER_TO_IDX:
            readable_names.append(UCF101_CLASSES[UCF101_FOLDER_TO_IDX[name]])
        else:
            readable_names.append(name)
    
    print(f"  Loaded {len(images_arr)} frames, {len(np.unique(labels_arr))} classes")
    return images_arr, labels_arr, readable_names


def generate_action_descriptions(class_names, llm_model, llm_provider, batch_size=10):
    """Generate visual descriptions for action classes."""
    from visprompt.utils.llm import LLMClient
    
    client = LLMClient(model=llm_model, provider=llm_provider, temperature=0.7)
    
    all_descriptions = {}
    total_cost = 0.0
    
    for i in range(0, len(class_names), batch_size):
        batch = class_names[i:i + batch_size]
        batch_str = ", ".join(batch)
        
        prompt = f"""For each action below, generate 8-10 short visual descriptions that would help identify the action in a still photograph or video frame. Focus on:
- Body posture and position
- Key objects or equipment involved
- Setting/environment clues
- Motion blur or movement indicators visible in a still frame

Actions: {batch_str}

Return JSON: {{"action_name": ["description1", "description2", ...]}}
Each description should be a complete sentence starting with "A photo of" or "A person" or similar.
Return ONLY valid JSON, no markdown."""

        try:
            response, usage = client.call(prompt, json_mode=True)
            import json as json_mod
            descs = json_mod.loads(response)
            
            for cls_name in batch:
                # Try exact match and variations
                found = None
                for key in descs:
                    if key.lower().strip() == cls_name.lower().strip():
                        found = descs[key]
                        break
                if found is None:
                    # Fuzzy match
                    for key in descs:
                        if cls_name.lower().replace(" ", "") in key.lower().replace(" ", ""):
                            found = descs[key]
                            break
                
                if found and isinstance(found, list):
                    all_descriptions[cls_name] = found
                else:
                    all_descriptions[cls_name] = []
            
            cost = usage.total_cost if hasattr(usage, 'total_cost') else 0
            total_cost += cost
            
        except Exception as e:
            logger.warning(f"Description batch failed: {e}")
            for cls_name in batch:
                all_descriptions[cls_name] = []
    
    n_with_desc = sum(1 for v in all_descriptions.values() if len(v) > 0)
    print(f"  Generated descriptions for {n_with_desc}/{len(class_names)} actions (${total_cost:.4f})")
    
    return all_descriptions, total_cost


def build_action_prompts_weighted(class_names, descriptions, base_weight, desc_weight):
    """Build weighted prompts for action recognition."""
    from visprompt.baselines import IMAGENET_TEMPLATES
    
    prompts_per_class = {}
    
    for cls_name in class_names:
        # Use action-specific templates
        base_prompts = [t.format(cls_name.lower()) for t in ACTION_TEMPLATES]
        # Also add standard ImageNet templates
        base_prompts += [t.format(cls_name.lower()) for t in IMAGENET_TEMPLATES]
        
        desc_prompts = descriptions.get(cls_name, [])
        
        n_base = len(base_prompts)
        n_desc = len(desc_prompts)
        
        if n_desc > 0 and desc_weight > 0:
            per_base = base_weight / n_base
            per_desc = desc_weight / n_desc
            all_prompts = base_prompts + desc_prompts
            all_weights = [per_base] * n_base + [per_desc] * n_desc
        else:
            all_prompts = base_prompts
            all_weights = [1.0 / n_base] * n_base
        
        prompts_per_class[cls_name] = {
            "prompts": all_prompts,
            "weights": all_weights,
        }
    
    return {
        "type": "classification",
        "prompts_per_class": prompts_per_class,
        "ensemble_method": "weighted_average",
    }


def zero_shot_classify_weighted(model, tokenizer, test_features, test_labels,
                                 prompts_dict, device, class_names):
    """Zero-shot classification with weighted prompts."""
    import torch
    
    ppc = prompts_dict["prompts_per_class"]
    
    all_text_features = []
    for cls_name in class_names:
        cls_data = ppc[cls_name]
        texts = cls_data["prompts"]
        weights = cls_data["weights"]
        
        feats = []
        for text in texts:
            tokens = tokenizer(text).to(device)
            with torch.no_grad():
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            feats.append(feat.squeeze(0))
        
        feats = torch.stack(feats)
        w = torch.tensor(weights, dtype=feats.dtype, device=device)
        class_feat = (w.unsqueeze(-1) * feats).sum(dim=0)
        class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
        all_text_features.append(class_feat)
    
    text_features = torch.stack(all_text_features).to(device)
    test_feat = test_features.to(device)
    
    if hasattr(model, 'logit_scale'):
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = torch.tensor(100.0).to(device)
    
    with torch.no_grad():
        logits = logit_scale * test_feat @ text_features.T
        preds = logits.argmax(dim=-1).cpu().numpy()
    
    return float(np.mean(preds == test_labels))


def main():
    parser = argparse.ArgumentParser(description="Zero-shot action recognition on UCF-101")
    parser.add_argument("--data-dir", type=str, default="./data/UCF-101")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Path to testlist01.txt")
    parser.add_argument("--use-hf-subset", action="store_true",
                        help="Use HuggingFace UCF-101 subset instead of local videos")
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/action_recognition")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    
    import torch
    import open_clip
    from PIL import Image
    
    # Load data
    print(f"\n{'='*65}")
    print(f"  UCF-101 ZERO-SHOT ACTION RECOGNITION")
    print(f"  Model: {args.clip_model}")
    print(f"{'='*65}")
    
    if args.use_hf_subset:
        images, labels, class_names = load_ucf101_from_hf()
    else:
        images, labels, class_names = load_ucf101_frames(
            args.data_dir, args.split_file, args.max_per_class
        )
    
    n_classes = len(np.unique(labels))
    
    # Load CLIP
    print(f"\nLoading CLIP: {args.clip_model}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained="openai", device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()
    
    # Encode images
    print(f"Encoding {len(images)} frames...")
    all_features = []
    batch_size = 128
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = []
        for img_arr in images[start:end]:
            img = Image.fromarray(img_arr)
            batch.append(preprocess(img))
        
        image_input = torch.stack(batch).to(args.device)
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
    
    test_features = torch.cat(all_features, dim=0)
    print(f"  Features: {test_features.shape}")
    
    # ── 1. Templates-only baseline ────────────────────────────────
    print(f"\n--- Templates-only ({len(ACTION_TEMPLATES)} action + 80 ImageNet) ---")
    from visprompt.baselines import IMAGENET_TEMPLATES
    
    template_prompts = build_action_prompts_weighted(
        class_names, {}, 1.0, 0.0
    )
    templates_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        template_prompts, args.device, class_names
    )
    print(f"  Accuracy: {templates_acc:.4f}")
    
    # ── 2. LLM descriptions ──────────────────────────────────────
    print(f"\n--- Generating action descriptions ({args.llm}) ---")
    descriptions, desc_cost = generate_action_descriptions(
        class_names, args.llm, args.llm_provider
    )
    
    # ── 3. Weight ablation ────────────────────────────────────────
    WEIGHT_CONFIGS = [
        (1.0, 0.0, "100/0"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100"),
    ]
    
    print(f"\n--- Weight ablation ---")
    print(f"{'Config':<12} {'Accuracy':>10} {'Δ':>10}")
    print(f"{'-'*35}")
    
    results = {
        "dataset": "ucf101",
        "clip_model": args.clip_model,
        "n_classes": n_classes,
        "n_samples": len(labels),
        "llm": args.llm,
        "templates_accuracy": float(templates_acc),
        "description_cost": float(desc_cost),
        "weights": [],
    }
    
    best_acc = 0
    best_config = ""
    
    for base_w, desc_w, label in WEIGHT_CONFIGS:
        prompts = build_action_prompts_weighted(
            class_names, descriptions, base_w, desc_w
        )
        acc = zero_shot_classify_weighted(
            model, tokenizer, test_features, labels,
            prompts, args.device, class_names
        )
        delta = acc - templates_acc
        
        marker = " ←" if acc == max(best_acc, acc) and desc_w > 0 else ""
        print(f"  {label:<10} {acc:>10.4f} {delta:>+9.4f}{marker}")
        
        results["weights"].append({
            "config": label,
            "base_weight": base_w,
            "desc_weight": desc_w,
            "accuracy": float(acc),
            "delta": float(delta),
        })
        
        if acc > best_acc:
            best_acc = acc
            best_config = label
    
    results["best_accuracy"] = float(best_acc)
    results["best_config"] = best_config
    results["best_delta"] = float(best_acc - templates_acc)
    
    print(f"\n  Best: {best_config} → {best_acc:.4f} (Δ {best_acc - templates_acc:+.4f})")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ucf101_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
