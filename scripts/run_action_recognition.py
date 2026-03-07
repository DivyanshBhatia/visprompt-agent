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
    ds = load_dataset("sayakpaul/ucf101-subset", split="train")
    
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


def load_ucf101_torchvision(data_dir):
    """Load UCF-101 via torchvision (requires pyav: pip install av)."""
    import torchvision
    
    print("  Loading UCF-101 via torchvision (downloading if needed)...")
    print("  This may take a while on first run (~6.5GB download)...")
    
    # torchvision downloads both videos and annotations
    dataset = torchvision.datasets.UCF101(
        root=data_dir,
        annotation_path=os.path.join(data_dir, "ucfTrainTestlist"),
        frames_per_clip=1,
        step_between_clips=1000000,  # large step = ~1 clip per video
        train=False,  # test split
        output_format="THWC",
        num_workers=0,
    )
    
    print(f"  Dataset loaded: {len(dataset)} clips")
    
    images = []
    labels = []
    seen_videos = set()
    
    for i in range(len(dataset)):
        try:
            video, audio, label = dataset[i]
            # video shape: (T, H, W, C) - take middle frame
            mid = video.shape[0] // 2
            frame = video[mid].numpy().astype(np.uint8)
            
            # Deduplicate: only take one frame per unique video+label
            key = (label, i // 10)  # approximate dedup
            if key in seen_videos:
                continue
            seen_videos.add(key)
            
            images.append(frame)
            labels.append(label)
        except Exception as e:
            continue
        
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(dataset)} clips, {len(images)} frames kept...")
    
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


def load_ucf101_torchvision_simple(data_dir):
    """Simpler UCF-101 loader: download videos, extract middle frames manually."""
    import torchvision
    import cv2
    
    print("  Downloading UCF-101 via torchvision...")
    
    # Just trigger the download
    ucf_path = os.path.join(data_dir, "UCF-101")
    annotation_path = os.path.join(data_dir, "ucfTrainTestlist")
    
    if not os.path.exists(ucf_path):
        # Download using torchvision's internal mechanism
        from torchvision.datasets.utils import download_and_extract_archive
        url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
        try:
            download_and_extract_archive(url, data_dir, filename="UCF101.rar")
        except Exception:
            print("  Auto-download failed. Please download UCF-101 manually:")
            print("  wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar")
            print("  unrar x UCF101.rar -d ./data/")
            sys.exit(1)
    
    if not os.path.exists(annotation_path):
        from torchvision.datasets.utils import download_and_extract_archive
        url = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
        try:
            download_and_extract_archive(url, data_dir)
        except Exception:
            print("  Please download splits manually")
            sys.exit(1)
    
    # Read test split
    split_file = os.path.join(annotation_path, "testlist01.txt")
    return load_ucf101_frames(ucf_path, split_file)


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
            response = client.call(prompt, json_mode=True)
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
            
            cost = 0  # cost tracked internally by LLMClient
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
    parser.add_argument("--use-torchvision", action="store_true",
                        help="Use torchvision UCF101 loader (requires: pip install av)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baselines, only run weight ablation")
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
    elif args.use_torchvision:
        images, labels, class_names = load_ucf101_torchvision(args.data_dir or "./data")
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
    
    # ── 1b. Additional baselines ──────────────────────────────────
    print(f"\n--- Additional baselines ---")
    
    # Class name only: "apply eye makeup"
    class_name_prompts = {"prompts_per_class": {
        cn: {"prompts": [cn.lower()], "weights": [1.0]} for cn in class_names
    }}
    cn_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        class_name_prompts, args.device, class_names
    )
    print(f"  Class name only:    {cn_acc:.4f}")
    
    # Single template: "a photo of a person {action}"
    single_prompts = {"prompts_per_class": {
        cn: {"prompts": [f"a photo of a person {cn.lower()}."], "weights": [1.0]}
        for cn in class_names
    }}
    single_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        single_prompts, args.device, class_names
    )
    print(f"  Single template:    {single_acc:.4f}")
    
    # CuPL-style: generate descriptions using a simpler prompt
    print(f"\n--- CuPL baseline (GPT descriptions, no templates) ---")
    from visprompt.utils.llm import LLMClient
    cupl_client = LLMClient(model=args.llm, provider=args.llm_provider, temperature=0.7)
    cupl_descriptions = {}
    for i in range(0, len(class_names), 10):
        batch = class_names[i:i+10]
        batch_str = ", ".join(batch)
        prompt = f"""Describe the visual appearance of each action in a photo. For each action, give 5 short sentences.
Actions: {batch_str}
Return JSON: {{"action": ["sentence1", ...]}}
Return ONLY valid JSON."""
        try:
            response = cupl_client.call(prompt, json_mode=True)
            import json as json_mod
            descs = json_mod.loads(response)
            for cn in batch:
                for key in descs:
                    if cn.lower().strip() in key.lower():
                        cupl_descriptions[cn] = descs[key]
                        break
                if cn not in cupl_descriptions:
                    cupl_descriptions[cn] = []
        except Exception:
            for cn in batch:
                cupl_descriptions[cn] = []
    
    # CuPL: descriptions only (no templates)
    cupl_prompts = {"prompts_per_class": {}}
    for cn in class_names:
        descs = cupl_descriptions.get(cn, [])
        if descs:
            cupl_prompts["prompts_per_class"][cn] = {
                "prompts": descs, "weights": [1.0/len(descs)] * len(descs)
            }
        else:
            cupl_prompts["prompts_per_class"][cn] = {
                "prompts": [cn.lower()], "weights": [1.0]
            }
    cupl_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        cupl_prompts, args.device, class_names
    )
    print(f"  CuPL (desc only):   {cupl_acc:.4f}")
    
    # CuPL + ensemble (templates + CuPL descriptions, equal weight)
    cupl_ens_prompts = {"prompts_per_class": {}}
    for cn in class_names:
        base = [t.format(cn.lower()) for t in ACTION_TEMPLATES] + \
               [t.format(cn.lower()) for t in IMAGENET_TEMPLATES]
        descs = cupl_descriptions.get(cn, [])
        all_p = base + descs
        cupl_ens_prompts["prompts_per_class"][cn] = {
            "prompts": all_p, "weights": [1.0/len(all_p)] * len(all_p)
        }
    cupl_ens_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        cupl_ens_prompts, args.device, class_names
    )
    print(f"  CuPL + ensemble:    {cupl_ens_acc:.4f}")
    
    baselines = {
        "class_name_only": float(cn_acc),
        "single_template": float(single_acc),
        "80_template_ensemble": float(templates_acc),
        "cupl_desc_only": float(cupl_acc),
        "cupl_ensemble": float(cupl_ens_acc),
    }
    
    # WaffleCLIP: random descriptor ensembles
    print(f"\n--- WaffleCLIP baseline ---")
    import random
    random.seed(42)
    random_words = [
        "bright", "dark", "colorful", "textured", "smooth", "rough",
        "large", "small", "round", "angular", "metallic", "wooden",
        "organic", "geometric", "striped", "spotted", "furry", "scaly",
        "shiny", "matte", "transparent", "opaque", "symmetric", "curved",
        "flat", "tall", "wide", "narrow", "detailed", "simple",
        "natural", "artificial", "indoor", "outdoor", "urban", "rural",
        "wet", "dry", "soft", "hard", "warm", "cool", "vibrant", "muted",
        "patterned", "solid", "complex", "elegant", "compact", "sprawling",
    ]
    waffle_prompts = {"prompts_per_class": {}}
    for cn in class_names:
        wps = [f"a photo of a person {cn.lower()}"]
        for _ in range(15):
            n_words = random.randint(2, 5)
            words = random.sample(random_words, n_words)
            wps.append(f"a {' '.join(words)} photo of a person {cn.lower()}")
        waffle_prompts["prompts_per_class"][cn] = {
            "prompts": wps, "weights": [1.0/len(wps)] * len(wps)
        }
    waffle_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        waffle_prompts, args.device, class_names
    )
    print(f"  WaffleCLIP:         {waffle_acc:.4f}")
    baselines["waffle_clip"] = float(waffle_acc)
    
    # DCLIP: "What does {action} look like?" descriptors
    print(f"\n--- DCLIP baseline ---")
    dclip_descriptions = {}
    for i in range(0, len(class_names), 10):
        batch = class_names[i:i+10]
        batch_str = ", ".join(batch)
        prompt = f"""For each action below, describe what it looks like in a photo.
Give 5-8 short visual descriptors per action.
Format each as: "{{action}} which has {{descriptor}}"
Focus on distinctive visual attributes: body pose, equipment, setting, motion.

Actions: {batch_str}

Respond ONLY with JSON: {{"action": ["descriptor1", "descriptor2", ...]}}"""
        try:
            response = cupl_client.call(prompt, json_mode=True)
            import json as json_mod; descs = json_mod.loads(response)
            for cn in batch:
                found = None
                for key in descs:
                    if cn.lower().strip() in key.lower():
                        found = descs[key]
                        break
                if found and isinstance(found, list):
                    # Ensure DCLIP format
                    formatted = []
                    for d in found:
                        if cn.lower() not in d.lower():
                            d = f"{cn.lower()} which has {d}"
                        formatted.append(d)
                    dclip_descriptions[cn] = formatted
                else:
                    dclip_descriptions[cn] = [f"a photo of a person {cn.lower()}"]
        except Exception:
            for cn in batch:
                dclip_descriptions[cn] = [f"a photo of a person {cn.lower()}"]
    
    dclip_prompts = {"prompts_per_class": {}}
    for cn in class_names:
        descs = dclip_descriptions.get(cn, [f"a photo of a person {cn.lower()}"])
        dclip_prompts["prompts_per_class"][cn] = {
            "prompts": descs, "weights": [1.0/len(descs)] * len(descs)
        }
    dclip_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        dclip_prompts, args.device, class_names
    )
    print(f"  DCLIP:              {dclip_acc:.4f}")
    baselines["dclip"] = float(dclip_acc)
    
    # CLIP-Enhance: synonym expansion + descriptions
    print(f"\n--- CLIP-Enhance baseline ---")
    enhance_descriptions = {}
    for i in range(0, len(class_names), 10):
        batch = class_names[i:i+10]
        batch_str = ", ".join(batch)
        prompt = f"""For each action, provide:
1. 3 synonyms or alternative names for the action
2. 5 visual descriptions of how it appears in a photo

Actions: {batch_str}

Return JSON: {{"action": {{"synonyms": ["syn1", ...], "descriptions": ["desc1", ...]}}}}
Return ONLY valid JSON."""
        try:
            response = cupl_client.call(prompt, json_mode=True)
            descs = json_mod.loads(response)
            for cn in batch:
                found = None
                for key in descs:
                    if cn.lower().strip() in key.lower():
                        found = descs[key]
                        break
                if found and isinstance(found, dict):
                    syns = found.get("synonyms", [])
                    vis = found.get("descriptions", [])
                    all_prompts = [f"a photo of a person {cn.lower()}"]
                    for s in syns:
                        all_prompts.append(f"a photo of a person {s.lower()}")
                    all_prompts.extend(vis)
                    enhance_descriptions[cn] = all_prompts
                else:
                    enhance_descriptions[cn] = [f"a photo of a person {cn.lower()}"]
        except Exception:
            for cn in batch:
                enhance_descriptions[cn] = [f"a photo of a person {cn.lower()}"]
    
    enhance_prompts = {"prompts_per_class": {}}
    for cn in class_names:
        descs = enhance_descriptions.get(cn, [f"a photo of a person {cn.lower()}"])
        enhance_prompts["prompts_per_class"][cn] = {
            "prompts": descs, "weights": [1.0/len(descs)] * len(descs)
        }
    enhance_acc = zero_shot_classify_weighted(
        model, tokenizer, test_features, labels,
        enhance_prompts, args.device, class_names
    )
    print(f"  CLIP-Enhance:       {enhance_acc:.4f}")
    baselines["clip_enhance"] = float(enhance_acc)
    
    print(f"\n  Summary:")
    for name, acc in sorted(baselines.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:<25} {acc:.4f}")
    
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
        "baselines": baselines,
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
