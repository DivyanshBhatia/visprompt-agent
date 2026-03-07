#!/usr/bin/env python3
"""Few-shot comparison: our zero-shot prompt method vs k-shot linear probe.

Compares:
  1. Zero-shot CLIP (80 templates)
  2. Zero-shot Ours (templates + LLM descriptions, best weight)
  3. k-shot linear probe (k=1,2,4,8,16) on CLIP features

This shows our zero-shot method can match or beat k-shot approaches
that require labeled training data.

Example:
    python scripts/run_fewshot_comparison.py \
        --datasets flowers102 dtd oxford_pets \
        --clip-model ViT-L/14 \
        --n-trials 5
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

# Datasets with train/test splits for few-shot
DATASET_CONFIGS = {
    "cifar100": {
        "train_loader": "cifar100_train",
        "test_loader": "cifar100_test",
        "val_size": 10000,
        "best_weight": (0.7, 0.3),  # from weight ablation
    },
    "flowers102": {
        "train_loader": "flowers102_train",
        "test_loader": "flowers102_test",
        "val_size": 6149,
        "best_weight": (0.0, 1.0),
    },
    "dtd": {
        "train_loader": "dtd_train",
        "test_loader": "dtd_test",
        "val_size": 1880,
        "best_weight": (0.4, 0.6),
    },
    "oxford_pets": {
        "train_loader": "pets_train",
        "test_loader": "pets_test",
        "val_size": 3669,
        "best_weight": (0.0, 1.0),
    },
    "food101": {
        "train_loader": "food101_train",
        "test_loader": "food101_test",
        "val_size": 10000,
        "best_weight": (0.4, 0.6),
    },
}

K_SHOTS = [1, 2, 4, 8, 16]


def load_train_test(dataset_name, args):
    """Load train and test splits for a dataset."""
    import torchvision

    data_dir = args.data_dir or "./data"
    val_size = DATASET_CONFIGS[dataset_name]["val_size"]

    if dataset_name == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True)
        test_ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True)

        # Train: use all
        train_images = np.array(train_ds.data)
        train_labels = np.array(train_ds.targets)

        # Test: subsample
        indices = np.random.RandomState(42).permutation(len(test_ds))[:val_size]
        test_images = np.array(test_ds.data)[indices]
        test_labels = np.array(test_ds.targets)[indices]

        return train_images, train_labels, test_images, test_labels

    elif dataset_name == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
        test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
        train_images = np.array(train_ds.data)
        train_labels = np.array(train_ds.targets)
        indices = np.random.RandomState(42).permutation(len(test_ds))[:val_size]
        test_images = np.array(test_ds.data)[indices]
        test_labels = np.array(test_ds.targets)[indices]
        return train_images, train_labels, test_images, test_labels

    # Datasets with PIL images
    LOADERS = {
        "flowers102": ("Flowers102", {"split": "train"}, {"split": "test"}),
        "dtd": ("DTD", {"split": "train"}, {"split": "test"}),
        "oxford_pets": ("OxfordIIITPet", {"split": "trainval"}, {"split": "test"}),
        "food101": ("Food101", {"split": "train"}, {"split": "test"}),
    }

    loader_name, train_kwargs, test_kwargs = LOADERS[dataset_name]
    dataset_cls = getattr(torchvision.datasets, loader_name)

    # Load train
    train_ds = dataset_cls(root=data_dir, download=True, **train_kwargs)
    train_images, train_labels = _pil_dataset_to_arrays(train_ds, max_n=None)

    # Load test
    test_ds = dataset_cls(root=data_dir, download=True, **test_kwargs)
    test_images, test_labels = _pil_dataset_to_arrays(test_ds, max_n=val_size)

    return train_images, train_labels, test_images, test_labels


def _pil_dataset_to_arrays(dataset, max_n=None):
    """Convert PIL dataset to numpy arrays."""
    n_total = len(dataset)
    if max_n and max_n < n_total:
        indices = np.random.RandomState(42).permutation(n_total)[:max_n]
    else:
        indices = range(n_total)

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(np.array(img.convert("RGB")))
        labels.append(label)

    try:
        images_arr = np.stack(images)
    except ValueError:
        images_arr = np.empty(len(images), dtype=object)
        for i, img in enumerate(images):
            images_arr[i] = img

    return images_arr, np.array(labels)


def encode_images(model, preprocess, images, device, batch_size=128):
    """Encode images to CLIP features."""
    import torch
    from PIL import Image

    all_features = []
    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        batch = []
        for img_arr in images[start:end]:
            img = Image.fromarray(img_arr) if isinstance(img_arr, np.ndarray) else img_arr
            batch.append(preprocess(img))

        image_input = torch.stack(batch).to(device)
        with torch.no_grad():
            features = model.encode_image(image_input)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def sample_k_shot(train_features, train_labels, k, seed=42):
    """Sample k examples per class."""
    rng = np.random.RandomState(seed)
    classes = np.unique(train_labels)
    selected_features = []
    selected_labels = []

    for c in classes:
        idx = np.where(train_labels == c)[0]
        if len(idx) < k:
            chosen = idx  # use all if fewer than k
        else:
            chosen = rng.choice(idx, k, replace=False)
        selected_features.append(train_features[chosen])
        selected_labels.extend([c] * len(chosen))

    import torch
    return torch.cat(selected_features, dim=0), np.array(selected_labels)


def linear_probe(train_features, train_labels, test_features, test_labels, C=0.316):
    """Train logistic regression on CLIP features (standard CLIP protocol)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        random_state=42,
        C=C,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
    )

    train_feat_np = train_features.numpy() if hasattr(train_features, 'numpy') else train_features
    test_feat_np = test_features.numpy() if hasattr(test_features, 'numpy') else test_features

    clf.fit(train_feat_np, train_labels)
    preds = clf.predict(test_feat_np)
    acc = np.mean(preds == test_labels)
    return acc


def zero_shot_classify(model, tokenizer, test_features, test_labels, prompts, device, logit_scale=None):
    """Zero-shot classification using text prompts.
    
    Args:
        prompts: Either list of list of strings (uniform weight),
                 or dict with prompts_per_class (weighted).
    """
    import torch

    # Handle weighted prompt dict format
    if isinstance(prompts, dict) and "prompts_per_class" in prompts:
        return _zero_shot_weighted(model, tokenizer, test_features, test_labels,
                                   prompts, device, logit_scale)

    # Simple format: list of list of strings (uniform averaging)
    all_text_features = []
    for class_prompts in prompts:
        class_feats = []
        for prompt_text in class_prompts:
            tokens = tokenizer(prompt_text).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            class_feats.append(text_feat)
        # Average across prompts for this class
        class_feat = torch.stack(class_feats).mean(dim=0)
        class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
        all_text_features.append(class_feat.squeeze(0))

    text_features = torch.stack(all_text_features).to(device)
    test_feat = test_features.to(device)

    if logit_scale is None:
        if hasattr(model, 'logit_scale'):
            logit_scale = model.logit_scale.exp()
        else:
            logit_scale = torch.tensor(100.0).to(device)

    with torch.no_grad():
        logits = logit_scale * test_feat @ text_features.T
        preds = logits.argmax(dim=-1).cpu().numpy()

    return float(np.mean(preds == test_labels))


def _zero_shot_weighted(model, tokenizer, test_features, test_labels, prompts_dict, device, logit_scale=None):
    """Zero-shot with per-prompt weights (group-normalized)."""
    import torch

    ppc = prompts_dict["prompts_per_class"]
    class_names = list(ppc.keys())
    
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
        
        feats = torch.stack(feats)  # (P, D)
        w = torch.tensor(weights, dtype=feats.dtype, device=device)
        class_feat = (w.unsqueeze(-1) * feats).sum(dim=0)
        class_feat = class_feat / class_feat.norm(dim=-1, keepdim=True)
        all_text_features.append(class_feat)

    text_features = torch.stack(all_text_features).to(device)
    test_feat = test_features.to(device)

    if logit_scale is None:
        if hasattr(model, 'logit_scale'):
            logit_scale = model.logit_scale.exp()
        else:
            logit_scale = torch.tensor(100.0).to(device)

    with torch.no_grad():
        logits = logit_scale * test_feat @ text_features.T
        preds = logits.argmax(dim=-1).cpu().numpy()

    return float(np.mean(preds == test_labels))


def main():
    parser = argparse.ArgumentParser(description="Few-shot vs zero-shot comparison")
    parser.add_argument("--datasets", nargs="+",
                        default=["flowers102", "dtd", "oxford_pets"],
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of random seeds for k-shot sampling")
    parser.add_argument("--k-shots", nargs="+", type=int, default=K_SHOTS)
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--output-dir", type=str, default="experiments/fewshot")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args
    parser.add_argument("--config", type=str)
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    import torch
    import open_clip

    # Load CLIP model once
    print(f"\nLoading CLIP model: {args.clip_model}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained="openai", device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    model.eval()

    # Import templates and description generation
    from visprompt.baselines import IMAGENET_TEMPLATES
    from scripts.run import build_task_spec
    from scripts.run_weight_ablation import generate_descriptions, build_prompts_with_weights

    all_results = {}

    for dataset_name in args.datasets:
        print(f"\n{'#'*70}")
        print(f"#  FEW-SHOT COMPARISON: {dataset_name.upper()}")
        print(f"{'#'*70}")

        cfg = DATASET_CONFIGS[dataset_name]
        base_w, desc_w = cfg["best_weight"]

        # Load train and test splits
        print(f"\nLoading {dataset_name} train/test splits...")
        train_images, train_labels, test_images, test_labels = \
            load_train_test(dataset_name, args)
        n_classes = len(np.unique(test_labels))
        print(f"  Train: {len(train_labels)} images, Test: {len(test_labels)} images, "
              f"Classes: {n_classes}")

        # Encode test images (reused for all methods)
        print(f"\nEncoding test images...")
        test_features = encode_images(model, preprocess, test_images, args.device)
        print(f"  Test features: {test_features.shape}")

        # Encode train images
        print(f"Encoding train images...")
        train_features = encode_images(model, preprocess, train_images, args.device)
        print(f"  Train features: {train_features.shape}")

        # ── 1. Zero-shot: templates only ──────────────────────────────
        print(f"\n--- Zero-shot: 80 templates ---")
        args_copy = argparse.Namespace(**vars(args))
        args_copy.dataset = dataset_name
        args_copy.val_size = cfg["val_size"]
        task_spec = build_task_spec(args_copy)

        template_prompts = []
        for class_name in task_spec.class_names:
            class_prompts = [t.format(class_name) for t in IMAGENET_TEMPLATES]
            template_prompts.append(class_prompts)

        zs_templates_acc = zero_shot_classify(
            model, tokenizer, test_features, test_labels,
            template_prompts, args.device
        )
        print(f"  Templates-only accuracy: {zs_templates_acc:.4f}")

        # ── 2. Zero-shot: Ours (templates + descriptions) ────────────
        print(f"\n--- Zero-shot: Ours ({base_w:.0%}/{desc_w:.0%}) ---")
        descriptions, desc_cost = generate_descriptions(
            task_spec, args.llm, args.llm_provider
        )

        ours_prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions, base_w, desc_w
        )

        zs_ours_acc = zero_shot_classify(
            model, tokenizer, test_features, test_labels,
            ours_prompts, args.device
        )
        print(f"  Ours accuracy: {zs_ours_acc:.4f} (Δ {zs_ours_acc - zs_templates_acc:+.4f})")

        # ── 3. k-shot linear probe ────────────────────────────────────
        print(f"\n--- k-shot linear probe ---")
        print(f"{'k':>4}  {'Mean Acc':>10}  {'± Std':>8}  {'vs Templates':>14}  {'vs Ours':>10}")
        print(f"{'-'*55}")

        fewshot_results = {}
        for k in args.k_shots:
            trial_accs = []
            for trial in range(args.n_trials):
                seed = 42 + trial
                k_features, k_labels = sample_k_shot(
                    train_features, train_labels, k, seed=seed
                )
                acc = linear_probe(
                    k_features, k_labels,
                    test_features, test_labels
                )
                trial_accs.append(acc)

            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs)
            delta_templates = mean_acc - zs_templates_acc
            delta_ours = mean_acc - zs_ours_acc

            fewshot_results[k] = {
                "mean": float(mean_acc),
                "std": float(std_acc),
                "trials": [float(a) for a in trial_accs],
            }

            marker = "←" if delta_ours < 0 else ""
            print(f"  {k:>3}  {mean_acc:>10.4f}  ±{std_acc:.4f}  "
                  f"{delta_templates:>+13.4f}  {delta_ours:>+9.4f} {marker}")

        # ── Store results ──────────────────────────────────────────────
        all_results[dataset_name] = {
            "clip_model": args.clip_model,
            "zero_shot_templates": float(zs_templates_acc),
            "zero_shot_ours": float(zs_ours_acc),
            "ours_weight": f"{base_w}/{desc_w}",
            "ours_delta": float(zs_ours_acc - zs_templates_acc),
            "fewshot": fewshot_results,
            "n_classes": n_classes,
            "n_train": len(train_labels),
            "n_test": len(test_labels),
        }

        # Find crossover point
        for k in sorted(fewshot_results.keys()):
            if fewshot_results[k]["mean"] >= zs_ours_acc:
                print(f"\n  → {k}-shot linear probe matches/beats our zero-shot method")
                all_results[dataset_name]["crossover_k"] = k
                break
        else:
            print(f"\n  → Our zero-shot method beats all tested k-shot ({max(args.k_shots)}-shot)")
            all_results[dataset_name]["crossover_k"] = f">{max(args.k_shots)}"

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fewshot_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # ── Final summary ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Zero-Shot Ours vs k-Shot Linear Probe")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'Templates':>10} {'Ours':>10} {'1-shot':>10} {'4-shot':>10} {'16-shot':>10} {'Cross-k':>8}")
    print(f"{'-'*73}")
    for ds, res in all_results.items():
        fs = res["fewshot"]
        print(f"{ds:<15} {res['zero_shot_templates']:>10.4f} {res['zero_shot_ours']:>10.4f} "
              f"{fs.get(1, {}).get('mean', 0):>10.4f} "
              f"{fs.get(4, {}).get('mean', 0):>10.4f} "
              f"{fs.get(16, {}).get('mean', 0):>10.4f} "
              f"{res['crossover_k']:>8}")


if __name__ == "__main__":
    main()
