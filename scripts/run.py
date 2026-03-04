#!/usr/bin/env python3
"""Run VisPromptAgent on any supported dataset.

Examples:
    # CIFAR-100 classification
    python scripts/run.py --task classification --dataset cifar100 \\
        --clip-model ViT-B/32 --llm gpt-4o --max-iter 3

    # DAVIS 2017 segmentation
    python scripts/run.py --task segmentation --dataset davis2017 \\
        --sam-checkpoint sam_vit_b.pth --data-dir /data/davis

    # LVIS detection
    python scripts/run.py --task detection --dataset lvis \\
        --gdino-config config.py --gdino-checkpoint gdino.pth \\
        --data-dir /data/lvis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visprompt.agents.base import TaskSpec
from visprompt.pipeline import VisPromptPipeline


def build_task_spec(args) -> TaskSpec:
    """Build TaskSpec from command-line arguments."""
    if args.dataset == "cifar100":
        return _build_cifar100_spec(args)
    elif args.dataset == "davis2017":
        return _build_davis2017_spec(args)
    elif args.dataset == "lvis":
        return _build_lvis_spec(args)
    elif args.config:
        return _build_from_config(args.config)
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            "Use --config for custom datasets."
        )


def build_task_runner(args, task_spec: TaskSpec):
    """Build the appropriate TaskRunner."""
    device = args.device

    if task_spec.task_type == "classification":
        from visprompt.tasks.classification import CLIPClassificationRunner

        # Load dataset
        if args.dataset == "cifar100":
            import torchvision
            import torchvision.transforms as T

            dataset = torchvision.datasets.CIFAR100(
                root=args.data_dir or "./data",
                train=False,  # Test set for final eval
                download=True,
            )
            # Use subset for validation during optimization
            import numpy as np
            n_val = args.val_size or 5000
            indices = np.random.RandomState(42).permutation(len(dataset))[:n_val]
            images = np.array(dataset.data)[indices]
            labels = np.array(dataset.targets)[indices]

            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-B/32",
                device=device,
                images=images,
                labels=labels,
            )
        else:
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-B/32",
                device=device,
            )

        return runner

    elif task_spec.task_type == "segmentation":
        from visprompt.tasks.segmentation import SAMSegmentationRunner

        return SAMSegmentationRunner(
            sam_checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type or "vit_b",
            device=device,
            image_dir=args.data_dir,
            annotation_dir=args.annotation_dir,
        )

    elif task_spec.task_type == "detection":
        from visprompt.tasks.detection import GroundingDINODetectionRunner

        return GroundingDINODetectionRunner(
            model_config=args.gdino_config,
            checkpoint=args.gdino_checkpoint,
            device=device,
            image_dir=args.data_dir,
            annotation_file=args.annotation_file,
        )

    else:
        raise ValueError(f"Unknown task type: {task_spec.task_type}")


def _build_cifar100_spec(args) -> TaskSpec:
    """Build CIFAR-100 task specification."""
    import torchvision
    dataset = torchvision.datasets.CIFAR100(
        root=args.data_dir or "./data", train=False, download=True
    )
    class_names = dataset.classes

    # Build superclass hierarchy
    coarse_labels = [
        "aquatic_mammals", "fish", "flowers", "food_containers",
        "fruit_and_vegetables", "household_electrical_devices",
        "household_furniture", "insects", "large_carnivores",
        "large_man-made_outdoor_things", "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores", "medium_mammals",
        "non-insect_invertebrates", "people", "reptiles",
        "small_mammals", "trees", "vehicles_1", "vehicles_2",
    ]
    # Mapping: each superclass contains 5 fine classes
    fine_to_coarse = {}
    # CIFAR-100 has a standard mapping; simplified here
    hierarchy = {label: [] for label in coarse_labels}

    return TaskSpec(
        task_type="classification",
        dataset_name="cifar100",
        class_names=class_names,
        num_classes=100,
        class_hierarchy=hierarchy if any(hierarchy.values()) else None,
        image_resolution=(32, 32),
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 5000,
    )


def _build_davis2017_spec(args) -> TaskSpec:
    return TaskSpec(
        task_type="segmentation",
        dataset_name="davis2017",
        class_names=[],  # Instance-level, no fixed classes
        num_classes=0,
        image_resolution=None,
        domain="natural",
        foundation_model="sam2",
        prompt_modality="point",
        metric_name="J&F",
    )


def _build_lvis_spec(args) -> TaskSpec:
    # Load class names from annotation file
    class_names = []
    if args.annotation_file:
        with open(args.annotation_file) as f:
            coco = json.load(f)
        class_names = [cat["name"] for cat in coco.get("categories", [])]

    return TaskSpec(
        task_type="detection",
        dataset_name="lvis",
        class_names=class_names,
        num_classes=len(class_names),
        domain="natural",
        foundation_model="grounding_dino",
        prompt_modality="text",
        metric_name="AP_rare",
    )


def _build_from_config(config_path: str) -> TaskSpec:
    """Build TaskSpec from a YAML config file."""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return TaskSpec(
        task_type=cfg["task_type"],
        dataset_name=cfg["dataset_name"],
        class_names=cfg.get("class_names", []),
        num_classes=cfg.get("num_classes", 0),
        class_hierarchy=cfg.get("class_hierarchy"),
        image_resolution=tuple(cfg["image_resolution"]) if cfg.get("image_resolution") else None,
        domain=cfg.get("domain", "natural"),
        foundation_model=cfg.get("foundation_model", "clip"),
        prompt_modality=cfg.get("prompt_modality", "text"),
        metric_name=cfg.get("metric_name", "accuracy"),
        val_split_size=cfg.get("val_split_size"),
        extra=cfg.get("extra", {}),
    )


def main():
    parser = argparse.ArgumentParser(description="Run VisPromptAgent")

    # Task/dataset
    parser.add_argument("--task", choices=["classification", "segmentation", "detection"], required=True)
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name or 'custom'")
    parser.add_argument("--config", type=str, help="Path to YAML config for custom datasets")

    # Data
    parser.add_argument("--data-dir", type=str, help="Dataset directory")
    parser.add_argument("--annotation-dir", type=str, help="Annotation directory (segmentation)")
    parser.add_argument("--annotation-file", type=str, help="Annotation file (detection, COCO format)")
    parser.add_argument("--val-size", type=int, default=5000, help="Validation split size")

    # Models
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    # LLM
    parser.add_argument("--llm", type=str, default="gpt-4o", help="LLM model name")
    parser.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "anthropic", "google"])
    parser.add_argument("--llm-api-key", type=str, help="LLM API key (or set env var)")
    parser.add_argument("--temperature", type=float, default=0.3)

    # Pipeline
    parser.add_argument("--max-iter", type=int, default=3, help="Max refinement iterations")
    parser.add_argument("--output-dir", type=str, default="experiments", help="Output directory")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Build task spec and runner
    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    # Get text embeddings for Analyst (classification only)
    text_embeddings = None
    if task_spec.task_type == "classification" and hasattr(task_runner, "get_text_embeddings"):
        logging.info("Computing text embeddings for Analyst...")
        text_embeddings = task_runner.get_text_embeddings(task_spec.class_names)

    # Run pipeline
    pipeline = VisPromptPipeline(
        task_spec=task_spec,
        task_runner=task_runner,
        llm_model=args.llm,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key,
        llm_temperature=args.temperature,
        text_embeddings=text_embeddings,
        output_dir=args.output_dir,
    )

    result = pipeline.run(max_iterations=args.max_iter)

    # Print results
    print("\n" + result.summary())
    print(f"\nDetailed results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
