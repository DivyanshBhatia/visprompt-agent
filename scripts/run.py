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

from visprompt.task_spec import TaskSpec


def build_task_spec(args) -> TaskSpec:
    """Build TaskSpec from command-line arguments."""
    if args.dataset == "cifar100":
        return _build_cifar100_spec(args)
    elif args.dataset == "flowers102":
        return _build_flowers102_spec(args)
    elif args.dataset == "dtd":
        return _build_dtd_spec(args)
    elif args.dataset == "eurosat":
        return _build_eurosat_spec(args)
    elif args.dataset == "food101":
        return _build_food101_spec(args)
    elif args.dataset == "cifar10":
        return _build_cifar10_spec(args)
    elif args.dataset == "fgvc_aircraft":
        return _build_fgvc_aircraft_spec(args)
    elif args.dataset == "oxford_pets":
        return _build_oxford_pets_spec(args)
    elif args.dataset == "caltech101":
        return _build_caltech101_spec(args)
    elif args.dataset == "sun397":
        return _build_sun397_spec(args)
    elif args.dataset == "country211":
        return _build_country211_spec(args)
    elif args.dataset == "coco":
        return _build_coco_spec(args)
    elif args.dataset == "davis2017":
        return _build_davis2017_spec(args)
    elif args.dataset == "lvis":
        return _build_lvis_spec(args)
    elif args.config:
        return _build_from_config(args.config)
    else:
        raise ValueError(
            f"Unknown dataset: {args.dataset}. "
            "Supported: cifar10, cifar100, flowers102, dtd, eurosat, food101, "
            "fgvc_aircraft, oxford_pets, caltech101, sun397, country211. "
            "Or use --config for custom datasets."
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
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "flowers102":
            images, labels = _load_pil_dataset(
                "Flowers102", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "dtd":
            images, labels = _load_pil_dataset(
                "DTD", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "eurosat":
            images, labels = _load_pil_dataset(
                "EuroSAT", args, split=None,
                dataset_kwargs={},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "food101":
            images, labels = _load_pil_dataset(
                "Food101", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "cifar10":
            import torchvision
            import numpy as np

            dataset = torchvision.datasets.CIFAR10(
                root=args.data_dir or "./data",
                train=False,
                download=True,
            )
            n_val = min(args.val_size or 10000, len(dataset))
            indices = np.random.RandomState(42).permutation(len(dataset))[:n_val]
            images = np.array(dataset.data)[indices]
            labels = np.array(dataset.targets)[indices]

            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "fgvc_aircraft":
            images, labels = _load_pil_dataset(
                "FGVCAircraft", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "oxford_pets":
            images, labels = _load_pil_dataset(
                "OxfordIIITPet", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "caltech101":
            images, labels = _load_pil_dataset(
                "Caltech101", args, split=None,
                dataset_kwargs={},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "sun397":
            images, labels = _load_pil_dataset(
                "SUN397", args, split=None,
                dataset_kwargs={},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        elif args.dataset == "country211":
            images, labels = _load_pil_dataset(
                "Country211", args, split="test",
                dataset_kwargs={"split": "test"},
            )
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
                device=device,
                images=images,
                labels=labels,
            )

        else:
            runner = CLIPClassificationRunner(
                clip_model_name=args.clip_model or "ViT-L/14",
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
        det_model = getattr(args, 'det_model', 'owlvit')

        if det_model == "owlvit":
            from visprompt.tasks.detection_owlvit import OWLViTDetectionRunner

            return OWLViTDetectionRunner(
                model_name=getattr(args, 'owlvit_model', "google/owlv2-base-patch16-ensemble"),
                device=device,
                image_dir=args.data_dir,
                annotation_file=args.annotation_file,
                max_images=getattr(args, 'max_det_images', None),
            )
        else:
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


def _load_pil_dataset(dataset_name, args, split=None, dataset_kwargs=None):
    """Load a torchvision dataset that returns PIL images into numpy arrays.

    Works for Flowers102, DTD, EuroSAT, and similar datasets.
    Returns (images_array, labels_array) with optional subsampling.
    """
    import numpy as np
    import torchvision

    dataset_cls = getattr(torchvision.datasets, dataset_name)
    dataset = dataset_cls(
        root=args.data_dir or "./data",
        download=True,
        **(dataset_kwargs or {}),
    )

    n_total = len(dataset)
    n_val = min(args.val_size or n_total, n_total)
    indices = np.random.RandomState(42).permutation(n_total)[:n_val]

    logging.info(f"Loading {dataset_name}: {n_val}/{n_total} images...")

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        # Convert PIL to numpy RGB array
        img_np = np.array(img.convert("RGB"))
        images.append(img_np)
        labels.append(label)

    # For variable-size images, store as object array
    # For fixed-size, store as regular array
    try:
        images_arr = np.stack(images)
    except ValueError:
        # Variable image sizes — store as object array
        images_arr = np.empty(len(images), dtype=object)
        for i, img in enumerate(images):
            images_arr[i] = img

    labels_arr = np.array(labels)
    logging.info(f"Loaded {dataset_name}: {len(images_arr)} images, "
                 f"{len(np.unique(labels_arr))} classes")
    return images_arr, labels_arr


# ── Flowers102 class names (Oxford 102 Flowers) ──────────────────────────
FLOWERS102_CLASSES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]

DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered",
    "cobwebbed", "cracked", "crosshatched", "crystalline", "dotted",
    "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid",
    "grooved", "honeycombed", "interlaced", "knitted", "lacelike",
    "lined", "marbled", "matted", "meshed", "paisley", "perforated",
    "pitted", "pleated", "polka-dotted", "porous", "potholed", "scaly",
    "smeared", "spiralled", "sprinkled", "stained", "stratified",
    "striped", "studded", "swirly", "veined", "waffled", "woven",
    "wrinkled", "zigzagged",
]

EUROSAT_CLASSES = [
    "annual crop land", "forest", "herbaceous vegetation land",
    "highway", "industrial buildings", "pasture", "permanent crop land",
    "residential buildings", "river", "sea or lake",
]


def _build_flowers102_spec(args) -> TaskSpec:
    """Build Flowers102 task specification."""
    return TaskSpec(
        task_type="classification",
        dataset_name="flowers102",
        class_names=FLOWERS102_CLASSES,
        num_classes=102,
        class_hierarchy=None,
        image_resolution=None,  # Variable size
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 6149,
    )


def _build_dtd_spec(args) -> TaskSpec:
    """Build DTD (Describable Textures Dataset) task specification."""
    import torchvision
    dataset = torchvision.datasets.DTD(
        root=args.data_dir or "./data", split="test", download=True
    )
    # DTD exposes .classes
    class_names = dataset.classes if hasattr(dataset, 'classes') else DTD_CLASSES
    return TaskSpec(
        task_type="classification",
        dataset_name="dtd",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,  # Variable size
        domain="textures",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 1880,
    )


def _build_eurosat_spec(args) -> TaskSpec:
    """Build EuroSAT task specification."""
    import torchvision
    try:
        dataset = torchvision.datasets.EuroSAT(
            root=args.data_dir or "./data", download=True
        )
        # EuroSAT is an ImageFolder, has .classes
        class_names = dataset.classes if hasattr(dataset, 'classes') else EUROSAT_CLASSES
        # Clean up folder names to readable names
        eurosat_name_map = {
            "AnnualCrop": "annual crop land",
            "Forest": "forest",
            "HerbaceousVegetation": "herbaceous vegetation land",
            "Highway": "highway",
            "Industrial": "industrial buildings",
            "Pasture": "pasture",
            "PermanentCrop": "permanent crop land",
            "Residential": "residential buildings",
            "River": "river",
            "SeaLake": "sea or lake",
        }
        class_names = [eurosat_name_map.get(c, c) for c in class_names]
    except Exception:
        class_names = EUROSAT_CLASSES

    return TaskSpec(
        task_type="classification",
        dataset_name="eurosat",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=(64, 64),
        domain="remote_sensing",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 5000,
    )


def _build_food101_spec(args) -> TaskSpec:
    """Build Food101 task specification."""
    import torchvision
    dataset = torchvision.datasets.Food101(
        root=args.data_dir or "./data", split="test", download=True
    )
    # Food101 has .classes — folder names with underscores
    class_names = [c.replace("_", " ") for c in dataset.classes]
    return TaskSpec(
        task_type="classification",
        dataset_name="food101",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,  # Variable size
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 10000,
    )


def _build_cifar10_spec(args) -> TaskSpec:
    """Build CIFAR-10 task specification."""
    import torchvision
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir or "./data", train=False, download=True
    )
    class_names = list(dataset.classes)
    return TaskSpec(
        task_type="classification",
        dataset_name="cifar10",
        class_names=class_names,
        num_classes=10,
        class_hierarchy=None,
        image_resolution=(32, 32),
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 10000,
    )


def _build_fgvc_aircraft_spec(args) -> TaskSpec:
    """Build FGVC-Aircraft task specification."""
    import torchvision
    dataset = torchvision.datasets.FGVCAircraft(
        root=args.data_dir or "./data", split="test", download=True
    )
    # FGVCAircraft has .classes
    class_names = list(dataset.classes) if hasattr(dataset, 'classes') else []
    if not class_names:
        # Fallback: extract from dataset
        class_names = sorted(set(dataset._labels))
    return TaskSpec(
        task_type="classification",
        dataset_name="fgvc_aircraft",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,  # Variable size
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 3333,
    )


def _build_oxford_pets_spec(args) -> TaskSpec:
    """Build Oxford-IIIT Pets task specification."""
    import torchvision
    dataset = torchvision.datasets.OxfordIIITPet(
        root=args.data_dir or "./data", split="test", download=True
    )
    class_names = list(dataset.classes) if hasattr(dataset, 'classes') else []
    if not class_names:
        # Fallback: known 37 breeds
        class_names = sorted(set(dataset._labels)) if hasattr(dataset, '_labels') else []
    # Clean names: "Abyssinian" -> "abyssinian"
    class_names = [c.replace("_", " ") for c in class_names]
    return TaskSpec(
        task_type="classification",
        dataset_name="oxford_pets",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 3669,
    )


def _build_caltech101_spec(args) -> TaskSpec:
    """Build Caltech-101 task specification."""
    import torchvision
    dataset = torchvision.datasets.Caltech101(
        root=args.data_dir or "./data", download=True
    )
    class_names = list(dataset.categories)
    # Clean names
    class_names = [c.replace("_", " ") for c in class_names]
    return TaskSpec(
        task_type="classification",
        dataset_name="caltech101",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,
        domain="natural",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 6084,
    )


def _build_sun397_spec(args) -> TaskSpec:
    """Build SUN397 task specification."""
    import torchvision
    dataset = torchvision.datasets.SUN397(
        root=args.data_dir or "./data", download=True
    )
    # SUN397 class names are paths like "/a/abbey" — clean them
    if hasattr(dataset, 'classes'):
        class_names = [c.split("/")[-1].replace("_", " ") for c in dataset.classes]
    else:
        class_names = [f"scene_{i}" for i in range(397)]
    return TaskSpec(
        task_type="classification",
        dataset_name="sun397",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,
        domain="scenes",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 10000,
    )


ISO_TO_COUNTRY = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AF": "Afghanistan",
    "AG": "Antigua and Barbuda", "AI": "Anguilla", "AL": "Albania",
    "AM": "Armenia", "AO": "Angola", "AR": "Argentina", "AT": "Austria",
    "AU": "Australia", "AW": "Aruba", "AZ": "Azerbaijan", "BA": "Bosnia and Herzegovina",
    "BB": "Barbados", "BD": "Bangladesh", "BE": "Belgium", "BF": "Burkina Faso",
    "BG": "Bulgaria", "BH": "Bahrain", "BJ": "Benin", "BM": "Bermuda",
    "BN": "Brunei", "BO": "Bolivia", "BR": "Brazil", "BS": "Bahamas",
    "BT": "Bhutan", "BW": "Botswana", "BY": "Belarus", "BZ": "Belize",
    "CA": "Canada", "CD": "Democratic Republic of the Congo", "CF": "Central African Republic",
    "CH": "Switzerland", "CI": "Ivory Coast", "CL": "Chile", "CM": "Cameroon",
    "CN": "China", "CO": "Colombia", "CR": "Costa Rica", "CU": "Cuba",
    "CW": "Curacao", "CY": "Cyprus", "CZ": "Czech Republic", "DE": "Germany",
    "DK": "Denmark", "DM": "Dominica", "DO": "Dominican Republic", "DZ": "Algeria",
    "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt", "ES": "Spain",
    "ET": "Ethiopia", "FI": "Finland", "FJ": "Fiji", "FK": "Falkland Islands",
    "FO": "Faroe Islands", "FR": "France", "GA": "Gabon", "GB": "United Kingdom",
    "GD": "Grenada", "GE": "Georgia", "GF": "French Guiana", "GG": "Guernsey",
    "GH": "Ghana", "GI": "Gibraltar", "GL": "Greenland", "GM": "Gambia",
    "GP": "Guadeloupe", "GR": "Greece", "GT": "Guatemala", "GU": "Guam",
    "GY": "Guyana", "HK": "Hong Kong", "HN": "Honduras", "HR": "Croatia",
    "HT": "Haiti", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland",
    "IL": "Israel", "IM": "Isle of Man", "IN": "India", "IQ": "Iraq",
    "IR": "Iran", "IS": "Iceland", "IT": "Italy", "JE": "Jersey",
    "JM": "Jamaica", "JO": "Jordan", "JP": "Japan", "KE": "Kenya",
    "KG": "Kyrgyzstan", "KH": "Cambodia", "KN": "Saint Kitts and Nevis",
    "KR": "South Korea", "KW": "Kuwait", "KY": "Cayman Islands", "KZ": "Kazakhstan",
    "LA": "Laos", "LB": "Lebanon", "LC": "Saint Lucia", "LI": "Liechtenstein",
    "LK": "Sri Lanka", "LR": "Liberia", "LS": "Lesotho", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya", "MA": "Morocco",
    "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro", "MF": "Saint Martin",
    "MG": "Madagascar", "MK": "North Macedonia", "ML": "Mali", "MM": "Myanmar",
    "MN": "Mongolia", "MO": "Macau", "MQ": "Martinique", "MR": "Mauritania",
    "MT": "Malta", "MU": "Mauritius", "MV": "Maldives", "MW": "Malawi",
    "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique", "NA": "Namibia",
    "NC": "New Caledonia", "NE": "Niger", "NG": "Nigeria", "NI": "Nicaragua",
    "NL": "Netherlands", "NO": "Norway", "NP": "Nepal", "NZ": "New Zealand",
    "OM": "Oman", "PA": "Panama", "PE": "Peru", "PF": "French Polynesia",
    "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PR": "Puerto Rico", "PS": "Palestine", "PT": "Portugal", "PW": "Palau",
    "PY": "Paraguay", "QA": "Qatar", "RE": "Reunion", "RO": "Romania",
    "RS": "Serbia", "RU": "Russia", "RW": "Rwanda", "SA": "Saudi Arabia",
    "SC": "Seychelles", "SD": "Sudan", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SL": "Sierra Leone", "SM": "San Marino",
    "SN": "Senegal", "SO": "Somalia", "SR": "Suriname", "SV": "El Salvador",
    "SX": "Sint Maarten", "SY": "Syria", "SZ": "Eswatini", "TC": "Turks and Caicos Islands",
    "TG": "Togo", "TH": "Thailand", "TJ": "Tajikistan", "TL": "East Timor",
    "TN": "Tunisia", "TO": "Tonga", "TR": "Turkey", "TT": "Trinidad and Tobago",
    "TW": "Taiwan", "TZ": "Tanzania", "UA": "Ukraine", "UG": "Uganda",
    "US": "United States", "UY": "Uruguay", "UZ": "Uzbekistan",
    "VC": "Saint Vincent and the Grenadines", "VE": "Venezuela", "VG": "British Virgin Islands",
    "VI": "US Virgin Islands", "VN": "Vietnam", "VU": "Vanuatu", "WS": "Samoa",
    "XK": "Kosovo", "YE": "Yemen", "YT": "Mayotte", "ZA": "South Africa",
    "ZM": "Zambia", "ZW": "Zimbabwe",
}


def _build_country211_spec(args) -> TaskSpec:
    """Build Country211 task specification."""
    import torchvision
    dataset = torchvision.datasets.Country211(
        root=args.data_dir or "./data", split="test", download=True
    )
    # Get ISO codes from dataset
    iso_codes = list(dataset.classes) if hasattr(dataset, 'classes') else []
    if not iso_codes:
        iso_codes = [f"country_{i}" for i in range(211)]
    # Map ISO codes to full country names for CLIP
    class_names = [ISO_TO_COUNTRY.get(c.strip(), c) for c in iso_codes]
    return TaskSpec(
        task_type="classification",
        dataset_name="country211",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,
        domain="geolocation",
        foundation_model="clip",
        prompt_modality="text",
        metric_name="top1_accuracy",
        val_split_size=args.val_size or 10000,
    )


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _build_coco_spec(args) -> TaskSpec:
    """Build COCO detection task specification."""
    # Try loading class names from annotation file
    class_names = COCO_CLASSES
    if args.annotation_file:
        try:
            import json
            with open(args.annotation_file) as f:
                coco = json.load(f)
            cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
            class_names = [c["name"] for c in cats]
        except Exception as e:
            logging.warning(f"Could not load COCO categories from annotation file: {e}")

    return TaskSpec(
        task_type="detection",
        dataset_name="coco",
        class_names=class_names,
        num_classes=len(class_names),
        class_hierarchy=None,
        image_resolution=None,
        domain="natural",
        foundation_model="owlvit",
        prompt_modality="text",
        metric_name="mAP",
        val_split_size=args.val_size,
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
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)
    parser.add_argument("--det-model", type=str, default="owlvit",
                        choices=["owlvit", "gdino"], help="Detection model")
    parser.add_argument("--owlvit-model", type=str,
                        default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--max-det-images", type=int, default=None,
                        help="Max images for detection eval (None=all)")
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

    # Run single-shot description generation + weighted fusion
    from scripts.run_weight_ablation import generate_descriptions, build_prompts_with_weights
    from visprompt.baselines import IMAGENET_TEMPLATES

    # Templates-only baseline
    logging.info("Running templates-only baseline...")
    baseline_prompts = build_prompts_with_weights(
        task_spec.class_names, {}, 1.0, 0.0
    )
    baseline_result = task_runner.evaluate(baseline_prompts, task_spec)
    print(f"\nTemplates-only: {baseline_result.primary_metric:.4f}")

    # Generate LLM descriptions
    logging.info(f"Generating descriptions with {args.llm}...")
    descriptions, desc_cost = generate_descriptions(
        task_spec, args.llm, args.llm_provider
    )

    # Weight ablation
    WEIGHT_CONFIGS = [
        (1.0, 0.0, "100/0"),
        (0.85, 0.15, "85/15"),
        (0.70, 0.30, "70/30"),
        (0.55, 0.45, "55/45"),
        (0.40, 0.60, "40/60"),
        (0.20, 0.80, "20/80"),
        (0.0, 1.0, "0/100"),
    ]

    print(f"\n{'Config':<12} {'Accuracy':>10} {'Δ':>10}")
    print(f"{'-'*35}")

    best_acc = 0
    best_config = ""
    for base_w, desc_w, label in WEIGHT_CONFIGS:
        prompts = build_prompts_with_weights(
            task_spec.class_names, descriptions, base_w, desc_w
        )
        result = task_runner.evaluate(prompts, task_spec)
        acc = result.primary_metric
        delta = acc - baseline_result.primary_metric
        marker = " ← best" if acc > best_acc else ""
        print(f"  {label:<10} {acc:>10.4f} {delta:>+9.4f}{marker}")
        if acc > best_acc:
            best_acc = acc
            best_config = label

    print(f"\nBest: {best_config} → {best_acc:.4f} "
          f"(Δ {best_acc - baseline_result.primary_metric:+.4f})")
    print(f"Description cost: {desc_cost}")
    print(f"\nDetailed results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
