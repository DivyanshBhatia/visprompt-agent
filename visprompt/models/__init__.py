"""Convenience wrappers for loading foundation vision models."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_clip(
    model_name: str = "ViT-B/32",
    device: str = "cuda",
    prefer_open_clip: bool = False,
):
    """Load CLIP model, trying openai/clip first then open_clip."""
    if not prefer_open_clip:
        try:
            import clip
            model, preprocess = clip.load(model_name, device=device)
            tokenizer = clip.tokenize
            logger.info(f"Loaded CLIP {model_name} via openai/clip")
            return model, preprocess, tokenizer, "clip"
        except ImportError:
            pass

    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai"
        )
        model = model.to(device)
        tokenizer = open_clip.get_tokenizer(model_name)
        logger.info(f"Loaded CLIP {model_name} via open_clip")
        return model, preprocess, tokenizer, "open_clip"
    except ImportError:
        raise ImportError(
            "Install CLIP: pip install git+https://github.com/openai/CLIP.git "
            "or open_clip: pip install open-clip-torch"
        )


def load_sam(
    model_type: str = "vit_b",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
):
    """Load SAM or SAM 2 model."""
    try:
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        predictor = SamPredictor(sam)
        logger.info(f"Loaded SAM {model_type}")
        return sam, predictor, "sam"
    except ImportError:
        pass

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam = build_sam2(model_type, checkpoint)
        sam.to(device)
        predictor = SAM2ImagePredictor(sam)
        logger.info(f"Loaded SAM 2 {model_type}")
        return sam, predictor, "sam2"
    except ImportError:
        raise ImportError(
            "Install segment-anything or sam2: "
            "pip install segment-anything or "
            "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
        )


def load_grounding_dino(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda",
):
    """Load GroundingDINO model."""
    try:
        from groundingdino.util.inference import load_model
        model = load_model(config_path, checkpoint_path, device=device)
        logger.info("Loaded GroundingDINO")
        return model, "grounding_dino"
    except ImportError:
        raise ImportError(
            "Install GroundingDINO: "
            "pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
        )
