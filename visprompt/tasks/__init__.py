from visprompt.tasks.base import BaseTaskRunner
from visprompt.tasks.classification import CLIPClassificationRunner
from visprompt.tasks.segmentation import SAMSegmentationRunner
from visprompt.tasks.detection import GroundingDINODetectionRunner

__all__ = [
    "BaseTaskRunner",
    "CLIPClassificationRunner",
    "SAMSegmentationRunner",
    "GroundingDINODetectionRunner",
]
