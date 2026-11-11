"""Module 3 — Geometric Transformations & Project Integration

Student 3:
- Geometric Transformations (Rotation, Scaling, Translation)
- Interpolation (Nearest-neighbor, Bilinear, Bicubic)
- Affine Warping
- Gesture Classifier (SVM Recognition)
"""

from .geometric import (
    rotate_image,
    scale_image,
    translate_image,
    affine_warp,
)
from .classifier import GestureClassifier

__all__ = [
    "rotate_image",
    "scale_image",
    "translate_image",
    "affine_warp",
    "GestureClassifier",
]

