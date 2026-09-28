"""Deterministic gamma correction for OpenCV uint8 images."""

import cv2
import numpy as np


def brighten(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply 255 * (I / 255) ** gamma; gamma below one brightens."""
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be positive and finite")
    if image.dtype != np.uint8:
        raise ValueError("Gamma correction requires a uint8 image")
    lookup = np.clip(255.0 * (np.arange(256, dtype=np.float64) / 255.0) ** gamma, 0, 255)
    return cv2.LUT(image, lookup.astype(np.uint8))
