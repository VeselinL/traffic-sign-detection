"""CLAHE on LAB luminance, with the LAB chromatic channels unchanged."""

import cv2
import numpy as np


def enhance_brightness(image: np.ndarray, clip_limit: float, grid_size: int = 8) -> np.ndarray:
    """Enhance local luminance contrast independently of gamma correction."""
    if not np.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("Clip limit must be positive and finite")
    if not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("Grid size must be a positive integer")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("CLAHE requires a three-channel uint8 BGR image")
    luminance, channel_a, channel_b = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
    transform = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    corrected = cv2.merge((transform.apply(luminance), channel_a, channel_b))
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
