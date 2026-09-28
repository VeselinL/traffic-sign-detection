"""Dataset paths and split verification for frozen-model preprocessing."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "data/GTSDB/dataset/data.yaml"
WEIGHTS = ROOT / "models/yolov8n_baseline_1280/best.pt"


def load_images(config: dict, split: str) -> list[Path]:
    original_list = (DATA_YAML.parent / config[split]).resolve()
    png_list = original_list.with_name(f"{original_list.stem}_png.txt")
    originals = [(original_list.parent / line.strip()).resolve()
                 for line in original_list.read_text().splitlines() if line.strip()]
    images = [(png_list.parent / line.strip()).resolve()
              for line in png_list.read_text().splitlines() if line.strip()]
    if not images or len(images) != len(set(images)):
        raise ValueError("Empty or duplicate image list")
    if [path.stem for path in originals] != [path.stem for path in images]:
        raise ValueError("PNG image IDs differ from the original split")
    for original, png in zip(originals, images):
        original_image = cv2.imread(str(original))
        png_image = cv2.imread(str(png))
        if original_image is None or png_image is None or not np.array_equal(original_image, png_image):
            raise ValueError(f"PNG does not match the original image: {png}")
    return images
