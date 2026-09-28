"""Convert GTSDB PPM images to PNG and create matching split manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "GTSDB" / "dataset"
SPLITS = ("train", "val", "test", "trainval")


def convert_images(images_dir: Path) -> tuple[int, int]:
    """Create PNG copies for every PPM image, leaving existing files untouched."""
    converted = 0
    skipped = 0

    for ppm_path in sorted(images_dir.rglob("*.ppm")):
        png_path = ppm_path.with_suffix(".png")
        if png_path.exists():
            skipped += 1
            continue

        image = cv2.imread(str(ppm_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {ppm_path}")
        if not cv2.imwrite(str(png_path), image):
            raise OSError(f"Could not write image: {png_path}")
        converted += 1

    return converted, skipped


def create_png_manifest(manifest_path: Path) -> Path:
    """Create a manifest whose PPM references are replaced by PNG references."""
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    png_lines = [
        f"{line[:-4]}.png" if line.lower().endswith(".ppm") else line for line in lines
    ]
    output_path = manifest_path.with_stem(f"{manifest_path.stem}_png")
    output_path.write_text("\n".join(png_lines) + "\n", encoding="utf-8")
    return output_path


def convert_dataset(dataset_dir: Path) -> None:
    """Convert images and create PNG variants of all GTSDB split manifests."""
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    converted, skipped = convert_images(images_dir)
    print(f"PNG images: converted={converted}, already existed={skipped}")

    for split in SPLITS:
        manifest_path = dataset_dir / f"{split}.txt"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        output_path = create_png_manifest(manifest_path)
        print(f"Created {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GTSDB PPM images to PNG and create PNG split manifests."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args().dataset_dir)
