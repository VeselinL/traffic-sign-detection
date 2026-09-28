"""Compare independent luminance CLAHE settings with the frozen clean baseline."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import ultralytics
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimization.dataset import DATA_YAML, WEIGHTS, load_images
from src.optimization.clahe import enhance_brightness

FIELDS = [
    "split", "pipeline", "clip_limit", "grid_size", "images", "instances",
    "precision", "recall", "mAP50", "mAP50-95", "mAP50_delta", "mAP50-95_delta",
    "clahe_ms_per_image", "inference_ms_per_image",
]


def prepare_suite(images: list[Path], config: dict, split: str, clip_limit: float,
                  grid_size: int, suite: Path) -> tuple[Path, float]:
    image_dir = suite / "images"
    label_dir = suite / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0
    for source in images:
        target = image_dir / source.name
        if clip_limit == 0:
            if not target.exists():
                target.symlink_to(source)
        else:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Cannot read {source}")
            started = time.perf_counter()
            corrected = enhance_brightness(image, clip_limit, grid_size)
            elapsed += time.perf_counter() - started
            if not cv2.imwrite(str(target), corrected):
                raise RuntimeError(f"Cannot save {target}")
        original_label = source.parent.parent / "labels" / f"{source.stem}.txt"
        if original_label.exists():
            shutil.copyfile(original_label, label_dir / original_label.name)
    split_file = suite / f"{split}.txt"
    split_file.write_text("\n".join(f"./images/{source.name}" for source in images) + "\n")
    suite_config = {
        "train": str(DATA_YAML.parent / config["train"]),
        "val": str(DATA_YAML.parent / config["val"]),
        "test": str(DATA_YAML.parent / config["test"]), "names": config["names"],
    }
    suite_config[split] = str(split_file)
    suite_yaml = suite / "data.yaml"
    suite_yaml.write_text(yaml.safe_dump(suite_config, sort_keys=False))
    (suite / "manifest.json").write_text(json.dumps({
        "split": split, "pipeline": "baseline" if clip_limit == 0 else "clahe_lab_luminance",
        "clip_limit": clip_limit, "grid_size": grid_size, "stochastic": False,
        "source_yaml": str(DATA_YAML), "source_images": [str(path) for path in images],
        "image_count": len(images), "gamma_applied": False,
    }, indent=2) + "\n")
    return suite_yaml, elapsed * 1000 / len(images)


def write_summary(output: Path, rows: list[dict]) -> None:
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# CLAHE optimization — {rows[0]['split']} split", "",
        "CLAHE is applied to LAB luminance only, independently of gamma correction.", "",
        "| Pipeline | Clip limit | Grid | Precision | Recall | mAP50 | mAP50–95 | Δ mAP50–95 | CLAHE ms/image |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['pipeline']} | {row['clip_limit']:g} | {row['grid_size']}×{row['grid_size']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['mAP50']:.4f} | "
            f"{row['mAP50-95']:.4f} | {row['mAP50-95_delta']:+.4f} | {row['clahe_ms_per_image']:.3f} |"
        )
    best = max(rows, key=lambda row: row["mAP50-95"])
    lines.extend(["", f"Highest measured mAP50–95: {best['pipeline']}, clip limit {best['clip_limit']:g}.",
                  "Clip limit 0 denotes unchanged baseline, not a CLAHE parameter.",
                  "Timing includes color conversions and CLAHE, excluding disk reads and writes."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--clip-limits", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0])
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/optimization_1280_clahe")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for value in args.clip_limits) or args.grid_size <= 0:
        parser.error("Clip limits must be positive and finite; grid size must be positive")
    config = yaml.safe_load(DATA_YAML.read_text())
    images = load_images(config, args.split)
    output = args.output.resolve() / args.split
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "weights": str(WEIGHTS), "checkpoint_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "split": args.split, "imgsz": 1280, "batch": args.batch, "device": args.device,
        "clip_limits": args.clip_limits, "grid_size": args.grid_size, "selection_metric": "mAP50-95",
        "gamma_applied": False,
    }, indent=2) + "\n")
    model = YOLO(str(WEIGHTS))
    rows = []
    for clip_limit in dict.fromkeys([0.0, *args.clip_limits]):
        name = "baseline" if clip_limit == 0 else f"clahe_clip_{clip_limit:g}_grid_{args.grid_size}"
        suite = output / "datasets" / name
        suite_yaml, clahe_ms = prepare_suite(images, config, args.split, clip_limit, args.grid_size, suite)
        settings = {
            "data": str(suite_yaml), "split": args.split, "imgsz": 1280, "batch": args.batch,
            "device": args.device, "plots": False, "verbose": False,
            "conf": 0.001, "iou": 0.7, "max_det": 300, "rect": True,
            "half": False, "augment": False,
            "project": str(output / "evaluations"), "name": name, "exist_ok": True,
        }
        metrics = model.val(**settings)
        scores = {
            "split": args.split, "pipeline": "baseline" if clip_limit == 0 else "CLAHE",
            "clip_limit": clip_limit, "grid_size": args.grid_size, "images": len(images),
            "instances": int(metrics.nt_per_class.sum()),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
            "clahe_ms_per_image": clahe_ms, "inference_ms_per_image": float(metrics.speed["inference"]),
            "per_class_mAP50-95": {
                str(class_id): float(value)
                for class_id, value in zip(metrics.box.ap_class_index, metrics.box.ap)
            },
        }
        baseline = rows[0] if rows else scores
        scores["mAP50_delta"] = scores["mAP50"] - baseline["mAP50"]
        scores["mAP50-95_delta"] = scores["mAP50-95"] - baseline["mAP50-95"]
        (suite / "metrics.json").write_text(json.dumps(scores, indent=2) + "\n")
        (suite / "evaluation_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        rows.append(scores)
        write_summary(output, rows)
        print(f"{name}: mAP50={scores['mAP50']:.4f}, mAP50-95={scores['mAP50-95']:.4f}", flush=True)
        if args.split == "test" and clip_limit == 0:
            if abs(scores["mAP50"] - 0.5579) > 0.005 or abs(scores["mAP50-95"] - 0.4921) > 0.005:
                raise RuntimeError("Clean test baseline does not reproduce recorded results")


if __name__ == "__main__":
    main()
