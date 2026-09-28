"""Generate independent degraded GTSDB test suites and evaluate frozen YOLO weights."""

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.degradation.operations import LEVELS, degrade

DATA_YAML = ROOT / "data/GTSDB/dataset/data.yaml"
WEIGHTS = ROOT / "models/yolov8n_baseline_1280/best.pt"
REFERENCE = {"mAP50": 0.5579, "mAP50-95": 0.4921}
TOLERANCE = 0.005


def test_images(config_path: Path, config: dict) -> list[Path]:
    split_path = (config_path.parent / config["test"]).resolve()
    original_paths = [(split_path.parent / line.strip()).resolve() for line in split_path.read_text().splitlines() if line.strip()]
    png_list = split_path.with_name("test_png.txt")
    paths = [(png_list.parent / line.strip()).resolve() for line in png_list.read_text().splitlines() if line.strip()]
    if [path.stem for path in paths] != [path.stem for path in original_paths]:
        raise ValueError("PNG test list differs from the YAML test split")
    if len(paths) != len(set(paths)) or not paths or any(not path.is_file() for path in paths + original_paths):
        raise ValueError("Test list contains duplicate, missing, or no images")
    for original, png in zip(original_paths, paths):
        original_image = cv2.imread(str(original))
        png_image = cv2.imread(str(png))
        if original_image is None or png_image is None or not np.array_equal(original_image, png_image):
            raise ValueError(f"PNG pixels differ from the YAML test image: {png}")
    return paths


def image_seed(seed: int, kind: str, severity: str, stem: str) -> int:
    key = f"{seed}/{kind}/{severity}/{stem}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def generate_suite(source_images: list[Path], names: dict, suite: Path, kind: str, severity: str, seed: int) -> Path:
    image_dir = suite / "images"
    label_dir = suite / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_lines = []
    for source in source_images:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot decode {source}")
        target = image_dir / f"{source.stem}.png"
        result = degrade(image, kind, severity, image_seed(seed, kind, severity, source.stem))
        if not cv2.imwrite(str(target), result):
            raise RuntimeError(f"Cannot save {target}")
        original_label = source.parent.parent / "labels" / f"{source.stem}.txt"
        target_label = label_dir / f"{source.stem}.txt"
        if original_label.exists():
            shutil.copyfile(original_label, target_label)
        image_lines.append(f"./images/{target.name}")
    (suite / "test.txt").write_text("\n".join(image_lines) + "\n")
    suite_yaml = suite / "data.yaml"
    suite_yaml.write_text(yaml.safe_dump({
        "train": str(DATA_YAML.parent / "train_png.txt"),
        "val": str(DATA_YAML.parent / "val_png.txt"),
        "test": "test.txt", "names": names,
    }, sort_keys=False))
    (suite / "manifest.json").write_text(json.dumps({
        "kind": kind, "severity": severity, "parameters": LEVELS[kind][severity],
        "seed": seed, "image_count": len(source_images),
        "source_test_list": str(DATA_YAML.parent / "test.txt"),
        "source_images": [str(path) for path in source_images],
        "weather_is_synthetic": kind in {"fog", "rain"},
        "rain_is_compound": kind == "rain",
    }, indent=2) + "\n")
    return suite_yaml


def prepare_clean_suite(source_images: list[Path], names: dict, suite: Path) -> Path:
    image_dir = suite / "images"
    label_dir = suite / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for source in source_images:
        target = image_dir / source.name
        if not target.exists():
            target.symlink_to(source)
        original_label = source.parent.parent / "labels" / f"{source.stem}.txt"
        if original_label.exists():
            shutil.copyfile(original_label, label_dir / original_label.name)
    (suite / "test.txt").write_text("\n".join(f"./images/{path.name}" for path in source_images) + "\n")
    suite_yaml = suite / "data.yaml"
    suite_yaml.write_text(yaml.safe_dump({
        "train": str(DATA_YAML.parent / "train_png.txt"),
        "val": str(DATA_YAML.parent / "val_png.txt"),
        "test": "test.txt", "names": names,
    }, sort_keys=False))
    return suite_yaml


def evaluate(model: YOLO, data_yaml: Path, run_dir: Path, batch: int, device: str) -> dict:
    metrics = model.val(
        data=str(data_yaml), split="test", imgsz=1280, batch=batch,
        device=device, plots=False, project=str(run_dir.parent),
        name=run_dir.name, exist_ok=True, verbose=False,
    )
    return {
        "instances": int(metrics.nt_per_class.sum()),
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
        "per_class_mAP50-95": {
            str(class_id): float(value)
            for class_id, value in zip(metrics.box.ap_class_index, metrics.box.ap)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/robustness_1280")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", choices=list(LEVELS), help="Evaluate only one degradation after the clean check")
    parser.add_argument("--clean-only", action="store_true", help="Verify the recorded baseline without generating degradations")
    args = parser.parse_args()
    if not DATA_YAML.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError("GTSDB YAML or frozen 1280px checkpoint is missing")
    config = yaml.safe_load(DATA_YAML.read_text())
    originals = test_images(DATA_YAML, config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clean_yaml = prepare_clean_suite(originals, config["names"], output / "datasets/clean")
    model = YOLO(str(WEIGHTS))
    clean = evaluate(model, clean_yaml, output / "clean", args.batch, args.device)
    clean.update({"kind": "clean", "severity": "none", "weights": str(WEIGHTS), "images": len(originals)})
    (output / "clean_metrics.json").write_text(json.dumps(clean, indent=2) + "\n")
    print(f"Clean: mAP50={clean['mAP50']:.4f}, mAP50-95={clean['mAP50-95']:.4f}", flush=True)
    if any(abs(clean[metric] - expected) > TOLERANCE for metric, expected in REFERENCE.items()):
        raise RuntimeError("Clean evaluation differs from recorded baseline; inspect split, weights, and validator settings before degraded evaluation")
    if args.clean_only:
        return
    rows = [clean]
    kinds = [args.only] if args.only else list(LEVELS)
    for kind in kinds:
        for severity in LEVELS[kind]:
            suite = output / "datasets" / kind / severity
            metrics_path = suite / "metrics.json"
            if metrics_path.is_file():
                scores = json.loads(metrics_path.read_text())
                if (scores.get("seed") != args.seed or scores.get("parameters") != LEVELS[kind][severity]
                        or scores.get("weights") != str(WEIGHTS)):
                    raise ValueError(f"Existing suite has different settings: {suite}")
            else:
                scenario_yaml = generate_suite(originals, config["names"], suite, kind, severity, args.seed)
                scores = evaluate(model, scenario_yaml, output / "evaluations" / kind / severity, args.batch, args.device)
            scores.update({
                "kind": kind, "severity": severity, "parameters": LEVELS[kind][severity],
                "seed": args.seed, "weights": str(WEIGHTS), "images": len(originals),
                "mAP50_drop": clean["mAP50"] - scores["mAP50"],
                "mAP50-95_drop": clean["mAP50-95"] - scores["mAP50-95"],
                "scenario": "compound_synthetic_weather" if kind == "rain" else "synthetic_weather" if kind == "fog" else "single_factor",
            })
            metrics_path.write_text(json.dumps(scores, indent=2) + "\n")
            rows.append(scores)
            with (output / "summary.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["kind", "severity", "images", "instances", "precision", "recall", "mAP50", "mAP50-95", "mAP50_drop", "mAP50-95_drop", "scenario"])
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
            print(f"{kind}/{severity}: mAP50={scores['mAP50']:.4f}, mAP50-95={scores['mAP50-95']:.4f}", flush=True)


if __name__ == "__main__":
    main()
