"""Evaluate frozen gamma and CLAHE preprocessing after each test corruption."""

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

from src.optimization.dataset import DATA_YAML, WEIGHTS
from src.optimization.clahe import enhance_brightness
from src.optimization.gamma import brighten

ROBUSTNESS_ROOT = ROOT / "runs/robustness_1280"
GAMMA = 0.9
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8
FIELDS = [
    "degradation", "severity", "pipeline", "images", "instances", "precision", "recall",
    "mAP50", "mAP50-95", "mAP50_delta", "mAP50-95_delta",
    "preprocess_ms_per_image", "inference_ms_per_image",
]


def source_scenarios(root: Path) -> list[tuple[str, str, Path, dict]]:
    scenarios = []
    for metrics_path in sorted(root.glob("datasets/*/*/metrics.json")):
        source_suite = metrics_path.parent
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("kind") in {None, "clean"}:
            continue
        scenarios.append((metrics["kind"], metrics["severity"], source_suite, metrics))
    if len(scenarios) != 30:
        raise ValueError(f"Expected 30 completed degradation suites, found {len(scenarios)}")
    if len({(kind, severity) for kind, severity, _, _ in scenarios}) != len(scenarios):
        raise ValueError("Duplicate degradation/severity suites")
    return scenarios


def source_images(source_suite: Path) -> list[Path]:
    list_path = source_suite / "test.txt"
    images = [(list_path.parent / line.strip()).resolve()
              for line in list_path.read_text().splitlines() if line.strip()]
    if len(images) != 300 or len(images) != len(set(images)) or any(not path.is_file() for path in images):
        raise ValueError(f"Expected 300 existing images in {source_suite}")
    return images


def process(image: np.ndarray, pipeline: str) -> np.ndarray:
    if pipeline == "gamma_0.9":
        return brighten(image, GAMMA)
    if pipeline == "clahe_clip_2_grid_8":
        return enhance_brightness(image, CLAHE_CLIP_LIMIT, CLAHE_GRID_SIZE)
    raise ValueError(f"Unknown preprocessing pipeline: {pipeline}")


def prepare_suite(source_suite: Path, destination: Path, pipeline: str, names: dict) -> tuple[Path, float]:
    source_paths = source_images(source_suite)
    image_dir = destination / "images"
    label_dir = destination / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0
    for source in source_paths:
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Cannot read generated corruption {source}")
        started = time.perf_counter()
        corrected = process(image, pipeline)
        elapsed += time.perf_counter() - started
        target = image_dir / source.name
        if not cv2.imwrite(str(target), corrected):
            raise RuntimeError(f"Cannot save {target}")
        source_label = source_suite / "labels" / f"{source.stem}.txt"
        if source_label.exists():
            shutil.copyfile(source_label, label_dir / source_label.name)
    test_list = destination / "test.txt"
    test_list.write_text("\n".join(f"./images/{path.name}" for path in source_paths) + "\n")
    data_yaml = destination / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "train": str(DATA_YAML.parent / "train_png.txt"),
        "val": str(DATA_YAML.parent / "val_png.txt"),
        "test": "test.txt", "names": names,
    }, sort_keys=False))
    parameters = {"gamma": GAMMA} if pipeline == "gamma_0.9" else {
        "clip_limit": CLAHE_CLIP_LIMIT, "grid_size": CLAHE_GRID_SIZE, "color_space": "LAB luminance",
    }
    (destination / "manifest.json").write_text(json.dumps({
        "source_degraded_suite": str(source_suite), "post_degradation_pipeline": pipeline,
        "parameters": parameters, "image_count": len(source_paths),
        "operation_order": "original image -> named degradation -> preprocessing -> YOLO evaluation",
        "gamma_and_clahe_combined": False,
    }, indent=2) + "\n")
    return data_yaml, elapsed * 1000 / len(source_paths)


def write_summary(output: Path, rows: list[dict]) -> None:
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Post-degradation preprocessing evaluation", "",
        "Every candidate begins with the same saved degraded image. Gamma 0.9 and CLAHE clip limit 2/grid 8 are applied separately; they are never combined.", "",
        "Positive deltas are relative to the existing unprocessed result for the same degradation and severity.", "",
        "| Degradation | Severity | Pipeline | mAP50 | mAP50–95 | Δ mAP50–95 | Recall | Preprocess ms/image |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['degradation']} | {row['severity']} | {row['pipeline']} | {row['mAP50']:.4f} | "
            f"{row['mAP50-95']:.4f} | {row['mAP50-95_delta']:+.4f} | {row['recall']:.4f} | "
            f"{row['preprocess_ms_per_image']:.3f} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/degraded_preprocessing_1280")
    parser.add_argument("--only", choices=["gamma_0.9", "clahe_clip_2_grid_8"], help="Run one frozen pipeline")
    parser.add_argument("--degradation", help="Run one named degradation after checking it exists")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not DATA_YAML.is_file() or not WEIGHTS.is_file() or not ROBUSTNESS_ROOT.is_dir():
        raise FileNotFoundError("Dataset YAML, frozen checkpoint, or completed robustness output is missing")
    config = yaml.safe_load(DATA_YAML.read_text())
    scenarios = source_scenarios(ROBUSTNESS_ROOT)
    if args.degradation:
        scenarios = [scenario for scenario in scenarios if scenario[0] == args.degradation]
        if not scenarios:
            raise ValueError(f"No completed degradation named {args.degradation}")
    pipelines = [args.only] if args.only else ["gamma_0.9", "clahe_clip_2_grid_8"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_metadata.json").write_text(json.dumps({
        "weights": str(WEIGHTS), "checkpoint_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        "ultralytics": ultralytics.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
        "imgsz": 1280, "batch": args.batch, "device": args.device,
        "pipelines": pipelines, "source_robustness_root": str(ROBUSTNESS_ROOT),
        "gamma": GAMMA, "clahe_clip_limit": CLAHE_CLIP_LIMIT, "clahe_grid_size": CLAHE_GRID_SIZE,
    }, indent=2) + "\n")
    model = YOLO(str(WEIGHTS))
    rows = []
    for kind, severity, source_suite, baseline in scenarios:
        for pipeline in pipelines:
            suite = output / "datasets" / kind / severity / pipeline
            metrics_path = suite / "metrics.json"
            if metrics_path.is_file():
                scores = json.loads(metrics_path.read_text())
            else:
                suite_yaml, preprocess_ms = prepare_suite(source_suite, suite, pipeline, config["names"])
                settings = {
                    "data": str(suite_yaml), "split": "test", "imgsz": 1280, "batch": args.batch,
                    "device": args.device, "plots": False, "verbose": False,
                    "conf": 0.001, "iou": 0.7, "max_det": 300, "rect": True,
                    "half": False, "augment": False,
                    "project": str(output / "evaluations" / kind / severity), "name": pipeline, "exist_ok": True,
                }
                metrics = model.val(**settings)
                scores = {
                    "degradation": kind, "severity": severity, "pipeline": pipeline,
                    "images": 300, "instances": int(metrics.nt_per_class.sum()),
                    "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                    "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map),
                    "preprocess_ms_per_image": preprocess_ms,
                    "inference_ms_per_image": float(metrics.speed["inference"]),
                    "per_class_mAP50-95": {
                        str(class_id): float(value)
                        for class_id, value in zip(metrics.box.ap_class_index, metrics.box.ap)
                    },
                    "evaluation_settings": settings,
                }
                metrics_path.write_text(json.dumps(scores, indent=2) + "\n")
            scores["mAP50_delta"] = scores["mAP50"] - baseline["mAP50"]
            scores["mAP50-95_delta"] = scores["mAP50-95"] - baseline["mAP50-95"]
            metrics_path.write_text(json.dumps(scores, indent=2) + "\n")
            rows.append(scores)
            write_summary(output, rows)
            print(f"{kind}/{severity}/{pipeline}: mAP50-95={scores['mAP50-95']:.4f} ({scores['mAP50-95_delta']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
