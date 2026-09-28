from pathlib import Path
import json
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GTSDB_PATH = PROJECT_ROOT / "data" / "GTSDB"
data_yaml = GTSDB_PATH / "dataset" / "data.yaml"
project_dir = PROJECT_ROOT / "runs" / "detect" / "train"
models_dir = PROJECT_ROOT / "models"

weights = models_dir / "yolov8n_baseline_1280" / "weights" / "best.pt"
output_dir = project_dir / "Evaluations"

assert data_yaml.is_file(), f"Missing dataset YAML: {data_yaml}"
assert weights.is_file(), f"Missing weights: {weights}"
assert not output_dir.exists(), f"Evaluation already exists: {output_dir}"
assert yaml.safe_load(data_yaml.read_text()).get("test"), "No test split in YAML"

def main():

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=1280,
        batch=8,
        device=0,
        plots=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=False,
    )

    summary = {
        "weights": str(weights),
        "split": "test",
        "imgsz": 1280,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary)

if __name__ == "__main__":
    main()