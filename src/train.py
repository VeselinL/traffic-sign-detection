from pathlib import Path
from ultralytics import YOLO
import ultralytics

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GTSDB_PATH = PROJECT_ROOT / "data" / "GTSDB"

data_yaml = GTSDB_PATH / "dataset" / "data.yaml"
project_dir = PROJECT_ROOT / "runs" / "detect" / "train"
run_name = "yolov8n_baseline_1280"
assert data_yaml.is_file(), f"Dataset config missing: {data_yaml}"
assert not (project_dir / run_name).exists(), "Run folder already exists"

def main():

    print("Ultralytics:", ultralytics.__version__)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=100,
        imgsz=1280,
        batch=4,
        device=0,
        workers=2,
        cache="ram",
        seed=42,
        deterministic=True,
        optimizer="auto",
        amp=True,
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.0,
        flipud=0.0,
        multi_scale=0.0,
        val=True,
        plots=True,
        save=True,
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()