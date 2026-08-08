from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

NUM_CLASSES = 43
NUM_IMAGES = 900
NUM_OFFICIAL_TRAIN_IMAGES = 600
NUM_TRAIN_IMAGES = 480
NUM_VAL_IMAGES = 120
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "GTSDB" / "dataset"
IMAGES_PATH = DATASET_PATH / "images"
LABELS_PATH = DATASET_PATH / "labels"
CLASS_NAMES = (
    "speed limit 20",
    "speed limit 30",
    "speed limit 50",
    "speed limit 60",
    "speed limit 70",
    "speed limit 80",
    "restriction ends 80",
    "speed limit 100",
    "speed limit 120",
    "no overtaking",
    "no overtaking (trucks)",
    "priority at next intersection",
    "priority road",
    "give way",
    "stop",
    "no traffic both ways",
    "no trucks",
    "no entry",
    "danger",
    "bend left",
    "bend right",
    "bend",
    "uneven road",
    "slippery road",
    "road narrows",
    "construction",
    "traffic signal",
    "pedestrian crossing",
    "school crossing",
    "cycles crossing",
    "snow",
    "animals",
    "restriction ends",
    "go right",
    "go left",
    "go straight",
    "go right or straight",
    "go left or straight",
    "keep right",
    "keep left",
    "roundabout",
    "restriction ends (overtaking)",
    "restriction ends (overtaking trucks)",
)

def load_dataset() -> tuple[list[Path], np.ndarray]:
    images = [IMAGES_PATH / f"{image_id:05d}.ppm" for image_id in range(NUM_IMAGES)]
    missing_images = [image.name for image in images if not image.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing images: {', '.join(missing_images[:10])}")

    class_counts = np.zeros((NUM_IMAGES, NUM_CLASSES), dtype=np.uint16)
    for image_id, image in enumerate(images):
        label_path = LABELS_PATH / f"{image.stem}.txt"
        if not label_path.exists():
            continue

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid label at {label_path}:{line_number}")

            class_id = int(fields[0])
            if not 0 <= class_id < NUM_CLASSES:
                raise ValueError(f"Invalid class {class_id} at {label_path}:{line_number}")
            class_counts[image_id, class_id] += 1

    return images, class_counts

def build_stratification_targets(class_counts: np.ndarray) -> np.ndarray:
    class_presence = class_counts > 0
    background = (class_counts.sum(axis=1) == 0)[:, np.newaxis]
    return np.hstack((class_presence, background))

def save_manifest(name: str, images: list[Path]) -> None:
    lines = [f"./images/{image.name}" for image in sorted(images)]
    (DATASET_PATH / f"{name}.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

def save_split_manifests(
    split_indices: dict[str, np.ndarray], images: list[Path]
) -> None:
    for name, indices in split_indices.items():
        save_manifest(name, [images[index] for index in indices])

def create_data_yaml() -> Path:
    data_yaml = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    output_path = DATASET_PATH / "data.yaml"
    output_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    return output_path

def print_summary(name: str, indices: np.ndarray, class_counts: np.ndarray) -> None:
    split_counts = class_counts[indices]
    class_totals = split_counts.sum(axis=0)
    backgrounds = int(np.count_nonzero(split_counts.sum(axis=1) == 0))
    missing_classes = np.flatnonzero(class_totals == 0).tolist()

    print(
        f"{name}: images={len(indices)}, boxes={int(class_totals.sum())}, "
        f"backgrounds={backgrounds}, classes={NUM_CLASSES - len(missing_classes)}/{NUM_CLASSES}"
    )
    if missing_classes:
        print(f"  missing classes: {missing_classes}")

def split_gtsdb_dataset() -> None:
    images, class_counts = load_dataset()
    development_targets = build_stratification_targets(
        class_counts[:NUM_OFFICIAL_TRAIN_IMAGES]
    )

    splitter = MultilabelStratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )
    train_indices, val_indices = next(
        splitter.split(np.zeros(NUM_OFFICIAL_TRAIN_IMAGES), development_targets)
    )
    test_indices = np.arange(NUM_OFFICIAL_TRAIN_IMAGES, NUM_IMAGES)

    if len(train_indices) != NUM_TRAIN_IMAGES or len(val_indices) != NUM_VAL_IMAGES:
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_indices)}, val={len(val_indices)}"
        )
    if np.any(class_counts[train_indices].sum(axis=0) == 0):
        raise RuntimeError("The training split does not contain every class")

    split_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
        "trainval": np.arange(NUM_OFFICIAL_TRAIN_IMAGES),
    }
    save_split_manifests(split_indices, images)
    data_yaml_path = create_data_yaml()

    for name, indices in split_indices.items():
        print_summary(name, indices, class_counts)
    print(f"Saved dataset configuration to {data_yaml_path}")

if __name__ == "__main__":
    split_gtsdb_dataset()
