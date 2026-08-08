from collections import defaultdict
from pathlib import Path

import cv2

IMAGE_HEIGHT, IMAGE_WIDTH = 800, 1360
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GTSDB_PATH = PROJECT_ROOT / "data" / "GTSDB"

def convert_row_to_yolo(img: str):
    """Convert one GTSDB row to YOLO format: class_id center_x center_y width height."""
    img_name, left_col, top_row, right_col, bot_row, class_id = img.split(";")
    box_width = int(right_col) - int(left_col)
    box_height = int(bot_row) - int(top_row)

    center_x = int(left_col) + box_width / 2.0
    center_y = int(top_row) + box_height / 2.0

    norm_center_x = center_x / IMAGE_WIDTH
    norm_center_y = center_y / IMAGE_HEIGHT
    norm_width = box_width / IMAGE_WIDTH
    norm_height = box_height / IMAGE_HEIGHT

    yolo_line = (
        f"{class_id.strip()} {norm_center_x:.6f} {norm_center_y:.6f} "
        f"{norm_width:.6f} {norm_height:.6f}"
    )
    return img_name.strip(), yolo_line

def process_gtsdb_annotations(annotation_lines) -> dict[str, list[str]]:
    """Groups multiple bounding boxes by their respective image name."""
    image_to_labels = defaultdict(list)

    for line in annotation_lines:
        if not line.strip():
            continue
        img_name, yolo_line = convert_row_to_yolo(line)
        image_to_labels[img_name].append(yolo_line)

    return image_to_labels

def test_conversion(img_name, yolo_lines):
    """Denormalizes coordinates to test and visualize multiple boxes."""
    img_path = GTSDB_PATH / "dataset" / "images" / img_name
    img = cv2.imread(str(img_path))

    for yolo_format in yolo_lines:
        class_id, norm_cx, norm_cy, norm_w, norm_h = yolo_format.split(" ")

        center_x = float(norm_cx) * IMAGE_WIDTH
        center_y = float(norm_cy) * IMAGE_HEIGHT
        box_width = float(norm_w) * IMAGE_WIDTH
        box_height = float(norm_h) * IMAGE_HEIGHT

        left_col = center_x - (box_width / 2.0)
        right_col = left_col + box_width
        top_row = center_y - (box_height / 2.0)
        bot_row = top_row + box_height

        top_left = (int(round(left_col)), int(round(top_row)))
        bot_right = (int(round(right_col)), int(round(bot_row)))
        cv2.rectangle(img, top_left, bot_right, (0, 255, 0), 2)

    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_gt_file():
    """Converts the gt.txt file into a dict of yolo lines."""
    gt_path = GTSDB_PATH / "gt.txt"
    with open(gt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return process_gtsdb_annotations(lines)

def save_yolo_lines():
    """Saves text files in yolo format using the gt.txt file."""
    dataset = process_gt_file()
    target_path = GTSDB_PATH / "dataset" / "labels"
    target_path.mkdir(parents=True, exist_ok=True)
    for image in dataset:
        image_id = image.split(".")[0]
        with open(target_path / f"{image_id}.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(dataset[image]) + "\n")
    print(f"Successfully saved all lines to {target_path}.")

def main():
    save_yolo_lines()

if __name__ == "__main__":
    main()