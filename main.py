from slam import SLAMSystem
from annotations import parse_annotation, write_label
from logger import log_safety_event
from train import setup_model, train_model, predict_sample
from visualization import draw_predictions, analyze_predictions, plot_slam_trajectory
from environment import simulate_slam_pose, apply_random_weather_effects

from pathlib import Path
from PIL import Image
import torch
import yaml
import random
import numpy as np
from tqdm import tqdm

# Initialize SLAM system
slam = SLAMSystem()
slam.start_mapping()
slam.start_tracking()
print(f"[INFO] SLAM state: {slam.state}")

# Load class mapping
base_data_path = "carla-object-detection-dataset"
base_output = "working"
image_data = Path(base_data_path, "images/train")
train_imgs = Path(base_data_path, "images/train")
train_annotations = Path(base_data_path, "labels/train")

yolo_base = Path(base_output, "yolo_data")
yolo_base.mkdir(parents=True, exist_ok=True)
(yolo_base / "images/train").mkdir(parents=True, exist_ok=True)
(yolo_base / "images/val").mkdir(parents=True, exist_ok=True)
(yolo_base / "labels/train").mkdir(parents=True, exist_ok=True)
(yolo_base / "labels/val").mkdir(parents=True, exist_ok=True)

with open(Path(base_data_path, "labels.txt"), "r") as f:
    classes = f.read().splitlines()
class_mapping = {cls: idx for idx, cls in enumerate(classes)}

# Convert dataset to YOLO format
train_frac = 0.8
images = list(train_imgs.glob("*"))
for img in tqdm(images):
    split = "train" if random.random() < train_frac else "val"
    annotation = train_annotations / f"{img.stem}.xml"
    try:
        parsed = parse_annotation(annotation, class_mapping)
    except Exception:
        log_safety_event(f"Failed to parse annotation: {img.stem}")
        continue

    write_label(parsed, yolo_base / f"labels/{split}/{img.stem}.txt")
    image = Image.open(img).convert("RGB")
    image.save(yolo_base / f"images/{split}/{img.stem}.jpg", "JPEG")

# Create data.yaml for YOLO
yolo_metadata = {
    "path": str(Path(base_output, "yolo_data")),
    "train": str((yolo_base / "images/train").resolve()),
    "val": str((yolo_base / "images/val").resolve()),
    "names": classes,
    "nc": len(classes)
}
with open(Path(base_output, "data.yaml"), 'w') as f:
    yaml.safe_dump(yolo_metadata, f)

# Train YOLO model
model = setup_model()
results = train_model(model, Path(base_output, "data.yaml"))

# Predict with a test image
test_image_path = yolo_base / "images/train/Town01_002100.jpg"
try:
    test_image = Image.open(test_image_path)
    fogged_image = apply_random_weather_effects(test_image)
    fogged_image.save("fogged_sample.jpg")

    results_fin = predict_sample(model, "fogged_sample.jpg")
    output_image = draw_predictions(test_image, results_fin)
    output_image.show()

    most_common, count = analyze_predictions(results_fin, model)
    print(f"Most common class: {most_common} ({count} instances)")
except FileNotFoundError:
    slam.lose_tracking()
    log_safety_event(f"Image not found: {test_image_path}")
    print(f"[WARNING] SLAM lost tracking: {slam.state}")
    slam.recover_tracking()
    print(f"[INFO] SLAM recovered tracking: {slam.state}")

# Simulate SLAM trajectory
poses = np.array([simulate_slam_pose(i) for i in range(60)])
plot_slam_trajectory(poses)

