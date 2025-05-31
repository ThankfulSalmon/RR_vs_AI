from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def draw_predictions(image, results):
    boxes = results[0].boxes.xyxy
    labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls]
    image_tensor = to_tensor(results[0].orig_img)
    image_with_boxes = draw_bounding_boxes(
        image=image_tensor,
        boxes=boxes,
        labels=labels,
        width=6,
    )
    return to_pil_image(image_with_boxes)

def analyze_predictions(results, model):
    object_counts = Counter([model.names[int(cls)] for cls in results[0].boxes.cls])
    most_common_class, count_of_class = object_counts.most_common(n=1)[0]
    return most_common_class, count_of_class

def plot_slam_trajectory(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], label='Simulated SLAM Trajectory')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("Simulated SLAM Pose Trajectory")
    plt.show()

