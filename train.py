from ultralytics import YOLO
from pathlib import Path
import torch

def setup_model(weights_path="yolo11n.pt"):
    model = YOLO(weights_path, task="detect")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model

def train_model(model, data_yaml_path, epochs=60,imgsz=640):
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        save=True,
        save_txt=True,
    )
    return results

def predict_sample(model, image_path):
    results = model.predict(image_path, save=True, save_txt=True)
    return results
