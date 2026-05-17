from ultralytics import YOLO
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DATA_YAML = str(TASK_DIR / "vehicle.yaml")
SAVE_DIR = str(TASK_DIR / "runs")

def train_yolov8():
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        imgsz=640,
        batch=16,
        project=SAVE_DIR,
        name="yolov8n_vehicle",
        val=True,
    )
    print("Training completed!")
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return model, metrics

if __name__ == "__main__":
    train_yolov8()
