# detection/model.py
from ultralytics import YOLO

def load_model():
    """Load YOLO model and return it"""
    model = YOLO("models/yolo11n.pt")
    return model

if __name__ == "__main__":
    model = load_model()
    print("✅ YOLO model loaded successfully")