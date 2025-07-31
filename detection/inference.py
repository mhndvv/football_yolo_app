# detection/inference.py
from PIL import Image
from detection.model import load_model

def run_inference(image_path: str):
    """
    Run YOLO detection on a given image.
    
    Parameters
    ----------
    image_path : str
        Path to the image for detection.
    """
    model = load_model()
    img = Image.open(image_path)
    results = model(img)
    results[0].show()  # Show annotated image
    print("✅ Detection completed")

if __name__ == "__main__":
    run_inference("uploads/test.jpg")  # Change path to your test image
