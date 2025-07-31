import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Create uploads folder if it does not exist
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.title(":soccer: Football Object Detection - YOLO Streamlit App")
st.write("Upload an image to detect players on the field using a custom YOLO model.")

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show image in Streamlit
    st.image(file_path, caption="Uploaded Image", use_container_width=True)

@st.cache_resource
def load_model():
    """Load YOLO model only once"""
    model = YOLO("models/yolo11n.pt")
    return model

model = load_model()
st.success(":white_check_mark: YOLO model has been loaded successfully.")

# Detection parameters
conf = st.number_input("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou = st.number_input("Intersection Over Union (IoU)", 0.0, 1.0, 0.7, 0.01)
max_det = st.number_input("Maximum Detections", 1, 3000, 100, 1)
imgsz = st.number_input("Image Size", 64, 2048, 1024, 32)
augment = st.checkbox("Enable Augmentation", value=False)
agnostic_nms = st.checkbox("Agnostic NMS", value=True)

if uploaded_file and st.button(":dart: Detect Players"):
    img = Image.open(file_path)

    results = model(
        img,
        conf=conf,
        iou=iou,
        augment=augment,
        agnostic_nms=agnostic_nms,
        max_det=max_det,
        imgsz=imgsz,
        classes=None
    )

    result = results[0]

    # Count players (class 0 = person)
    player_count = 0
    if result.boxes is not None and result.boxes.cls is not None:
        for cls in result.boxes.cls:
            if int(cls) == 0:
                player_count += 1

    st.success(f":busts_in_silhouette: Number of players detected: {player_count}")

    # Show detection image
    model_output = result.plot()
    st.image(model_output, caption="Detected Players", use_container_width=True)
