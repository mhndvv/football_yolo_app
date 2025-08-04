import cv2
from ultralytics import YOLO
import os

model_path = "models/yolov8n.pt"  # Replace with your trained model if available
model = YOLO(model_path)

video_path = "media/test_video.mp4"  # Your test video
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_video_path = os.path.join(output_folder, "result_video.mp4")

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection per frame
    results = model(frame)
    annotated_frame = results[0].plot()  # Annotate detections

    out.write(annotated_frame)  # Save to output video

cap.release()
out.release()
print(f"✅ Video detection complete! Saved to: {output_video_path}")
