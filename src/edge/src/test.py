import os
import sys

sys.path.append(os.getcwd())
import cv2

from edge.src.yolo import YoloModel

model = YoloModel(model_path="best.onnx", ensure_onnx=True)

video_path = "non overlap.mp4"

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()

    print(f"dimensions: {frame.shape}")

    if not ret:
        break

    detections = model.infer(frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
