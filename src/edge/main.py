import os
import cv2
import json
import time
import argparse #Read arguments from command line

from kafka import KafkaProducer
from ultralytics import YOLO

#Argument parser lay video path
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="video.mp4", help='Path to the input video')
args = parser.parse_args()
VIDEO_PATH = args.path

KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")
DEVICE_NAME = os.getenv("edge-1")

CONF_THRESHOLD = 0.7

model = YOLO("best.pt")

#Create Kafka producer
producer = KafkaProducer(bootstrap_servers = KAFKA_SERVER,
                         value_serializer = lambda v: json.dumps(v).encode('utf-8')) #encode message -> json

#load video input
print(f"Reading: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open {VIDEO_PATH}")

frame_count = 0

#Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    resized =  cv2.resize(frame, (640, 640))
    results = model(resized, verbose = False)[0]

    #filter with threshold
    detections = []
    for box in results.boxes:
        if box.conf >= CONF_THRESHOLD:
            detections.append({
                "bbox": box.xyxy[0].tolist(), #bounding box [x1, y1, x2, y2]
                "conf": float(box.conf),
                "cls": int(box.cls)
            })
    #Create message to Kafka
    message = {
        "device": DEVICE_NAME,
        "frame": frame_count, #index
        "detections": detections
    }
    #send message to Kafka
    producer.send("infer result", message)

    frame_count+=1

    time.sleep(0.05)


cap.release()
print("Completed")
