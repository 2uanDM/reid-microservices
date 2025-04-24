import os
import cv2
import json
import time
import argparse #Read arguments from command line

from kafka import KafkaProducer
from ultralytics import YOLO

class VideoInferProcess:
    def __init__(self, video_path, model_path = "best.pt", conf_threshold = 0.7, kafka_server = "localhost:29092", topic="infer result", skip_frames = 2):
        self.video_path = video_path
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.topic = topic
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.device_name = os.getenv("DEVICE_NAME", "edge-1")

        #set up Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers = kafka_server,
            value_serializer = lambda v: json.dumps(v).encode('utf-8')
        )

    def frame_preprocess(self, frame):
        return cv2.resize(frame, (640, 640))
    
    def run_inference(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if box.conf >= self.conf_threshold:
                detections.append({
                    "bbox": box.xyxy[0].tolist(), #bounding box [x1, y1, x2, y2]
                    "conf": float(box.conf),
                    "cls": int(box.cls)
                })
        return detections
    
    def send_to_kafka(self, detections):
        #send detection results to Kafka
        message = {
            "device": self.device_name,
            "frame": self.frame_count, #index
            "detections": detections
        }
        self.producer.send(self.topic, message)

    def video_process(self):
        #load input video
        print(f"Reading video from: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open {self.video_path}")
        #process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_count % self.skip_frames == 0:
                preprocessed = self.frame_preprocess(frame)
                detections = self.run_inference(preprocessed)
                self.send_to_kafka(detections)
            self.frame_count+=1
            time.sleep(0.05)

        cap.release()
        print("Completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="video.mp4", help='Path to the input video')
    args = parser.parse_args()

    processor = VideoInferProcess(video_path=args.path)
    processor.video_process()

     



