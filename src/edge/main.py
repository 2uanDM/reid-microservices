import argparse  # Read arguments from command line
import json
import time
import uuid

import cv2
import numpy as np
from dotenv import load_dotenv
from kafka import KafkaProducer

from src.yolo import YoloModel

load_dotenv()


class EdgeDeviceRunner:
    def __init__(
        self,
        device_id: str,
        source_url: str,
        reid_topic: str,
        model_path: str = "weights/best.pt",
        kafka_server_uri: str = "localhost:29092",
    ):
        self.device_id = device_id
        self.source_url = source_url
        self.model_path = model_path
        self.reid_topic = reid_topic
        self.kafka_server_uri = kafka_server_uri

        # Init Kafka producer
        self.init_kafka()

        # Init Yolo model
        self.model = YoloModel(model_path=model_path, ensure_onnx=True)

    def init_kafka(self):
        # For binary serialization
        self.producer = KafkaProducer(
            client_id=self.device_id,
            bootstrap_servers=self.kafka_server_uri,
            acks="all",
            value_serializer=lambda x: x,
        )

    def produce(
        self,
        frame: np.ndarray | None = None,
        payload: dict | None = None,
    ):
        if payload is None:
            payload = {}

        # Convert metadata to bytes
        metadata_bytes = json.dumps(payload).encode("utf-8")

        if frame is not None and frame.size > 0:
            # Compress image to JPEG bytes
            _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_bytes = img_bytes.tobytes()

            # Format: [4-byte metadata length][metadata][image bytes]
            message = (
                len(metadata_bytes).to_bytes(4, byteorder="big")
                + metadata_bytes
                + img_bytes
            )

            # Send binary message
            self.producer.send(self.reid_topic, value=message)
        else:
            # Send metadata only
            message = len(metadata_bytes).to_bytes(4, byteorder="big") + metadata_bytes
            self.producer.send(self.reid_topic, value=message)

    def read_source(self) -> cv2.VideoCapture:
        if self.source_url.startswith("rtsp://") or self.source_url.startswith(
            "http://"
        ):
            # TODO: Handle RTSP stream
            raise NotImplementedError("RTSP stream handling not implemented")
        elif self.source_url.split(".")[-1].lower() in {
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm",
        }:
            return cv2.VideoCapture(self.source_url)

    def run(self):
        print("------ Information of Edge Device ------")
        print(f"Device ID: {self.device_id}")
        print(f"Source: {self.source_url}")
        print(f"Kafka server URI: {self.kafka_server_uri}")
        print(f"ReID topic: {self.reid_topic}")

        source = self.read_source()

        if isinstance(source, cv2.VideoCapture):
            cap = source
            total_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = source.get(cv2.CAP_PROP_FPS)

            print("------ Video Source Information ------")
            print(f"Total frames: {total_frames}")
            print(f"FPS: {fps}")

            # Calculate length in minutes
            length_seconds = total_frames / fps if fps > 0 else 0
            length_minutes = length_seconds / 60
            print(f"Length: {length_minutes:.2f} minutes")

            print("------ Processing Video ------")
            frame_count = 0
            start_time = time.perf_counter()
            last_fps_update = start_time
            fps_counter = 0
            processing_fps = 0

            while True:
                frame_count += 1
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break

                # Get the detection result
                detections = self.model.infer(image=frame)

                # Prepare metadata
                metadata = {
                    "device_id": self.device_id,
                    "frame_number": frame_count,
                    "result": detections,
                    "created_at": time.time_ns(),
                }

                # Produce the message
                self.produce(frame=frame, payload=metadata)

                # Update FPS calculation (every second)
                fps_counter += 1
                current_time = time.perf_counter()
                elapsed = current_time - last_fps_update
                if elapsed >= 1.0:
                    processing_fps = fps_counter / elapsed
                    fps_counter = 0
                    last_fps_update = current_time

                # Print progress every 100 frames
                if frame_count % 100 == 0:
                    progress = (
                        (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    )
                    elapsed_total = current_time - start_time
                    eta = (
                        (elapsed_total / frame_count) * (total_frames - frame_count)
                        if frame_count > 0
                        else 0
                    )
                    print(
                        f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | FPS: {processing_fps:.1f} | ETA: {eta:.1f}s"
                    )

        cap.release()
        # Display final stats
        total_time = time.perf_counter() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Completed in {total_time:.2f}s | Avg FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="video.mp4", help="Source of the video"
    )
    parser.add_argument(
        "--device_id",
        type=str,
        default=f"edge_device_{uuid.uuid4().hex[:8]}",
        help="Device ID",
    )
    parser.add_argument(
        "--reid_topic",
        type=str,
        default="reid_topic",
        help="Kafka topic for re-identification",
    )
    parser.add_argument(
        "--kafka_server_uri",
        type=str,
        default="localhost:29092",
        help="Kafka server URI",
    )
    args = parser.parse_args()

    runner = EdgeDeviceRunner(
        device_id=args.device_id,
        source_url=args.source,
        reid_topic=args.reid_topic,
        kafka_server_uri=args.kafka_server_uri,
    )
    runner.run()
