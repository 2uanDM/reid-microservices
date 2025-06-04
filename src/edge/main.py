import argparse
import io as python_io
import os
import time
import uuid

import cv2
import numpy as np
from avro import io, schema
from avro.io import DatumWriter
from dotenv import load_dotenv
from yolo import YoloModel

from kafka import KafkaProducer

load_dotenv()


class EdgeDeviceRunner:
    def __init__(
        self,
        device_id: str,  # Each edge device (camera) has a unique ID
        source_url: str,  # Source of the demo video or RTSP stream
        reid_topic: str,  # Kafka topic for re-identification
        kafka_bootstrap_servers: str,  # Kafka server URI
        ensure_onnx: bool = True,  # Whether to ensure the weights are in ONNX format
        ensure_openvino: bool = True,  # Whether to ensure the weights are in OpenVINO format
        model_path: str = "weights/best.onnx",  # Path to the YOLO model
    ):
        self.device_id = device_id
        self.source_url = source_url
        self.model_path = model_path
        self.reid_topic = reid_topic
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.printed_image_size = False

        # Load Avro Schema for Kafka message serialization
        self.avro_schema = self.load_schema()

        # Init Kafka producer
        self.init_kafka()

        # Init Yolo model
        self.model = YoloModel(
            model_path=model_path,
            ensure_onnx=ensure_onnx,
            ensure_openvino=ensure_openvino,
        )

    def load_schema(self) -> schema.Schema:
        possible_paths = [
            "schema.avsc",  # Current directory
            "src/edge/schema.avsc",  # From project root
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "schema.avsc"
            ),  # Same directory as this file
        ]

        for path in possible_paths:
            try:
                with open(path, "r") as f:
                    schema_str = f.read()
                    print(f"Successfully loaded schema from {path}")
                    try:
                        return schema.parse(schema_str)
                    except Exception as e:
                        print(f"Error parsing schema: {e}")
                        print(f"Schema content: {schema_str}")
                        raise
            except FileNotFoundError:
                continue

        # If we get here, schema wasn't found
        raise FileNotFoundError(
            f"Could not find schema.avsc in any of: {possible_paths}"
        )

    def init_kafka(self):
        self.producer = KafkaProducer(
            client_id=self.device_id,
            bootstrap_servers=self.kafka_bootstrap_servers,
            key_serializer=lambda x: x.encode("utf-8"),  # Key here is the device ID
            value_serializer=self.serialize_message,  # Serialize the message using Avro
            linger_ms=10,
            batch_size=16384,
            acks=0,
            max_in_flight_requests_per_connection=5,
        )

    def serialize_message(self, message: dict) -> bytes:
        try:
            writer = DatumWriter(self.avro_schema)
            bytes_writer = python_io.BytesIO()
            encoder = io.BinaryEncoder(bytes_writer)
            writer.write(message, encoder)
            return bytes_writer.getvalue()
        except Exception as e:
            print(f"Error serializing message: {e}")
            print(f"Message: {message}")
            raise e

    def produce(
        self,
        frame: np.ndarray | None = None,  # Input frame from the video source
        payload: dict | None = None,  # Metadata about the frame
    ):
        if payload is None:
            payload = {}

        # Format detection results to match our schema
        results = []
        for detection in payload.get("result", []):
            # Ensure detections match our schema format
            formatted_detection = {
                "bbox": detection.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                "confidence": float(detection.get("confidence", 0.0)),
                "class_id": int(detection.get("class_id", 0)),
            }
            results.append(formatted_detection)

        # Prepare the message according to Avro schema
        message = {
            "device_id": self.device_id,
            "frame_number": int(payload.get("frame_number", 0)),
            "result": results,
            "created_at": int(payload.get("created_at", time.time_ns())),
            "image_data": None,
        }

        if frame is not None and frame.size > 0:
            # Compress image to JPEG bytes
            _, img_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            message["image_data"] = img_bytes.tobytes()
            if not self.printed_image_size:
                print(f"Image size: {len(img_bytes) / 1024} KB")
                self.printed_image_size = True

        # Send message using Avro serialization
        self.producer.send(self.reid_topic, value=message, key=self.device_id)

    def read_source(self) -> cv2.VideoCapture:
        if self.source_url.startswith("rtsp://") or self.source_url.startswith(
            "http://"
        ):
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
        print(f"Kafka server URI: {self.kafka_bootstrap_servers}")
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

        # Shutdown Kafka producer
        self.producer.close()


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
        default="reid_input",
        help="Kafka topic for re-identification",
    )
    parser.add_argument(
        "--kafka_bootstrap_servers",
        type=str,
        default="localhost:9092",
        help="Kafka bootstrap servers",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/best.onnx",
        help="Path to the YOLO model",
    )
    parser.add_argument(
        "--ensure_onnx",
        action="store_true",
        help="Ensure the model is in ONNX format",
    )
    parser.add_argument(
        "--ensure_openvino",
        action="store_true",
        help="Ensure the model is in OpenVINO format",
    )
    args = parser.parse_args()

    runner = EdgeDeviceRunner(
        device_id=args.device_id,
        source_url=args.source,
        reid_topic=args.reid_topic,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        ensure_onnx=args.ensure_onnx,
        ensure_openvino=args.ensure_openvino,
        model_path=args.model_path,
    )
    runner.run()
