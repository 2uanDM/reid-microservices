import asyncio
import base64
import io
import json
import os
from collections import defaultdict, deque
from typing import Dict, List

import avro.schema
import cv2
import numpy as np
import websockets
from avro.io import BinaryDecoder, DatumReader
from dotenv import load_dotenv
from websockets.server import WebSocketServerProtocol

from kafka import KafkaConsumer
from src.utils import Logger

load_dotenv()

logger = Logger(__name__)


class StreamingService:
    def __init__(self):
        # Kafka configuration
        self.kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        self.output_topic_name = os.getenv("OUTPUT_TOPIC_NAME", "reid_output")
        self.consumer_group = os.getenv(
            "STREAMING_CONSUMER_GROUP", "reid_consumer_group"
        )

        # WebSocket configuration
        self.websocket_host = os.getenv("WEBSOCKET_HOST", "localhost")
        self.websocket_port = int(os.getenv("WEBSOCKET_PORT", 8765))

        # Load output schema
        with open("src/configs/output.avsc", "r") as f:
            self.output_schema = avro.schema.parse(f.read())
        self.reader = DatumReader(self.output_schema)

        # Store for latest frames from each device
        self.device_frames: Dict[str, dict] = {}
        self.frame_buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )  # Buffer last 30 frames per device

        # WebSocket connections
        self.websocket_connections: List[WebSocketServerProtocol] = []

        # Threading control
        self.running = False

    def init_kafka_consumer(self):
        """Initialize Kafka consumer for reid_output topic"""
        try:
            self.consumer = KafkaConsumer(
                self.output_topic_name,
                bootstrap_servers=self.kafka_bootstrap_servers,
                auto_offset_reset="latest",  # Only get new messages
                enable_auto_commit=True,
                group_id=self.consumer_group,
                value_deserializer=self._decode_message,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_interval_ms=300000,
                fetch_min_bytes=512 * 1024,  # Reduced to get messages faster
                fetch_max_bytes=100 * 1024 * 1024,  # Increased for higher throughput
                max_partition_fetch_bytes=100 * 1024 * 1024,
            )

            logger.info("Kafka Consumer for streaming initialized successfully")
            logger.info(f"Consumer group: {self.consumer_group}")

            # Log partition assignment for debugging
            partitions = self.consumer.partitions_for_topic(self.output_topic_name)
            if partitions:
                logger.info(
                    f"Available partitions for topic {self.output_topic_name}: {sorted(partitions)}"
                )
                logger.info("This consumer will receive messages from ALL partitions")

            return True
        except Exception:
            logger.error("Error initializing Kafka Consumer", exc_info=True)
            return False

    def _decode_message(self, message_value: bytes) -> dict:
        """Decode Avro message from reid_output topic"""
        decoder = BinaryDecoder(io.BytesIO(message_value))
        return self.reader.read(decoder)

    async def websocket_handler(self, websocket: WebSocketServerProtocol):
        """Handle WebSocket connections for streaming"""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        self.websocket_connections.append(websocket)

        try:
            # Send initial device list
            device_list = list(self.device_frames.keys())
            await websocket.send(
                json.dumps({"type": "device_list", "devices": device_list})
            )

            # Handle incoming messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data["type"] == "subscribe_device":
                        device_id = data["device_id"]
                        await self._send_device_stream(websocket, device_id)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON received from WebSocket client")
                except Exception:
                    logger.error("Error handling WebSocket message", exc_info=True)

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed: {websocket.remote_address}")
        except Exception:
            logger.error("WebSocket error", exc_info=True)
        finally:
            if websocket in self.websocket_connections:
                self.websocket_connections.remove(websocket)

    async def _send_device_stream(
        self, websocket: WebSocketServerProtocol, device_id: str
    ):
        """Send video stream for a specific device"""
        if device_id not in self.device_frames:
            await websocket.send(
                json.dumps(
                    {"type": "error", "message": f"Device {device_id} not found"}
                )
            )
            return

        # Send latest frame
        frame_data = self.device_frames[device_id]
        await websocket.send(
            json.dumps(
                {
                    "type": "frame",
                    "device_id": device_id,
                    "frame_number": frame_data["frame_number"],
                    "tracked_persons": frame_data["tracked_persons"],
                    "created_at": frame_data["created_at"],
                    "image_base64": frame_data["image_base64"],
                }
            )
        )

    async def broadcast_frame_update(self, device_id: str):
        """Broadcast frame update to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        frame_data = self.device_frames[device_id]
        message = json.dumps(
            {
                "type": "frame_update",
                "device_id": device_id,
                "frame_number": frame_data["frame_number"],
                "tracked_persons": frame_data["tracked_persons"],
                "created_at": frame_data["created_at"],
                "image_base64": frame_data["image_base64"],
            }
        )

        # Send to all connected clients
        disconnected_clients = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.append(websocket)
            except Exception:
                logger.error("Error sending to WebSocket client", exc_info=True)
                disconnected_clients.append(websocket)

        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_connections:
                self.websocket_connections.remove(client)

    async def kafka_consumer_loop(self):
        """Main loop to consume from Kafka and update frame store"""
        logger.info("Starting Kafka consumer loop...")

        while self.running:
            try:
                messages = self.consumer.poll(timeout_ms=1000, max_records=30)

                if messages:
                    message_count = sum(len(msgs) for msgs in messages.values())
                    logger.info(
                        f"Processing {message_count} messages from {len(messages)} partitions"
                    )

                for topic_partition, msgs in messages.items():
                    for msg in msgs:
                        processed_frame = msg.value
                        device_id = processed_frame["device_id"]

                        # Convert image bytes back to numpy array and then to base64
                        image_bytes = processed_frame["image_data"]
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # Convert to base64 for web transmission
                        _, buffer = cv2.imencode(
                            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )

                        image_base64 = base64.b64encode(buffer).decode("utf-8")

                        # Store frame data
                        frame_data = {
                            "frame_number": processed_frame["frame_number"],
                            "tracked_persons": processed_frame["tracked_persons"],
                            "created_at": processed_frame["created_at"],
                            "image_base64": image_base64,
                        }

                        self.device_frames[device_id] = frame_data
                        self.frame_buffers[device_id].append(frame_data)

                        # Broadcast update to WebSocket clients
                        await self.broadcast_frame_update(device_id)

                        logger.info(
                            f"Processed frame {processed_frame['frame_number']} from device {device_id}"
                        )

                if not messages:
                    await asyncio.sleep(0.05)

            except Exception:
                logger.error("Error in Kafka consumer loop", exc_info=True)
                await asyncio.sleep(1)

    async def start_websocket_server(self):
        """Start the WebSocket server"""
        logger.info(
            f"Starting WebSocket server on {self.websocket_host}:{self.websocket_port}"
        )

        start_server = websockets.serve(
            self.websocket_handler,
            self.websocket_host,
            self.websocket_port,
            ping_interval=20,
            ping_timeout=10,
            max_size=10 * 1024 * 1024,  # 10MB max message size
        )

        await start_server
        logger.info("WebSocket server started successfully")

    async def run(self):
        """Main run method"""
        if not self.init_kafka_consumer():
            return

        self.running = True

        # Start WebSocket server and Kafka consumer concurrently
        await asyncio.gather(
            self.start_websocket_server(),
            self.kafka_consumer_loop(),
            return_exceptions=True,
        )

    def stop(self):
        """Stop the streaming service"""
        self.running = False
        if hasattr(self, "consumer"):
            self.consumer.close()
        logger.info("Streaming service stopped")
