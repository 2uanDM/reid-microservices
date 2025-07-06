import asyncio
import base64
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import avro.schema
import cv2
import numpy as np
import websockets
from avro.io import BinaryDecoder, DatumReader
from dotenv import load_dotenv
from websockets.server import WebSocketServerProtocol

from kafka import KafkaConsumer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.coordinator.assignors.roundrobin import RoundRobinPartitionAssignor
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

        # Max messages to process in one poll
        self.max_poll_records = int(os.getenv("MAX_POLL_RECORDS", 100))

        # Load output schema
        with open("src/configs/output.avsc", "r") as f:
            self.output_schema = avro.schema.parse(f.read())
        self.reader = DatumReader(self.output_schema)

        # Store for latest frames from each device
        self.device_frames: Dict[str, dict] = {
            "edge_device_1": {},
            "edge_device_2": {},
            "edge_device_3": {},
        }

        # WebSocket connections
        self.websocket_connections: List[WebSocketServerProtocol] = []

        # Threading control
        self.running = False
        self.executor = ThreadPoolExecutor(
            max_workers=10
        )  # Decode and compress in parallel

        # Task references for better control
        self.websocket_task = None
        self.kafka_task = None

    def init_kafka_consumer(self):
        """Initialize Kafka consumer for reid_output topic"""
        try:
            self.consumer = KafkaConsumer(
                self.output_topic_name,
                bootstrap_servers=self.kafka_bootstrap_servers,
                auto_offset_reset="latest",  # Only get new messages
                enable_auto_commit=True,
                group_id=self.consumer_group,
                partition_assignment_strategy=[RoundRobinPartitionAssignor],
                value_deserializer=self._decode_message,
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_interval_ms=300000,
                fetch_min_bytes=512 * 1024,  # Reduced to get messages faster
                fetch_max_bytes=10 * 1024 * 1024,  # Increased for higher throughput
                max_partition_fetch_bytes=10 * 1024 * 1024,
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
        """Handle WebSocket connections for streaming with high priority"""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        self.websocket_connections.append(websocket)

        try:
            # Set high priority for this coroutine
            current_task = asyncio.current_task()
            if current_task and hasattr(current_task, "set_name"):
                current_task.set_name(f"websocket-handler-{websocket.remote_address}")

            # Send initial device list with priority
            device_list = list(self.device_frames.keys())
            await self._send_with_priority(
                websocket, {"type": "device_list", "devices": device_list}
            )

            # Handle incoming messages from client
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data["type"] == "subscribe_device":
                        device_id = data["device_id"]
                        # Handle subscription with priority
                        await self._send_device_stream_priority(websocket, device_id)
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

    async def _send_with_priority(self, websocket: WebSocketServerProtocol, data: dict):
        """Send data to WebSocket with priority handling"""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Priority send failed: {e}")

    async def _send_device_stream_priority(
        self, websocket: WebSocketServerProtocol, device_id: str
    ):
        """Send video stream for a specific device with priority"""
        if device_id not in self.device_frames:
            await self._send_with_priority(
                websocket, {"type": "error", "message": f"Device {device_id} not found"}
            )
            return

        # Send latest frame with priority
        frame_data = self.device_frames[device_id]
        await self._send_with_priority(
            websocket,
            {
                "type": "frame",
                "device_id": device_id,
                "frame_number": frame_data["frame_number"],
                "tracked_persons": frame_data["tracked_persons"],
                "created_at": frame_data["created_at"],
                "image_base64": frame_data["image_base64"],
            },
        )

    async def broadcast_frame_update(self, device_id: str):
        """Broadcast frame update to all connected WebSocket clients with high priority"""
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

        # Send to all connected clients with higher concurrency for WebSocket priority
        disconnected_clients = []

        # Increased semaphore for WebSocket priority
        async with asyncio.Semaphore(10):  # Increased for better WebSocket performance
            send_tasks = []
            for websocket in self.websocket_connections:
                # Create priority send tasks
                task = asyncio.create_task(
                    self._priority_send_to_client(websocket, message),
                    name=f"ws-send-{websocket.remote_address}",
                )
                send_tasks.append(task)

            # Wait for all sends to complete
            if send_tasks:
                results = await asyncio.gather(*send_tasks, return_exceptions=True)

                # Check for failed connections
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        disconnected_clients.append(self.websocket_connections[i])

        # Remove disconnected clients
        for client in disconnected_clients:
            if client in self.websocket_connections:
                self.websocket_connections.remove(client)

    async def _priority_send_to_client(
        self, websocket: WebSocketServerProtocol, message: str
    ):
        """Send message to a single WebSocket client with priority handling"""
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            raise  # Re-raise to mark for disconnection
        except Exception as e:
            logger.error(
                f"Error sending to WebSocket client {websocket.remote_address}: {e}"
            )
            raise  # Re-raise to mark for disconnection

    def preprocess_single_msg(
        self, idx: int, msg: ConsumerRecord
    ) -> tuple[int, str, dict]:
        """Preprocess a single message. Return the frame data and device id"""
        processed_frame = msg.value
        device_id = processed_frame["device_id"]

        # Convert image bytes back to numpy array and then to base64
        image_bytes = processed_frame["image_data"]
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to base64 for web transmission
        _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 40])
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Store frame data
        frame_data = {
            "frame_number": processed_frame["frame_number"],
            "tracked_persons": processed_frame["tracked_persons"],
            "created_at": processed_frame["created_at"],
            "image_base64": image_base64,
        }

        return idx, device_id, frame_data

    async def kafka_consumer_loop(self):
        """Main loop to consume from Kafka with WebSocket priority"""
        logger.info("Starting Kafka consumer loop with WebSocket priority...")

        while self.running:
            try:
                # Yield control to allow WebSocket operations to run first
                await asyncio.sleep(0.1)

                messages = self.consumer.poll(
                    timeout_ms=1000,  # Reduced timeout for better responsiveness
                    max_records=self.max_poll_records,
                )

                if messages:
                    message_count = sum(len(msgs) for msgs in messages.values())
                    logger.info(
                        f"Processing {message_count} messages from {len(messages)} partitions"
                    )

                    # Process messages in smaller batches to allow WebSocket priority
                    batch_size = self.max_poll_records
                    processed_count = 0

                    futures = []
                    output = {}  # {device_id: [idx, frame_data]}

                    # Submit all messages of all partitions to the executor
                    start_time = time.time()
                    for _, partition_data in messages.items():
                        for idx, msg in enumerate(partition_data):
                            futures.append(
                                self.executor.submit(
                                    self.preprocess_single_msg, idx, msg
                                )
                            )

                    # Wait for all tasks to complete
                    for future in as_completed(futures):
                        idx, device_id, frame_data = future.result()
                        if device_id not in output:
                            output[device_id] = []
                        output[device_id].append((idx, frame_data))

                    end_time = time.time()
                    print(f"Time taken: {end_time - start_time} seconds")

                    for device_id, frame_data in output.items():
                        frame_data.sort(key=lambda x: x[0])
                        for idx, frame_data in frame_data:
                            self.device_frames[device_id] = frame_data
                            await self.broadcast_frame_update(device_id)
                            processed_count += 1
                            # Yield control every batch_size messages to prioritize WebSocket operations
                            if processed_count % batch_size == 0:
                                # Allow WebSocket operations to run
                                await asyncio.sleep(0.1)

                if not messages:
                    # Longer sleep when no messages, but still yield frequently for WebSocket priority
                    await asyncio.sleep(0.1)

            except Exception:
                logger.error("Error in Kafka consumer loop", exc_info=True)
                await asyncio.sleep(1)

    async def start_websocket_server(self):
        """Start the WebSocket server with high priority"""
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
            compression=None,  # Disable compression for lower latency
        )

        await start_server
        logger.info("WebSocket server started successfully")

    async def run(self):
        """Main run method with WebSocket priority"""
        if not self.init_kafka_consumer():
            return

        self.running = True

        try:
            # Create tasks with priority settings
            self.websocket_task = asyncio.create_task(
                self.start_websocket_server(), name="websocket-server-high-priority"
            )

            # Small delay to ensure WebSocket server starts first
            await asyncio.sleep(0.1)

            self.kafka_task = asyncio.create_task(
                self.kafka_consumer_loop(), name="kafka-consumer-low-priority"
            )

            # Use gather with return_exceptions to handle failures gracefully
            # WebSocket task is listed first for priority
            await asyncio.gather(
                self.websocket_task,
                self.kafka_task,
                return_exceptions=True,
            )

        except Exception as e:
            logger.error(f"Error in main run loop: {e}", exc_info=True)
        finally:
            await self._cleanup_tasks()

    async def _cleanup_tasks(self):
        """Clean up tasks with priority for WebSocket operations"""
        logger.info("Starting cleanup of tasks...")

        # Cancel Kafka task first to stop consuming messages
        if self.kafka_task and not self.kafka_task.done():
            self.kafka_task.cancel()
            try:
                await self.kafka_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connections gracefully
        if self.websocket_connections:
            close_tasks = []
            for websocket in self.websocket_connections.copy():
                close_tasks.append(
                    asyncio.create_task(
                        websocket.close(), name=f"close-ws-{websocket.remote_address}"
                    )
                )

            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            self.websocket_connections.clear()

        # Cancel WebSocket server task last
        if self.websocket_task and not self.websocket_task.done():
            logger.info("Cancelling WebSocket server task...")
            self.websocket_task.cancel()
            try:
                await self.websocket_task
            except asyncio.CancelledError:
                logger.info("WebSocket server task cancelled successfully")

    def stop(self):
        """Stop the streaming service with priority cleanup"""
        logger.info("Stopping streaming service...")
        self.running = False

        if hasattr(self, "consumer"):
            try:
                self.consumer.close()
                logger.info("Kafka consumer closed")
            except Exception as e:
                logger.error(f"Error closing Kafka consumer: {e}")

        logger.info("Streaming service stopped")
