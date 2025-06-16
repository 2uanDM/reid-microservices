import asyncio
import io
import os
import traceback
import uuid
from typing import Any, Dict, List, Tuple

import avro.schema
import cv2
import numpy as np
from avro.io import BinaryDecoder, DatumReader
from dotenv import load_dotenv
from rich.console import Console

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from src.apis import ModelServiceClient
from src.schemas import EdgeDeviceMessage, PersonMetadata
from src.trackers import BYTETracker, Namespace
from src.utils.ops import crop_image, draw_bbox, xyxy2xywh

console = Console()


load_dotenv()


class ReIdConsumer:
    def __init__(self, reid_model: str = "osnet"):
        assert reid_model in ["osnet", "lmbn"], "Invalid reid model"
        self.reid_model = reid_model

        # Environment variables
        self.feature_extraction_url = os.getenv(
            "FEATURE_EXTRACTION_URL", "http://localhost:8000/embedding"
        )
        self.identity_storage_url = os.getenv(
            "IDENTITY_STORAGE_URL", "http://localhost:8004/persons"
        )
        self.input_topic_name = os.getenv("INPUT_TOPIC_NAME", "reid_input")
        self.output_topic_name = os.getenv("OUTPUT_TOPIC_NAME", "reid_output")
        self.kafka_bootstrap_servers = os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.consumer_group = os.getenv("CONSUMER_GROUP", "reid_consumer_group")
        self.poll_timeout = os.getenv("POLL_TIMEOUT", 1000)
        self.max_messages = os.getenv("MAX_MESSAGES", 100)
        self.client_id = f"reid-consumer-{uuid.uuid4()}"

        # Load Avro schema
        with open("src/configs/schema.avsc", "r") as f:
            self.schema = avro.schema.parse(f.read())
        self.reader = DatumReader(self.schema)

        # Initialize service
        self.model_client = ModelServiceClient()

        # TODO: Handle multi cameras track id synchronization
        self.byte_tracker = BYTETracker(
            args=Namespace(
                track_buffer=30,
                track_high_thresh=0.7,  # first association threshold
                track_low_thresh=0.35,  # second association threshold
                match_thresh=0.3,  # matching threshold for linear assignment
                fuse_score=True,  # whether to fuse confidence scores with the iou distances before matching
                new_track_thresh=0.8,  # threshold for init new track if the detection does not match any tracks
            )
        )

        # Variables
        self.tracked_ids = []  # Store tracked id

        # Video writer for creating output videos
        self.video_writers = {}  # device_id -> cv2.VideoWriter
        self.frame_size = None  # Will be set when first frame is processed

        # Create output directories
        os.makedirs("tracking_frames", exist_ok=True)
        os.makedirs("tracking_videos", exist_ok=True)

    def init_kafka_consumer(self):
        """
        This function initializes the Kafka consumer.
        """
        try:
            self.consumer = KafkaConsumer(
                self.input_topic_name,
                client_id=self.client_id,
                bootstrap_servers=self.kafka_bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id=self.consumer_group,
                session_timeout_ms=30000,  # 30 seconds
                heartbeat_interval_ms=3000,
                max_poll_interval_ms=300000,  # 5 minutes
                retry_backoff_ms=100,  # Time to wait before retrying
                reconnect_backoff_ms=1000,  # Time to wait before reconnecting
                reconnect_backoff_max_ms=1000,  # Maximum time to wait before reconnecting
                fetch_min_bytes=1024 * 1024,  # 1MB
                fetch_max_bytes=100 * 1024 * 1024,
                max_partition_fetch_bytes=100
                * 1024
                * 1024,  # Increase from default 1MB to 100MB
            )

            # Verify consumer is properly subscribed
            topics = self.consumer.topics()
            if self.input_topic_name not in topics:
                console.log(
                    f"[bold red]Error:[/bold red] Topic {self.input_topic_name} not found in Kafka"
                )
                return False

        except KafkaError as e:
            console.log(
                f"[bold red]Kafka Error[/bold red] when initializing Kafka Consumer: {str(e)}"
            )
            return False
        except Exception as e:
            console.log(
                f"[bold red]Error[/bold red] when initializing Kafka Consumer: {str(e)}"
            )
            console.log(traceback.format_exc())
            return False
        else:
            console.log(
                "[bold cyan]Kafka Consumer[/bold cyan] initialized [bold green]successfully[/bold green] :vampire:"
            )
            return True

    def _decode_message(self, message_value: bytes) -> EdgeDeviceMessage:
        """
        Decode Avro message into Pydantic model
        """
        decoder = BinaryDecoder(io.BytesIO(message_value))
        avro_data = self.reader.read(decoder)
        return EdgeDeviceMessage(**avro_data)

    async def _get_embedding(
        self, images: List[Tuple[int, bytes]]
    ) -> List[List[float]]:
        """
        Get embeddings for a list of images

        Sample return:
        ```
        [
            [float],
            ...
        ]
        ```
        """
        tasks = [
            self.model_client.extract_features(
                img, model=self.reid_model, original_idx=original_idx
            )
            for original_idx, img in images
        ]
        embeddings = await asyncio.gather(*tasks)

        # Sort embeddings by original index
        embeddings.sort(key=lambda x: x[0])

        return [x[1]["features"] for x in embeddings]

    async def _get_genders(
        self, images: List[Tuple[int, bytes]]
    ) -> List[Dict[str, Any]]:
        """
        Get genders for a list of images

        Sample return:
        ```
        [
            {"gender": "male", "confidence": 0.95},
            ...
        ]
        ```
        """
        tasks = [
            self.model_client.classify_gender(img, original_idx)
            for original_idx, img in images
        ]
        genders = await asyncio.gather(*tasks)

        # Sort genders by original index
        genders.sort(key=lambda x: x[0])

        return [
            {"gender": x[1]["gender"], "confidence": x[1]["confidence"]}
            for x in genders
        ]

    async def get_persons_metadata(
        self, images: List[np.ndarray]
    ) -> List[PersonMetadata]:
        """
        Get person metadata for a list of images
        Return a list of people with their embeddings, genders and the cropped images
        """
        # Since asyncio.gather does not support returning original index
        # We add index here, so after sorting we can match the correct person metadata
        image_bytes = [
            (idx, cv2.imencode(".jpg", image)[1].tobytes())
            for idx, image in enumerate(images)
        ]
        embeddings = await self._get_embedding(image_bytes)
        genders = await self._get_genders(image_bytes)

        return [
            PersonMetadata(
                image=image,
                embedding=embedding,
                gender=gender["gender"],
                gender_confidence=gender["confidence"],
            )
            for image, embedding, gender in zip(images, embeddings, genders)
        ]

    def track(self, results: List[Tuple[int, EdgeDeviceMessage, List[PersonMetadata]]]):
        """
        Perform tracking logic for a list of person metadata

        Args:
            results: A list of tuples, each containing:
                - idx: The index of the message in the original batch
                - decoded_message: The decoded message from Kafka
                - person_metadatas: A list of person metadata for each bounding box in the frame
        """
        for frame_idx, decoded_message, person_metadatas in results:
            if decoded_message is None:
                continue

            print(f"Frame {decoded_message.frame_number}")

            # Convert image from bytes to numpy array
            image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if self.frame_size is None:
                self.frame_size = (
                    image.shape[1],
                    image.shape[0],
                )  # (width, height)

            bboxes = np.array([x.bbox for x in decoded_message.result])
            scores = np.array([x.confidence for x in decoded_message.result])
            cls = np.array([x.class_id for x in decoded_message.result])

            # No bboxes
            if len(bboxes) == 0:
                # Add frame to video
                self._add_frame_to_video(decoded_message.device_id, image)
                continue

            # Has bboxes --> Perform tracking
            tracking_results = self.byte_tracker.update(
                scores=scores,
                bboxes=bboxes,
                cls=cls,
            )

            if tracking_results is not None and len(tracking_results) > 0:
                # Parse tracking results: [x1, y1, x2, y2, track_id, score, cls, idx, state]
                bboxes_tracked = tracking_results[:, :4].tolist()  # [x, y, x, y]
                track_ids = tracking_results[:, 4].astype(int).tolist()  # track_id
                detection_confs = tracking_results[:, 5].tolist()  # score

                """
                Here, we need to:
                1. Find diff between track_ids and self.tracked_ids
                2. Iterate through all persons_metadatas:
                    - If id is not in diff: Save new feature to db
                    - If id is in diff (unverified):
                        + search for database
                        + if most match --> update feature to db & remap tracker
                        + if not match --> save new feature to db
                """

                # Draw bounding boxes with tracking information
                bboxes_tracked = [xyxy2xywh(bbox) for bbox in bboxes_tracked]
                image = draw_bbox(
                    image,
                    bboxes=bboxes_tracked,
                    detection_confs=detection_confs,
                    ids=track_ids,
                )

                # Add frame to video
                self._add_frame_to_video(decoded_message.device_id, image)

    def _add_frame_to_video(self, device_id: str, frame: np.ndarray):
        """
        Add frame to video writer for the specific device

        Args:
            device_id: Device identifier
            frame: Frame to add to video
        """
        try:
            if device_id not in self.video_writers:
                # Create new video writer for this device
                video_filename = f"tracking_videos/{device_id}_tracking.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 30.0  # Default FPS

                self.video_writers[device_id] = cv2.VideoWriter(
                    video_filename, fourcc, fps, self.frame_size
                )
                console.log(
                    f"Created video writer for device {device_id}: {video_filename}"
                )

            # Write frame to video
            self.video_writers[device_id].write(frame)

        except Exception as e:
            console.log(f"[bold red]Error[/bold red] adding frame to video: {str(e)}")

    def _finalize_videos(self):
        """
        Finalize and close all video writers
        """
        for device_id, writer in self.video_writers.items():
            try:
                writer.release()
                console.log(f"Finalized video for device {device_id}")
            except Exception as e:
                console.log(
                    f"[bold red]Error[/bold red] finalizing video for {device_id}: {str(e)}"
                )

        self.video_writers.clear()

    async def handle_incoming_message(self, messages):
        """
        Handle the batch of messages from Kafka `reid_input` topic

        Two steps:
        1. Get person's metadata for each frame (all bounding boxes inside the frame). This process is async
        2. Sort the results by the original timestamp of the frame
        3. Iterate through the frames and perform tracking logic.
        """
        partition, msgs = list(messages.items())[0]
        console.log(f"Received batch: {len(msgs)} messages on partition {partition}")

        async def process_msg(idx, msg):
            try:
                # Decode the message from bytes
                decoded_message = self._decode_message(msg.value)

                # Convert image from bytes to numpy array (4ms average - 300KB for Full HD Image)
                image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                # Crop images
                person_images = crop_image(
                    image, bboxes=[x.bbox for x in decoded_message.result]
                )

                # Get person metadata - now this is properly awaited within the async context
                person_metadatas = await self.get_persons_metadata(person_images)

                # # Draw bounding boxes
                # image = draw_bbox(
                #     image,
                #     bboxes=[x.bbox for x in decoded_message.result],
                #     genders=[x.gender for x in person_metadatas],
                #     gender_confs=[x.gender_confidence for x in person_metadatas],
                #     detection_confs=[x.confidence for x in decoded_message.result],
                # )
                # cv2.imwrite(f"test/frame_{decoded_message.frame_number}.jpg", image)

                console.log(
                    f"STEP 1: {decoded_message.device_id} - Frame: {decoded_message.frame_number} Done"
                )
                # Return index and any result you want to keep for downstream tracking
                return idx, decoded_message, person_metadatas

            except Exception as e:
                console.log(f"[bold red]Error[/bold red] processing message: {str(e)}")
                console.log(traceback.format_exc())
                return idx, None, None

        # 1. Launch all tasks concurrently, keeping track of their original order
        console.log("[bold cyan]Step 1: Async get person metadata[/bold cyan]")
        tasks = [process_msg(idx, msg) for idx, msg in enumerate(msgs)]
        results = await asyncio.gather(*tasks)

        # 2. Sort results by original index to preserve order
        console.log("[bold cyan]Step 2: Sort results by original index[/bold cyan]")
        results.sort(key=lambda x: x[0])

        # 3. Perform tracking logic
        console.log("[bold cyan]Step 3: Perform tracking logic[/bold cyan]")
        self.track(results)

    async def run_async(self):
        """
        Async version of the main consumer loop
        """
        if not self.init_kafka_consumer():
            return

        console.log(
            f"[bold cyan]Starting consumer[/bold cyan] ({self.client_id}) for topic: {self.input_topic_name}"
        )
        console.log(f"Consumer group: {self.consumer_group}")

        # Initialize the model client's HTTP client
        async with self.model_client:
            try:
                while True:
                    messages = self.consumer.poll(
                        timeout_ms=int(self.poll_timeout),
                        max_records=int(self.max_messages),
                    )

                    if messages:
                        await self.handle_incoming_message(messages)
                    else:
                        console.log("[yellow]No messages received[/yellow]")
                        await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                console.log("[yellow]Shutting down consumer...[/yellow]")
            except Exception as e:
                console.log(f"[bold red]Error in consumer loop:[/bold red] {str(e)}")
                console.log(traceback.format_exc())
            finally:
                self.consumer.close()
                self._finalize_videos()  # Ensure videos are properly finalized
                console.log("[yellow]Consumer closed[/yellow]")

    def start(self):
        """
        Start the consumer - now runs the async version
        """
        asyncio.run(self.run_async())
