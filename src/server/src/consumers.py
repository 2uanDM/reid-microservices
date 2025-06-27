import asyncio
import io
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import avro.schema
import cv2
import numpy as np
from avro.io import BinaryDecoder, DatumReader
from dotenv import load_dotenv
from rich.console import Console

from kafka import KafkaConsumer
from src.apis import ModelServiceClient
from src.embeddings import PersonID, RedisPersonIDsStorage
from src.schemas import EdgeDeviceMessage, PersonMetadata
from src.trackers import BYTETracker, Namespace
from src.utils.ops import crop_image, draw_bbox, xyxy2xywh

console = Console()

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create a file console for logging
log_filename = f"logs/reid_consumer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_console = None  # Will be set in ReIdConsumer.__init__


def log_both(message: str):
    """Log message to both console and file"""
    verbose = os.getenv("VERBOSE", "true").lower() == "true"
    if verbose:
        console.log(message)
        if file_console:
            file_console.log(message)


load_dotenv()


class ReIdConsumer:
    def __init__(self):
        self.reid_model = os.getenv("EMBEDDING_MODEL", "osnet")

        # Log file setup
        self.log_file = open(log_filename, "w", encoding="utf-8")
        self.file_console = Console(file=self.log_file, width=120)

        # Update the global file_console to use this instance
        global file_console
        file_console = self.file_console

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

        # Redis configuration
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        self.redis_db = int(os.getenv("REDIS_DB", 0))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
        self.gender_threshold = float(os.getenv("GENDER_THRESHOLD", 0.9))

        # Load Avro schema
        with open("src/configs/schema.avsc", "r") as f:
            self.schema = avro.schema.parse(f.read())
        self.reader = DatumReader(self.schema)

        # Initialize service
        self.model_client = ModelServiceClient()

        # Initialize Redis storage for person IDs
        self.person_storage = RedisPersonIDsStorage(
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            redis_db=self.redis_db,
        )
        self.person_storage.clear()

        # TODO: Handle multi cameras track id synchronization
        self.byte_tracker = BYTETracker(
            args=Namespace(
                track_buffer=30,
                track_high_thresh=0.7,  # first association threshold
                track_low_thresh=0.35,  # second association threshold
                match_thresh=0.3,  # matching threshold for linear assignment
                fuse_score=True,  # whether to fuse confidence scores with the iou distances before matching
                new_track_thresh=0.82,  # threshold for init new track if the detection does not match any tracks
            )
        )

        # Variables
        self.tracked_ids = []  # Store tracked id
        self.next_global_id = 1  # Global ID counter for new persons

        # Video writer for creating output videos
        self.video_writers = {}  # device_id -> cv2.VideoWriter
        self.frame_size = None  # Will be set when first frame is processed

        # Create output directories
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
                log_both(
                    f"[bold red]Error:[/bold red] Topic {self.input_topic_name} not found in Kafka"
                )
                return False
        except Exception as e:
            log_both(
                f"[bold red]Error[/bold red] when initializing Kafka Consumer: {str(e)}"
            )
            log_both(traceback.format_exc())
            return False
        else:
            log_both(
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
                img,
                model=self.reid_model,
                original_idx=original_idx,
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

            # Convert image from bytes to numpy array
            image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if self.frame_size is None:
                # w x h
                self.frame_size = (image.shape[1], image.shape[0])

            bboxes = np.array([x.bbox for x in decoded_message.result])
            scores = np.array([x.confidence for x in decoded_message.result])
            cls = np.array([x.class_id for x in decoded_message.result])

            # No bboxes
            if len(bboxes) == 0:
                image = draw_bbox(
                    image,
                    bboxes=[],
                    frame_number=decoded_message.frame_number,
                )
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
                detection_indices = (
                    tracking_results[:, 7].astype(int).tolist()
                )  # idx mapping to original detections

                """
                Here, we need to:
                1. Find diff between track_ids and self.tracked_ids
                2. Iterate through all persons_metadatas:
                    - If id is not in diff: Update existing feature in db
                    - If id is in diff (unverified):
                        + search for database
                        + if most match --> update feature to db & remap tracker
                        + if not match --> save new feature to db
                """

                # Get unverified ids (new ids)
                new_ids = [x for x in track_ids if x not in self.tracked_ids]
                existing_ids = [x for x in track_ids if x in self.tracked_ids]

                if new_ids != []:
                    log_both(f"New/Unverified IDs: {new_ids}")
                    log_both(f"Existing/Verified IDs: {existing_ids}")
                    log_both(f"Currently tracked IDs: {self.tracked_ids}")
                    log_both(
                        f"Total persons in storage: {len(self.person_storage._get_all_ids())}"
                    )

                # Initialize lists to collect gender information for drawing
                genders = []
                gender_confs = []

                # Process each tracked person
                for i, (bbox, track_id, detection_conf, det_idx) in enumerate(
                    zip(bboxes_tracked, track_ids, detection_confs, detection_indices)
                ):
                    # Get the corresponding person metadata
                    if det_idx < len(person_metadatas):
                        person_metadata = person_metadatas[det_idx]

                        # Collect gender information for drawing
                        genders.append(person_metadata.gender)
                        gender_confs.append(person_metadata.gender_confidence)

                        # Create PersonID object from metadata
                        current_person = PersonID(
                            fullbody_embedding=person_metadata.embedding,
                            fullbody_bbox=np.array(bbox),
                            body_conf=np.float32(detection_conf),
                            gender=person_metadata.gender,
                            gender_confidence=person_metadata.gender_confidence,
                        )

                        if track_id in new_ids:
                            if (
                                person_metadata.gender_confidence
                                > self.gender_threshold
                            ):
                                # Search for similar person in database with gender filtering
                                matched_person, similarity = self.person_storage.search(
                                    current_person_id=current_person,
                                    current_frame_id=track_ids,  # Exclude current frame IDs from search
                                    threshold=self.similarity_threshold,
                                )

                            else:
                                # Search without gender filtering when gender confidence is low
                                matched_person, similarity = self.person_storage.search(
                                    current_person_id=current_person,
                                    current_frame_id=track_ids,  # Exclude current frame IDs from search
                                    threshold=self.similarity_threshold,
                                )

                            if matched_person is not None:
                                log_both(
                                    f"[bold green]RE-IDENTIFIED![/bold green] Track ID {track_id} → Person ID {matched_person.id}"
                                )
                                log_both(
                                    f"Remapping BYTETracker ID {track_id} → {matched_person.id}"
                                )

                                # Update the matched person with new embedding
                                embedding_updated = matched_person.update_embedding(
                                    new_embedding=np.array(person_metadata.embedding),
                                    body_score=detection_conf,
                                    frame_number=decoded_message.frame_number,
                                )
                                if embedding_updated:
                                    self.person_storage.update(matched_person)

                                # Remap the tracker ID to use the existing global ID
                                self.remap_bytetracker_ids(
                                    self.byte_tracker, track_id, matched_person.id
                                )

                                # Update track_ids list for this frame
                                track_ids[i] = matched_person.id

                                # Add to tracked IDs if not already present
                                if matched_person.id not in self.tracked_ids:
                                    self.tracked_ids.append(matched_person.id)
                            else:
                                # No match found - create new person
                                log_both("[bold cyan]NEW PERSON CREATED![/bold cyan]")
                                log_both(
                                    f"Creating new person with gender: {person_metadata.gender} (confidence: {person_metadata.gender_confidence:.3f})"
                                )

                                # Get next global ID
                                global_id = self._get_next_global_id()
                                current_person.set_id(global_id)

                                log_both(f"Assigned Global ID: {global_id}")

                                # Update embedding for the new person
                                embedding_updated = current_person.update_embedding(
                                    new_embedding=np.array(person_metadata.embedding),
                                    body_score=detection_conf,
                                    frame_number=decoded_message.frame_number,
                                )

                                # Add to storage
                                self.person_storage.add(current_person)

                                # Remap track ID to global ID
                                self.remap_bytetracker_ids(
                                    self.byte_tracker, track_id, global_id
                                )

                                # Update track_ids list for this frame
                                track_ids[i] = global_id

                                # Add to tracked IDs
                                if global_id not in self.tracked_ids:
                                    self.tracked_ids.append(global_id)

                                log_both(
                                    f"Remapped: Track ID {track_id} → Global ID {global_id}"
                                )
                        else:
                            # This is an existing tracked person - update their embedding
                            existing_person = self.person_storage.get_person_by_id(
                                track_id
                            )
                            if existing_person is not None:
                                embedding_updated = existing_person.update_embedding(
                                    new_embedding=np.array(person_metadata.embedding),
                                    body_score=detection_conf,
                                    frame_number=decoded_message.frame_number,
                                )
                                if embedding_updated:
                                    self.person_storage.update(existing_person)
                            else:
                                current_person.set_id(track_id)
                                embedding_updated = current_person.update_embedding(
                                    new_embedding=np.array(person_metadata.embedding),
                                    body_score=detection_conf,
                                    frame_number=decoded_message.frame_number,
                                )
                                self.person_storage.add(current_person)
                    else:
                        log_both(
                            f"[bold red]ERROR![/bold red] Detection index {det_idx} >= person_metadatas length {len(person_metadatas)}"
                        )

                # Add frame to video with updated track IDs
                bboxes_tracked = [xyxy2xywh(bbox) for bbox in bboxes_tracked]

                image = draw_bbox(
                    image,
                    bboxes=bboxes_tracked,
                    detection_confs=detection_confs,
                    ids=track_ids,
                    genders=genders,
                    gender_confs=gender_confs,
                    frame_number=decoded_message.frame_number,
                )
                self._add_frame_to_video(decoded_message.device_id, image)
            else:
                # Draw frame number even when no tracking results
                image = draw_bbox(
                    image,
                    bboxes=[],
                    frame_number=decoded_message.frame_number,
                )
                self._add_frame_to_video(decoded_message.device_id, image)

    def _get_next_global_id(self) -> int:
        """Get the next available global ID"""
        while self.person_storage.get_person_by_id(self.next_global_id) is not None:
            self.next_global_id += 1
        current_id = self.next_global_id
        self.next_global_id += 1

        return current_id

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

            # Write frame to video
            self.video_writers[device_id].write(frame)

        except Exception as e:
            log_both(f"[bold red]Error[/bold red] adding frame to video: {str(e)}")

    def remap_bytetracker_ids(self, bytetracker: BYTETracker, old_id: int, new_id: int):
        """Remap a track ID in BYTETracker from old_id to new_id."""
        remapped = False

        # Check tracked_stracks
        for track in bytetracker.tracked_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                remapped = True
                break

        # Check lost_stracks
        for track in bytetracker.lost_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                remapped = True
                break

        # Check removed_stracks
        for track in bytetracker.removed_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                remapped = True
                break

        if remapped:
            log_both(f"Successfully remapped {old_id} → {new_id}")

    def _finalize_videos(self):
        """
        Finalize and close all video writers
        """
        for device_id, writer in self.video_writers.items():
            try:
                writer.release()
            except Exception as e:
                log_both(
                    f"[bold red]Error[/bold red] finalizing video for {device_id}: {str(e)}"
                )

        self.video_writers.clear()

    def _cleanup_logging(self):
        """
        Cleanup and close log file
        """
        try:
            # Flush and close the log file
            if hasattr(self, "log_file") and self.log_file:
                self.log_file.flush()
                self.log_file.close()
        except Exception as e:
            print(f"Error closing log file: {e}")

    async def handle_incoming_message(self, messages):
        """
        Handle the batch of messages from Kafka `reid_input` topic

        Two steps:
        1. Get person's metadata for each frame (all bounding boxes inside the frame). This process is async
        2. Sort the results by the original timestamp of the frame
        3. Iterate through the frames and perform tracking logic.
        """
        partition, msgs = list(messages.items())[0]
        log_both(
            f"Received batch: {len(msgs)} messages on partition {partition.partition}"
        )

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

                # Return index and any result you want to keep for downstream tracking
                return idx, decoded_message, person_metadatas

            except Exception as e:
                console.print(
                    f"[bold red]Error[/bold red] processing message: {str(e)}"
                )
                console.print(traceback.format_exc())
                return idx, None, None

        # 1. Launch all tasks concurrently, keeping track of their original order
        start_time = time.time()
        tasks = [process_msg(idx, msg) for idx, msg in enumerate(msgs)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        frame_numbers = [x[1].frame_number for x in results]
        console.print(
            f"Step 1: Async get person metadata from {min(frame_numbers)} to {max(frame_numbers)}  - Time taken: {end_time - start_time:.2f} seconds"
        )

        # 2. Sort results by original index to preserve order
        results.sort(key=lambda x: x[0])

        # 3. Perform tracking logic
        start_time = time.time()
        self.track(results)
        end_time = time.time()
        console.print(
            f"Step 2: Perform tracking logic from {min(frame_numbers)} to {max(frame_numbers)}  - Time taken: {end_time - start_time:.2f} seconds"
        )

    async def run_async(self):
        """
        Async version of the main consumer loop
        """
        if not self.init_kafka_consumer():
            return

        console.print(
            f"[bold cyan]Starting consumer[/bold cyan] ({self.client_id}) for topic: {self.input_topic_name}"
        )
        console.print(f"Consumer group: {self.consumer_group}")

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
                        console.print("[yellow]No messages received[/yellow]")
                        await asyncio.sleep(0.5)
            except KeyboardInterrupt:
                console.print("[yellow]Shutting down consumer...[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error in consumer loop:[/bold red] {str(e)}")
                console.print(traceback.format_exc())
            finally:
                self.consumer.close()
                self._finalize_videos()  # Ensure videos are properly finalized
                console.print("[yellow]Consumer closed[/yellow]")
                self._cleanup_logging()

    def start(self):
        """
        Start the consumer - now runs the async version
        """
        asyncio.run(self.run_async())
