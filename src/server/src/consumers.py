import asyncio
import io
import os
import time
import uuid
from typing import Any, Dict, List, Tuple

import avro.schema
import cv2
import numpy as np
from avro.io import BinaryDecoder, BinaryEncoder, DatumReader, DatumWriter
from dotenv import load_dotenv

from kafka import KafkaConsumer, KafkaProducer
from src.apis import ModelServiceClient
from src.embeddings import PersonID, RedisPersonIDsStorage
from src.schemas import EdgeDeviceMessage, PersonMetadata, ProcessedFrameMessage
from src.trackers import BYTETracker, Namespace
from src.utils import Logger
from src.utils.ops import crop_image, draw_bbox, xyxy2xywh

logger = Logger(__name__)


load_dotenv()


class ReIdConsumer:
    def __init__(self):
        # Environment variables
        self.reid_model = os.getenv("EMBEDDING_MODEL", "osnet")
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

        # Load input schema for consuming from reid_input topic
        with open("src/configs/input.avsc", "r") as f:
            self.schema = avro.schema.parse(f.read())
        self.reader = DatumReader(self.schema)

        # Load output schema for producing to reid_output topic
        with open("src/configs/output.avsc", "r") as f:
            self.output_schema = avro.schema.parse(f.read())
        self.output_writer = DatumWriter(self.output_schema)

        # Initialize service
        self.model_client = ModelServiceClient()

        # Initialize Redis storage for person IDs
        self.person_storage = RedisPersonIDsStorage(
            redis_host=self.redis_host,
            redis_port=self.redis_port,
            redis_db=self.redis_db,
        )
        self.person_storage.clear()

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

        # Frame size for processing
        self.frame_size = None  # Will be set when first frame is processed

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
                logger.error(f"Topic {self.input_topic_name} not found in Kafka")
                return False
        except Exception:
            logger.error("Error when initializing Kafka Consumer", exc_info=True)
            return False
        else:
            logger.info("Kafka Consumer initialized successfully")
            return True

    def init_kafka_producer(self):
        """
        Initialize the Kafka producer for sending processed frames to reid_output topic
        """
        try:
            self.producer = KafkaProducer(
                client_id=f"{self.client_id}-producer",
                bootstrap_servers=self.kafka_bootstrap_servers,
                key_serializer=lambda x: x.encode("utf-8"),  # Device ID as key
                value_serializer=self.serialize_output_message,  # Serialize using output schema
                linger_ms=10,
                batch_size=16384,
                acks=1,
                max_in_flight_requests_per_connection=5,
                max_request_size=100 * 1024 * 1024,  # 100MB for large images
            )
            logger.info("Kafka Producer initialized successfully")
            return True
        except Exception:
            logger.error("Error when initializing Kafka Producer", exc_info=True)
            return False

    def serialize_output_message(self, message: dict) -> bytes:
        """
        Serialize output message using Avro schema for reid_output topic
        """
        try:
            bytes_writer = io.BytesIO()
            encoder = BinaryEncoder(bytes_writer)
            self.output_writer.write(message, encoder)
            return bytes_writer.getvalue()
        except Exception:
            logger.error("Error serializing output message")
            raise Exception("Error serializing output message")

    def decode_input_message(self, message_value: bytes) -> EdgeDeviceMessage:
        """
        Decode Avro message into Pydantic model
        """
        decoder = BinaryDecoder(io.BytesIO(message_value))
        avro_data = self.reader.read(decoder)
        return EdgeDeviceMessage(**avro_data)

    async def _get_embedding(
        self, images: List[Tuple[int, bytes]]
    ) -> List[List[float]]:
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
        for _, decoded_message, person_metadatas in results:
            if decoded_message is None:
                continue

            # Convert image from bytes to numpy array
            image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            if self.frame_size is None:
                self.frame_size = (image.shape[1], image.shape[0])  # w x h

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
                self._produce_processed_frame(decoded_message, image, [])
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

                if new_ids != []:
                    logger.info(f"New/Unverified IDs: {new_ids}")

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
                                logger.info(
                                    f"RE-IDENTIFIED! Track ID {track_id} → Person ID {matched_person.id}"
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
                                logger.info("NEW PERSON CREATED!")
                                logger.info(
                                    f"Creating new person with gender: {person_metadata.gender} (confidence: {person_metadata.gender_confidence:.3f})"
                                )

                                # Get next global ID
                                global_id = self._get_next_global_id()
                                current_person.set_id(global_id)

                                logger.info(f"Assigned Global ID: {global_id}")

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

                                logger.info(
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
                        logger.error(
                            f"ERROR! Detection index {det_idx} >= person_metadatas length {len(person_metadatas)}"
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

                # Create tracked persons data
                tracked_persons = []
                for bbox, track_id, detection_conf, gender, gender_conf in zip(
                    bboxes_tracked, track_ids, detection_confs, genders, gender_confs
                ):
                    tracked_persons.append(
                        {
                            "person_id": int(track_id),
                            "bbox": bbox.tolist()
                            if isinstance(bbox, np.ndarray)
                            else bbox,
                            "confidence": float(detection_conf),
                            "gender": str(gender),
                            "gender_confidence": float(gender_conf),
                        }
                    )

                self._produce_processed_frame(decoded_message, image, tracked_persons)
            else:
                # Draw frame number even when no tracking results
                image = draw_bbox(
                    image,
                    bboxes=[],
                    frame_number=decoded_message.frame_number,
                )
                self._produce_processed_frame(decoded_message, image, [])

    def _get_next_global_id(self) -> int:
        """Get the next available global ID"""
        while self.person_storage.get_person_by_id(self.next_global_id) is not None:
            self.next_global_id += 1
        current_id = self.next_global_id
        self.next_global_id += 1

        return current_id

    def _produce_processed_frame(
        self,
        decoded_message: EdgeDeviceMessage,
        image: np.ndarray,
        tracked_persons: List[Dict[str, Any]],
    ):
        """
        Produce processed frame to reid_output topic

        Args:
            decoded_message: The decoded message from Kafka
            image: The processed image
            tracked_persons: A list of dictionaries containing tracked person information
        """
        try:
            # Encode processed image to bytes
            _, img_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_bytes = img_bytes.tobytes()

            # Create message for reid_output topic - must match output_schema.avsc exactly
            output_message = {
                "device_id": decoded_message.device_id,
                "frame_number": decoded_message.frame_number,
                "tracked_persons": tracked_persons,
                "created_at": decoded_message.created_at,
                "image_data": img_bytes,
            }

            # Validate message
            ProcessedFrameMessage(**output_message)

            # Send to reid_output topic
            self.producer.send(
                self.output_topic_name,
                key=decoded_message.device_id,
                value=output_message,
            )

        except Exception:
            logger.error("Error producing processed frame", exc_info=True)

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
            logger.info(f"Successfully remapped {old_id} -> {new_id}")

    async def handle_incoming_message(self, messages):
        """
        Handle the batch of messages from Kafka `reid_input` topic

        Two steps:
        1. Get person's metadata for each frame (all bounding boxes inside the frame). This process is async
        2. Sort the results by the original timestamp of the frame
        3. Iterate through the frames and perform tracking logic.
        """
        partition, msgs = list(messages.items())[0]
        logger.info(
            f"Received batch: {len(msgs)} messages on partition {partition.partition}"
        )

        async def process_msg(idx, msg):
            try:
                # Decode the message from bytes
                decoded_message = self.decode_input_message(msg.value)

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

            except Exception:
                logger.error("Error processing message", exc_info=True)
                return idx, None, None

        # 1. Launch all tasks concurrently, keeping track of their original order
        start_time = time.time()
        tasks = [process_msg(idx, msg) for idx, msg in enumerate(msgs)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        frame_numbers = [x[1].frame_number for x in results]
        logger.info(
            f"Step 1: Async get person metadata from {min(frame_numbers)} to {max(frame_numbers)}  - Time taken: {end_time - start_time:.2f} seconds"
        )

        # 2. Sort results by original index to preserve order
        results.sort(key=lambda x: x[0])

        # 3. Perform tracking logic
        start_time = time.time()
        self.track(results)
        end_time = time.time()
        logger.info(
            f"Step 2: Perform tracking logic from {min(frame_numbers)} to {max(frame_numbers)}  - Time taken: {end_time - start_time:.2f} seconds"
        )

    async def run_async(self):
        """
        Async version of the main consumer loop
        """
        if not self.init_kafka_consumer():
            return

        if not self.init_kafka_producer():
            return

        logger.info(
            f"Starting consumer ({self.client_id}) for topic: {self.input_topic_name}"
        )
        logger.info(f"Consumer group: {self.consumer_group}")
        logger.info(f"Output topic: {self.output_topic_name}")

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
            except KeyboardInterrupt:
                logger.info("Shutting down consumer...")
            except Exception:
                logger.error("Error in consumer loop", exc_info=True)
            finally:
                self.consumer.close()
                self.producer.flush()
                self.producer.close()

    def start(self):
        """
        Start the consumer - now runs the async version
        """
        asyncio.run(self.run_async())
