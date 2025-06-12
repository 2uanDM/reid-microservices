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
from src.utils.ops import crop_image, draw_bbox

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

        # Initialize model client
        self.model_client = ModelServiceClient()

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

    async def handle_incoming_message(self, messages):
        """
        Handle incoming messages asynchronously
        """
        partition, msgs = list(messages.items())[0]
        console.log(f"Received batch: {len(msgs)} messages on partition {partition}")
        os.makedirs("test", exist_ok=True)

        for msg in msgs:  # msg here represents for a frame of the video
            try:
                # Decode the message from bytes
                decoded_message = self._decode_message(msg.value)
                console.log(
                    f"ID: {decoded_message.device_id} - Frame: {decoded_message.frame_number}"
                )

                # Convert image from bytes to numpy array (4ms average - 300KB for Full HD Image)
                image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                # Crop images
                person_images = crop_image(
                    image, bboxes=[x.bbox for x in decoded_message.result]
                )

                # Get person metadata - now this is properly awaited within the async context
                person_metadatas = await self.get_persons_metadata(person_images)

                # Draw bounding boxes
                image = draw_bbox(
                    image,
                    bboxes=[x.bbox for x in decoded_message.result],
                    genders=[x.gender for x in person_metadatas],
                    gender_confs=[x.gender_confidence for x in person_metadatas],
                    detection_confs=[x.confidence for x in decoded_message.result],
                )

                cv2.imwrite(f"test/frame_{decoded_message.frame_number}.jpg", image)

            except Exception as e:
                console.log(f"[bold red]Error[/bold red] processing message: {str(e)}")
                console.log(traceback.format_exc())

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
                console.log("[yellow]Consumer closed[/yellow]")

    def start(self):
        """
        Start the consumer - now runs the async version
        """
        asyncio.run(self.run_async())
