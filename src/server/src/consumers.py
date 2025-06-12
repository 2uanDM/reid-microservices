import io
import os
import time
import traceback
import uuid

import avro.schema
import cv2
import numpy as np
from avro.io import BinaryDecoder, DatumReader
from dotenv import load_dotenv
from rich.console import Console

from kafka import KafkaConsumer
from kafka.errors import KafkaError
from src.schemas import EdgeDeviceMessage
from src.utils.ops import draw_bbox

console = Console()


load_dotenv()


class ReIdConsumer:
    def __init__(self):
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

    def decode_message(self, message_value: bytes) -> EdgeDeviceMessage:
        """
        Decode Avro message into Pydantic model
        """
        decoder = BinaryDecoder(io.BytesIO(message_value))
        avro_data = self.reader.read(decoder)
        return EdgeDeviceMessage(**avro_data)

    def handle_incoming_message(self, messages):
        for partition, msgs in messages.items():
            console.log(f"Processing messages from partition {partition}")
            console.log(f"Received batch: {len(msgs)} messages")
            os.makedirs("test", exist_ok=True)
            for msg in msgs:
                # print(f"Size in MB: {round(len(msg.value) / 1024 / 1024, 2)}")
                try:
                    decoded_message = self.decode_message(msg.value)
                    console.log(
                        f"Device ID: {decoded_message.device_id} - Frame Number: {decoded_message.frame_number}"
                    )

                    # Convert image from bytes to numpy array (4ms average - 300KB for Full HD Image)
                    image = np.frombuffer(decoded_message.image_data, dtype=np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    # Draw bounding boxes
                    image = draw_bbox(
                        image,
                        bboxes=[x.bbox for x in decoded_message.result],
                        class_ids=[x.class_id for x in decoded_message.result],
                        confidences=[x.confidence for x in decoded_message.result],
                    )

                    cv2.imwrite(f"test/frame_{decoded_message.frame_number}.jpg", image)

                except Exception as e:
                    console.log(
                        f"[bold red]Error[/bold red] processing message: {str(e)}"
                    )
                    console.log(traceback.format_exc())

    def start(self):
        if not self.init_kafka_consumer():
            return

        console.log(
            f"[bold cyan]Starting consumer[/bold cyan] ({self.client_id}) for topic: {self.input_topic_name}"
        )
        console.log(f"Consumer group: {self.consumer_group}")

        try:
            while True:
                messages = self.consumer.poll(
                    timeout_ms=int(self.poll_timeout),
                    max_records=int(self.max_messages),
                )

                if messages:
                    self.handle_incoming_message(messages)
                time.sleep(int(self.poll_timeout) / 1000)
        except KeyboardInterrupt:
            console.log("[yellow]Shutting down consumer...[/yellow]")
        except Exception as e:
            console.log(f"[bold red]Error in consumer loop:[/bold red] {str(e)}")
            console.log(traceback.format_exc())
        finally:
            self.consumer.close()
            console.log("[yellow]Consumer closed[/yellow]")
