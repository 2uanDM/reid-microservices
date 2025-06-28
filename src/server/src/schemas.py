from datetime import datetime
from typing import List, Literal

import numpy as np
from pydantic import BaseModel


class Detection(BaseModel):
    bbox: List[float]  # x1, y1, x2, y2
    confidence: float
    class_id: int


class EdgeDeviceMessage(BaseModel):
    device_id: str
    frame_number: int
    result: List[Detection]
    created_at: int
    image_data: bytes

    def get_created_at_datetime(self) -> datetime:
        return datetime.fromtimestamp(
            self.created_at / 1_000_000_000
        )  # Convert from nanoseconds to seconds


class TrackedPerson(BaseModel):
    """
    Represents a tracked person with their ID and detection information
    """

    person_id: int
    bbox: List[float]  # x1, y1, x2, y2 in xywh format
    confidence: float
    gender: str
    gender_confidence: float


class ProcessedFrameMessage(BaseModel):
    """
    Message sent to reid_output topic containing processed frame with tracking info
    """

    device_id: str
    frame_number: int
    tracked_persons: List[TrackedPerson]
    created_at: int  # Original timestamp from edge device
    image_data: bytes  # Frame with tracking overlays

    def get_created_at_datetime(self) -> datetime:
        return datetime.fromtimestamp(
            self.created_at / 1_000_000_000
        )  # Convert from nanoseconds to seconds


class PersonMetadata(BaseModel):
    """
    Including `image`, `embedding` and `gender` for a person
    """

    image: np.ndarray
    embedding: list
    gender: Literal["male", "female"]
    gender_confidence: float

    model_config = dict(arbitrary_types_allowed=True)


# Identity Storage
class Embedding(BaseModel):
    value: list[float]
    detect_conf: float


class Person(BaseModel):
    id: int
    first_embedding: Embedding | None
    last_embedding: Embedding | None
    top_k_embeddings: list[Embedding]
