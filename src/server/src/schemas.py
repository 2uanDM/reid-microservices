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


class PersonMetadata(BaseModel):
    """
    Including `image`, `embedding` and `gender` for a person
    """

    image: np.ndarray
    embedding: list
    gender: Literal["male", "female"]
    gender_confidence: float

    model_config = dict(arbitrary_types_allowed=True)
