from datetime import datetime
from typing import List

from pydantic import BaseModel


class Detection(BaseModel):
    bbox: List[float]
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
