import asyncio
from pathlib import Path
from typing import Any, Dict, Literal, Union

import httpx


class ModelServiceClient:
    """Async client library for consuming the Model Service APIs."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize the ModelServiceClient.

        Args:
            base_url (str): Base URL of the model service
            timeout (float): Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def _ensure_client(self):
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    async def extract_features(
        self,
        image_path: Union[str, Path, bytes],
        dynamic_batching: bool = False,
        model: Literal["osnet", "lmbn"] = "osnet",
    ) -> Dict[str, Any]:
        """
        Extract features from a single image.

        Args:
            image_path: Path to image file or bytes data
            model: Model to use for feature extraction ("osnet" or "lmbn")

        Returns:
            Dict containing features and shape information
        """
        self._ensure_client()

        # Handle different input types
        if isinstance(image_path, (str, Path)):
            with open(image_path, "rb") as f:
                image_data = f.read()
            filename = Path(image_path).name
        else:
            image_data = image_path
            filename = "image.jpg"

        files = {"image": (filename, image_data, "image/jpeg")}
        data = {"model": model}

        endpoint = "/embedding" if not dynamic_batching else "/embedding/batch"

        response = await self.client.post(
            f"{self.base_url}{endpoint}", files=files, data=data
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Feature extraction failed with status {response.status_code}: {response.text}"
            )

    async def classify_gender(
        self, image_path: Union[str, Path, bytes]
    ) -> Dict[str, Any]:
        """
        Classify gender from input image.

        Args:
            image_path: Path to image file or bytes data

        Returns:
            Dict containing gender prediction, confidence, and probabilities
        """
        self._ensure_client()

        # Handle different input types
        if isinstance(image_path, (str, Path)):
            with open(image_path, "rb") as f:
                image_data = f.read()
            filename = Path(image_path).name
        else:
            image_data = image_path
            filename = "image.jpg"

        files = {"image": (filename, image_data, "image/jpeg")}

        response = await self.client.post(
            f"{self.base_url}/gender/classify", files=files
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Gender classification failed with status {response.status_code}: {response.text}"
            )


# Convenience functions for synchronous usage
def sync_wrapper(async_func):
    """Wrapper to make async functions callable synchronously."""

    def wrapper(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))

    return wrapper
