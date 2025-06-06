import io
from typing import List, Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ray import serve
from torchvision import transforms

from src.models.lightmbn_n import LMBN_n
from src.models.osnet import osnet_x1_0

app = FastAPI()


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_cpus": 4,
        "num_gpus": 1,
    },
    max_ongoing_requests=16,
)
@serve.ingress(app)
class FeatureExtractionService:
    def __init__(self, img_size: tuple = (128, 64), device: str | None = None):
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Initializing FeatureExtractionService on device: {self.device}")

        # Initialize OSNet model
        self.os_model = osnet_x1_0(
            num_classes=1000,
            loss="softmax",
            pretrained=True,
        ).to(self.device)
        self.os_model.eval()

        # Initialize LMBN model
        self.lmbn_model = LMBN_n(
            num_classes=1000,
            feats=512,
            activation_map=False,
        ).to(self.device)
        self.lmbn_model.eval()

        # Initialize transform
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 128)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.img_size = img_size

        # Warmup the model
        self._warmup()

    def _warmup(self):
        """Warmup the model with dummy data to avoid cold start latency."""
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
        with torch.no_grad():
            _ = self.os_model(dummy_input)
            _ = self.lmbn_model(dummy_input)
        print("Model warmup completed")

    def preprocess_single(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess a single image."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {str(e)}"
            )

    def preprocess_batch(self, images_bytes: List[bytes]) -> torch.Tensor:
        """Preprocess a batch of images."""
        processed_images = []
        for img_bytes in images_bytes:
            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_tensor = self.transform(img).unsqueeze(0)
                processed_images.append(img_tensor)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid image format: {str(e)}"
                )

        batch_tensors = torch.cat(processed_images, dim=0)
        return batch_tensors.to(self.device)

    @torch.no_grad()
    async def extract_features_single(
        self,
        image_bytes: bytes,
        model: Literal["osnet", "lmbn"] = "osnet",
    ) -> List[float]:
        """Extract features from a single image - optimized for low latency."""
        img_tensor = self.preprocess_single(image_bytes)
        if model == "osnet":
            features = self.os_model(img_tensor)
            return features.detach().cpu().numpy().flatten().tolist()
        else:
            features = self.lmbn_model(img_tensor)  # Shape: [1, 512, 7]
            # Average across the 7 components to get [1, 512] features
            features_avg = features.mean(dim=2)  # Shape: [1, 512]
            return features_avg.detach().cpu().numpy().flatten().tolist()

    @serve.batch(
        max_batch_size=16,
        batch_wait_timeout_s=0.01,  # Much shorter timeout (10ms)
    )
    @torch.no_grad()
    async def extract_features_batch(
        self,
        images_bytes_list: List[bytes],
        model: Literal["osnet", "lmbn"] = "osnet",
    ) -> List[List[float]]:
        """Extract features from multiple images - optimized for throughput."""
        # images_bytes_list is already a list of bytes from different requests
        batch_tensors = self.preprocess_batch(images_bytes_list)
        if model == "osnet":
            features = self.os_model(batch_tensors)
            return features.detach().cpu().numpy().tolist()
        else:
            features = self.lmbn_model(batch_tensors)  # Shape: [batch, 512, 7]
            # Average across the 7 components to get [batch, 512] features
            features_avg = features.mean(dim=2)  # Shape: [batch, 512]
            return features_avg.detach().cpu().numpy().tolist()

    @app.post("/embedding")
    async def embedding(
        self,
        image: UploadFile = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """Single image endpoint - prioritizes latency."""
        image_data = await image.read()
        features = await self.extract_features_single(image_data, model)
        return {"features": features, "shape": [len(features)]}

    @app.post("/embedding/batch")
    async def embedding_batch(
        self,
        images: List[UploadFile] = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """Batch endpoint - uses batching for throughput."""
        images_data = []
        for img in images:
            img_data = await img.read()
            images_data.append(img_data)

        # This will trigger batching if multiple requests come simultaneously
        features_list = []
        for img_data in images_data:
            features = await self.extract_features_batch(img_data, model)
            features_list.append(features)

        return {"features": features_list, "count": len(features_list)}

    @app.post("/embedding/true-batch")
    async def embedding_true_batch(
        self,
        images: List[UploadFile] = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """True batch processing - process all images in a single batch."""
        images_data = [await img.read() for img in images]
        batch_tensors = self.preprocess_batch(images_data)

        with torch.no_grad():
            if model == "osnet":
                features = self.os_model(batch_tensors)
                features_list = features.cpu().numpy().tolist()
            else:
                features = self.lmbn_model(batch_tensors)  # Shape: [batch, 512, 7]
                # Average across the 7 components to get [batch, 512] features
                features_avg = features.mean(dim=2)  # Shape: [batch, 512]
                features_list = features_avg.cpu().numpy().tolist()

        return {
            "features": features_list,
            "count": len(features_list),
            "shape": [
                len(features_list),
                len(features_list[0]) if features_list else 0,
            ],
        }

    @app.post("/similarity")
    async def calculate_similarity(
        self,
        image1: UploadFile = File(...),
        image2: UploadFile = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """Calculate cosine similarity between two images."""
        try:
            # Read image data
            image1_data = await image1.read()
            image2_data = await image2.read()

            # Extract features for both images
            features1 = await self.extract_features_single(image1_data, model)
            features2 = await self.extract_features_single(image2_data, model)

            # Convert to tensors for cosine similarity calculation
            tensor1 = torch.tensor(features1, dtype=torch.float32)
            tensor2 = torch.tensor(features2, dtype=torch.float32)

            # Calculate cosine similarity
            similarity = F.cosine_similarity(
                tensor1.unsqueeze(0), tensor2.unsqueeze(0)
            ).item()

            return {
                "similarity": similarity,
                "model_used": model,
                "image1_features_shape": len(features1),
                "image2_features_shape": len(features2),
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error calculating similarity: {str(e)}"
            )

    @app.get("/health")
    async def health_check(self):
        return {"status": "healthy", "device": self.device}


# Deployment binding
feature_extraction_app = FeatureExtractionService.bind()
