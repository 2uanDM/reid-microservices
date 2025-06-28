import io
import traceback
from typing import List, Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from ray import serve
from torchvision import transforms

from src.models.efficientnet import GenderClassificationModel
from src.models.lightmbn_n import LMBN_n
from src.models.osnet import osnet_x1_0

# Create FastAPI app with explicit documentation configuration
app = FastAPI()


@serve.deployment
@serve.ingress(app)
class ModelService:
    def __init__(self, img_size: tuple = (128, 64), device: str | None = None):
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Initializing on device: {self.device}")

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

        self.gender_model = GenderClassificationModel()

        # Initialize transform
        self.transform = {
            "embedding": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((256, 128)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ],
            ),
            "classification": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # ImageNet defaults
                ],
            ),
        }

        self.img_size = img_size

        # Warmup the model
        self._warmup()

    def _warmup(self):
        """Warmup the model with dummy data to avoid cold start latency."""
        print("Warming up model...")
        dummy_input_osnet = torch.randn(1, 3, 256, 128).to(self.device)
        dummy_input_efficientnet = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.os_model(dummy_input_osnet)
            _ = self.lmbn_model(dummy_input_osnet)
            _ = self.gender_model(dummy_input_efficientnet)
        print("Model warmup completed")

    def preprocess_single(
        self,
        image_bytes: bytes,
        model: Literal["osnet", "lmbn", "efficientnet"] = "osnet",
    ) -> torch.Tensor:
        """Preprocess a single image."""
        if model == "lmbn" or model == "osnet":
            model_type = "embedding"
        else:
            model_type = "classification"

        # Preprocess image
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = self.transform[model_type](img).unsqueeze(0)
            return img_tensor.to(self.device)
        except Exception as e:
            print(traceback.format_exc())
            raise HTTPException(
                status_code=400, detail=f"Invalid image format: {str(e)}"
            )

    def preprocess_batch(
        self,
        images_bytes: List[bytes],
        models: List[Literal["osnet", "lmbn", "efficientnet"]] = ["osnet"],
    ) -> torch.Tensor:
        """Preprocess a batch of images."""
        model_types = [
            "embedding" if model == "lmbn" or model == "osnet" else "classification"
            for model in models
        ]

        processed_images = []
        for img_bytes, model_type in zip(images_bytes, model_types):
            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img_tensor = self.transform[model_type](img).unsqueeze(0)
                processed_images.append(img_tensor)
            except Exception as e:
                print(traceback.format_exc())
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
        img_tensor = self.preprocess_single(image_bytes, model)
        if model == "osnet":
            features = self.os_model(img_tensor)
            return features.detach().cpu().numpy().flatten().tolist()
        else:
            features = self.lmbn_model(img_tensor)  # Shape: [1, 512, 7]
            # Average across the 7 components to get [1s, 512] features
            features_avg = features.mean(dim=2)  # Shape: [1, 512]
            return features_avg.detach().cpu().numpy().flatten().tolist()

    @serve.batch(
        max_batch_size=32,
        batch_wait_timeout_s=0.01,  # Much shorter timeout (10ms)
    )
    @torch.no_grad()
    async def extract_features_batch(
        self,
        images_bytes_list: List[bytes],
        models: List[Literal["osnet", "lmbn"]] = ["osnet"],
    ) -> List[List[float]]:
        """Extract features from multiple images - optimized for throughput."""
        # images_bytes_list is already a list of bytes from different requests
        print(f"Dynamic batching: {len(images_bytes_list)} images")

        batch_tensors = self.preprocess_batch(images_bytes_list, models)

        if models[0] == "osnet":
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
        image: UploadFile = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """Batch endpoint - uses dynamic batching for throughput."""
        # This will trigger batching if multiple requests come simultaneously
        image_data = await image.read()
        features_list = await self.extract_features_batch(image_data, model)

        # extract_features_batch returns a list of feature vectors due to Ray Serve batching
        # For a single request, we get back a list with one element
        if len(features_list) == 1:
            # Single request case
            return {"features": features_list[0], "count": 1}
        else:
            # Multiple requests were batched together
            return {"features": features_list, "count": len(features_list)}

    @app.post("/embedding/true-batch")
    async def embedding_true_batch(
        self,
        images: List[UploadFile] = File(...),
        model: Literal["osnet", "lmbn"] = "osnet",
    ):
        """True batch processing - process all images in a single batch."""
        images_data = [await img.read() for img in images]
        batch_tensors = self.preprocess_batch(images_data, model)

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

    @app.post("/gender/classify")
    async def classify_gender(
        self,
        image: UploadFile = File(...),
    ):
        """Classify gender from input image."""
        try:
            # Read image data
            image_data = await image.read()

            # Preprocess image for EfficientNet
            img_tensor = self.preprocess_single(image_data, "efficientnet")

            # Get predictions
            with torch.no_grad():
                logits = self.gender_model(img_tensor)
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            # Map class to gender (assuming 0=female, 1=male)
            gender_labels = ["male", "female"]
            predicted_gender = gender_labels[predicted_class]

            return {
                "gender": predicted_gender,
                "confidence": confidence,
                "probabilities": {
                    "male": probabilities[0][0].item(),
                    "female": probabilities[0][1].item(),
                },
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error classifying gender: {str(e)}"
            )

    @app.get("/health")
    async def health_check(self):
        return {"status": "healthy", "device": self.device}


# Deployment binding
model_service = ModelService.bind()
