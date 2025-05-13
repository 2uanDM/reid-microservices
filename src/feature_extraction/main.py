import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ray import serve
from torchvision import transforms

from src.models.osnet import osnet_x1_0

app = FastAPI()


@serve.deployment(
    num_replicas=4,
    ray_actor_options={
        "num_cpus": 2,
        "num_gpus": 0.25,
    },
    max_ongoing_requests=40,
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
        self.model = osnet_x1_0(
            num_classes=767,
            loss="softmax",
            pretrained=True,
        ).to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 128)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.model.eval()
        self.img_size = img_size

    def preprocess(self, images: list[bytes]) -> torch.Tensor:
        processed_images = []

        for img in images:
            img = Image.open(io.BytesIO(img))
            img_tensor = self.transform(img).unsqueeze(0)
            processed_images.append(img_tensor)

        batch_tensors = torch.cat(processed_images, dim=0)
        return batch_tensors.to(self.device)

    @serve.batch(
        max_batch_size=16,
        batch_wait_timeout_s=0.1,
    )
    @torch.no_grad()
    async def model_inference(self, images: list[bytes]) -> list[list[float]]:
        batch_tensors = self.preprocess(images)
        features = self.model(batch_tensors)
        all_features = features.cpu().detach().numpy().tolist()

        return all_features

    @app.post("/embedding")
    async def embedding(self, image: UploadFile = File(...)):
        image_data = await image.read()
        features = await self.model_inference(image_data)
        return features


feature_extraction_app = FeatureExtractionService.bind()
