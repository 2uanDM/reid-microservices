import torch
from fastapi import FastAPI
from ray import serve
from torchvision import transforms

from src.models.osnet import osnet_x1_0

app = FastAPI()


@serve.deployment(
    num_replicas=4,
    ray_actor_options={
        "num_cpus": 1,
        "num_gpus": 0.25,
    },  # 1 CPU, 0.25 GPU for each replica
)
@serve.ingress(app)  # Attach FastAPI app to the deployment
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

    def preprocess(self, img: Image.Image) -> torch.Tensor:
        pass
