import torch
from efficientnet_pytorch import EfficientNet


class GenderClassificationModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet.from_pretrained("efficientnet-b0")

        # Gender classification setup
        num_classes = 2
        self.model._fc = torch.nn.Linear(self.model._fc.in_features, num_classes)
        self.model.load_state_dict(
            torch.load(
                "src/assets/models/efficientnet/best_model.pth",
                map_location=self.device,
            )
        )
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # Ensure the input tensor is on the same device as the model
        if isinstance(image, torch.Tensor):
            image = image.to(self.device)
        return self.model(image)
