import os
import sys

import torch

sys.path.append(os.getcwd())

from src.models.efficientnet import GenderClassificationModel

if __name__ == "__main__":
    model = GenderClassificationModel()

    # Dummy image - create 3D array (batchsize, height, width, channels) for PIL
    dummy_input = torch.randn(1, 3, 224, 224).to(model.device)
    output = model(dummy_input)
    print(output.shape)
