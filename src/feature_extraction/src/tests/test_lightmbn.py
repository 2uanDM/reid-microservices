import os
import sys

import torch

sys.path.append(os.getcwd())

from src.models.lightmbn_n import LMBN_n

if __name__ == "__main__":
    print("Testing LMBN_n model...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = LMBN_n(num_classes=1000, feats=512, activation_map=False)
    model.eval()

    # Create dummy input (batch_size=4, channels=3, height=256, width=128)
    # This is a typical input size for person re-identification
    dummy_input = torch.randn(1, 3, 256, 128).to(device)

    print("Model initialized successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Input device: {dummy_input.device}")

    # Test inference mode
    print("\n--- Testing Inference Mode ---")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape (inference): {output.shape}")
        print(f"Output tensor info: {output.dtype}, device: {output.device}")
