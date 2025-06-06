import argparse
import io
import os
import sys
import time

sys.path.append(os.getcwd())
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.osnet import osnet_x1_0


def load_image(image_path):
    """Load an image from file path."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return img_bytes


def parse_args():
    parser = argparse.ArgumentParser(description="Feature extraction latency test")
    parser.add_argument(
        "--test_images",
        type=str,
        default="src/assets/test_images",
        help="Path to directory containing test images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for batch inference test"
    )
    parser.add_argument(
        "--num_runs", type=int, default=100, help="Number of runs for latency testing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda or cpu)",
    )
    return parser.parse_args()


class FeatureExtractor:
    def __init__(self, img_size=(128, 64), device=None):
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")
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

    def preprocess(self, images):
        processed_images = []

        for img in images:
            img = Image.open(io.BytesIO(img))
            img_tensor = self.transform(img).unsqueeze(0)
            processed_images.append(img_tensor)

        batch_tensors = torch.cat(processed_images, dim=0)
        return batch_tensors.to(self.device)

    @torch.no_grad()
    def extract_features(self, images):
        batch_tensors = self.preprocess(images)
        features = self.model(batch_tensors)
        return features.cpu().detach().numpy()


def test_single_image_latency(extractor, images, num_runs=100):
    """Test latency for single image inference."""
    latencies = []

    for _ in range(num_runs):
        img = images[0]  # Use the first image for consistency
        start_time = time.time()
        _ = extractor.extract_features([img])
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return latencies


def test_batch_latency(extractor, images, batch_size=16, num_runs=100):
    """Test latency for batch inference."""
    latencies = []

    # Create a batch by repeating images if needed
    batch_images = []
    for i in range(batch_size):
        batch_images.append(images[i % len(images)])

    for _ in range(num_runs):
        start_time = time.time()
        _ = extractor.extract_features(batch_images)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return latencies


def main():
    args = parse_args()

    # Load test images
    test_image_dir = Path(args.test_images)
    if not test_image_dir.exists():
        print(f"Test image directory {test_image_dir} does not exist")
        print("Creating placeholder test image")
        # Create a dummy image if no test images are available
        dummy_img = Image.new("RGB", (128, 256), color="red")
        buffer = io.BytesIO()
        dummy_img.save(buffer, format="JPEG")
        images = [buffer.getvalue()]
    else:
        image_paths = list(test_image_dir.glob("*.jpg")) + list(
            test_image_dir.glob("*.png")
        )
        images = [load_image(path) for path in image_paths]

    if not images:
        print("No images found. Creating placeholder test image")
        dummy_img = Image.new("RGB", (128, 256), color="red")
        buffer = io.BytesIO()
        dummy_img.save(buffer, format="JPEG")
        images = [buffer.getvalue()]

    # Initialize feature extractor
    extractor = FeatureExtractor(device=args.device)

    # Test single image latency
    single_latencies = test_single_image_latency(
        extractor, images, num_runs=args.num_runs
    )

    # Test batch latency
    batch_latencies = test_batch_latency(
        extractor, images, batch_size=args.batch_size, num_runs=args.num_runs
    )

    # Print results
    print("\n--- Latency Test Results ---")
    print("Single Image Inference:")
    print(f"  Mean: {np.mean(single_latencies):.2f} ms")
    print(f"  Median: {np.median(single_latencies):.2f} ms")
    print(f"  P95: {np.percentile(single_latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(single_latencies, 99):.2f} ms")

    print(f"\nBatch Inference (batch_size={args.batch_size}):")
    print(f"  Mean: {np.mean(batch_latencies):.2f} ms")
    print(f"  Median: {np.median(batch_latencies):.2f} ms")
    print(f"  P95: {np.percentile(batch_latencies, 95):.2f} ms")
    print(f"  P99: {np.percentile(batch_latencies, 99):.2f} ms")
    print(f"  Per Image: {np.mean(batch_latencies) / args.batch_size:.2f} ms")


if __name__ == "__main__":
    main()
