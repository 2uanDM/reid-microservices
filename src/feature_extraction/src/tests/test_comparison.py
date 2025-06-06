import asyncio
import io
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import httpx
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add the current working directory to Python path
sys.path.append(os.getcwd())
from src.models.osnet import osnet_x1_0


class LocalFeatureExtractor:
    """Local feature extractor for comparison."""

    def __init__(self, device=None):
        self.device = (
            device
            if device is not None
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        print(f"Local extractor using device: {self.device}")

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

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup the model."""
        print("Warming up local model...")
        dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)
        print("Local model warmup completed")

    def preprocess(self, images_bytes: List[bytes]) -> torch.Tensor:
        """Preprocess a list of image bytes."""
        processed_images = []
        for img_bytes in images_bytes:
            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0)
            processed_images.append(img_tensor)

        batch_tensors = torch.cat(processed_images, dim=0)
        return batch_tensors.to(self.device)

    @torch.no_grad()
    def extract_features(self, images_bytes: List[bytes]) -> np.ndarray:
        """Extract features from images."""
        batch_tensors = self.preprocess(images_bytes)
        features = self.model(batch_tensors)
        return features.detach().cpu().numpy()


class RayServeClient:
    """Ray Serve client for comparison."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def health_check(self) -> bool:
        """Check if Ray Serve is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False

    async def extract_features_single(self, image_bytes: bytes) -> List[float]:
        """Extract features from a single image."""
        files = {"image": ("test.jpg", image_bytes, "image/jpeg")}
        response = await self.client.post(f"{self.base_url}/embedding", files=files)
        result = response.json()
        return result["features"]

    async def extract_features_batch(
        self, images_bytes: List[bytes]
    ) -> List[List[float]]:
        """Extract features from multiple images using true batch endpoint."""
        files = [
            ("images", (f"test_{i}.jpg", img_bytes, "image/jpeg"))
            for i, img_bytes in enumerate(images_bytes)
        ]
        response = await self.client.post(
            f"{self.base_url}/embedding/true-batch", files=files
        )
        result = response.json()
        return result["features"]


class ComparisonBenchmark:
    """Comprehensive benchmark comparing local vs Ray Serve."""

    def __init__(self):
        self.local_extractor = None
        self.ray_client = None

    def generate_test_image(self, width: int = 128, height: int = 256) -> bytes:
        """Generate a random test image."""
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, "RGB")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes.getvalue()

    def generate_test_images(self, count: int) -> List[bytes]:
        """Generate multiple test images."""
        return [self.generate_test_image() for _ in range(count)]

    def benchmark_local_single(
        self, images: List[bytes], num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark local single image processing."""
        latencies = []

        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.local_extractor.extract_features([images[0]])
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    def benchmark_local_batch(
        self, images: List[bytes], batch_size: int, num_runs: int = 50
    ) -> Dict[str, Any]:
        """Benchmark local batch processing."""
        latencies = []
        batch_images = images[:batch_size] * (batch_size // len(images) + 1)
        batch_images = batch_images[:batch_size]

        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = self.local_extractor.extract_features(batch_images)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "per_image_ms": statistics.mean(latencies) / batch_size,
            "batch_size": batch_size,
        }

    async def benchmark_ray_single(
        self, images: List[bytes], num_runs: int = 100
    ) -> Dict[str, Any]:
        """Benchmark Ray Serve single image processing."""
        latencies = []

        async with RayServeClient() as client:
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = await client.extract_features_single(images[0])
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)

        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        }

    async def benchmark_ray_batch(
        self, images: List[bytes], batch_size: int, num_runs: int = 50
    ) -> Dict[str, Any]:
        """Benchmark Ray Serve batch processing."""
        latencies = []
        batch_images = images[:batch_size] * (batch_size // len(images) + 1)
        batch_images = batch_images[:batch_size]

        async with RayServeClient() as client:
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = await client.extract_features_batch(batch_images)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)

        return {
            "mean_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "per_image_ms": statistics.mean(latencies) / batch_size,
            "batch_size": batch_size,
        }

    async def benchmark_ray_concurrent(
        self, images: List[bytes], num_concurrent: int
    ) -> Dict[str, Any]:
        """Benchmark Ray Serve concurrent processing - This is where Ray shines!"""
        start_time = time.perf_counter()

        async with RayServeClient() as client:
            tasks = [
                client.extract_features_single(images[i % len(images)])
                for i in range(num_concurrent)
            ]
            results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        return {
            "total_time_ms": total_time,
            "num_requests": num_concurrent,
            "avg_latency_ms": total_time / num_concurrent,
            "throughput_rps": num_concurrent / (total_time / 1000),
            "successful_requests": len(results),
        }

    def benchmark_local_concurrent_simulation(
        self, images: List[bytes], num_concurrent: int
    ) -> Dict[str, Any]:
        """Simulate concurrent processing with local extractor (sequential processing)."""
        start_time = time.perf_counter()

        # Local processing is sequential, no real concurrency
        for i in range(num_concurrent):
            _ = self.local_extractor.extract_features([images[i % len(images)]])

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000

        return {
            "total_time_ms": total_time,
            "num_requests": num_concurrent,
            "avg_latency_ms": total_time / num_concurrent,
            "throughput_rps": num_concurrent / (total_time / 1000),
            "note": "Sequential processing (no real concurrency)",
        }

    async def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark suite."""
        print("🚀 RAY SERVE vs LOCAL INFERENCE COMPARISON")
        print("=" * 70)

        # Initialize local extractor
        print("Initializing local feature extractor...")
        self.local_extractor = LocalFeatureExtractor()

        # Check Ray Serve availability
        print("Checking Ray Serve availability...")
        async with RayServeClient() as client:
            ray_available = await client.health_check()

        if not ray_available:
            print("❌ Ray Serve is not available. Please start the service first.")
            print("Run: serve run src.feature_extraction.main:feature_extraction_app")
            return

        print("✅ Ray Serve is available")

        # Generate test data
        test_images = self.generate_test_images(10)

        print("\n📊 BENCHMARK RESULTS")
        print("=" * 70)

        # 1. Single Image Latency Comparison
        print("\n1. 🔍 SINGLE IMAGE LATENCY (100 runs)")
        print("-" * 50)

        local_single = self.benchmark_local_single(test_images)
        ray_single = await self.benchmark_ray_single(test_images)

        print(
            f"Local  - Mean: {local_single['mean_latency_ms']:.2f}ms, "
            f"P95: {local_single['p95_latency_ms']:.2f}ms"
        )
        print(
            f"Ray    - Mean: {ray_single['mean_latency_ms']:.2f}ms, "
            f"P95: {ray_single['p95_latency_ms']:.2f}ms"
        )

        latency_improvement = (
            (local_single["mean_latency_ms"] - ray_single["mean_latency_ms"])
            / local_single["mean_latency_ms"]
            * 100
        )
        print(
            f"🎯 Ray is {abs(latency_improvement):.1f}% {'faster' if latency_improvement > 0 else 'slower'} for single requests"
        )

        # 2. Batch Processing Comparison
        print("\n2. 📦 BATCH PROCESSING (50 runs each)")
        print("-" * 50)

        for batch_size in [4, 8, 16]:
            local_batch = self.benchmark_local_batch(test_images, batch_size)
            ray_batch = await self.benchmark_ray_batch(test_images, batch_size)

            print(f"\nBatch Size {batch_size}:")
            print(
                f"  Local - Total: {local_batch['mean_latency_ms']:.2f}ms, "
                f"Per Image: {local_batch['per_image_ms']:.2f}ms"
            )
            print(
                f"  Ray   - Total: {ray_batch['mean_latency_ms']:.2f}ms, "
                f"Per Image: {ray_batch['per_image_ms']:.2f}ms"
            )

            batch_improvement = (
                (local_batch["per_image_ms"] - ray_batch["per_image_ms"])
                / local_batch["per_image_ms"]
                * 100
            )
            print(
                f"  🎯 Ray is {abs(batch_improvement):.1f}% {'faster' if batch_improvement > 0 else 'slower'} per image"
            )

        # 3. CONCURRENCY TEST - This is where Ray really shines!
        print("\n3. 🚀 CONCURRENCY TEST - RAY'S SUPERPOWER!")
        print("-" * 50)

        for num_concurrent in [5, 10, 20, 50]:
            local_concurrent = self.benchmark_local_concurrent_simulation(
                test_images, num_concurrent
            )
            ray_concurrent = await self.benchmark_ray_concurrent(
                test_images, num_concurrent
            )

            print(f"\n{num_concurrent} Concurrent Requests:")
            print(
                f"  Local  - Total: {local_concurrent['total_time_ms']:.0f}ms, "
                f"Throughput: {local_concurrent['throughput_rps']:.1f} RPS"
            )
            print(
                f"  Ray    - Total: {ray_concurrent['total_time_ms']:.0f}ms, "
                f"Throughput: {ray_concurrent['throughput_rps']:.1f} RPS"
            )

            throughput_improvement = (
                (ray_concurrent["throughput_rps"] - local_concurrent["throughput_rps"])
                / local_concurrent["throughput_rps"]
                * 100
            )
            print(f"  🚀 Ray is {throughput_improvement:.0f}x faster in throughput!")

        # 4. Scalability Test
        print("\n4. 📈 SCALABILITY STRESS TEST")
        print("-" * 50)

        stress_test_sizes = [10, 25, 50, 100]
        for size in stress_test_sizes:
            print(f"\nTesting {size} concurrent requests...")

            # Only test Ray for high concurrency (local would be too slow)
            ray_stress = await self.benchmark_ray_concurrent(test_images, size)
            print(
                f"  Ray Serve - {ray_stress['throughput_rps']:.1f} RPS, "
                f"Avg latency: {ray_stress['avg_latency_ms']:.2f}ms"
            )

            if size <= 25:  # Only test local for smaller loads
                local_stress = self.benchmark_local_concurrent_simulation(
                    test_images, size
                )
                improvement = (
                    ray_stress["throughput_rps"] / local_stress["throughput_rps"]
                )
                print(f"  Local     - {local_stress['throughput_rps']:.1f} RPS")
                print(f"  🎯 Ray is {improvement:.1f}x faster!")
            else:
                print("  Local     - Too slow to test at this scale")

        print("\n🏆 RAY SERVE BENEFITS SUMMARY")
        print("=" * 70)
        print("✅ TRUE CONCURRENCY: Ray handles multiple requests simultaneously")
        print("✅ HORIZONTAL SCALING: Easy to add more replicas")
        print("✅ SMART BATCHING: Automatic request batching for throughput")
        print("✅ PRODUCTION READY: Built-in health checks, monitoring, auto-scaling")
        print("✅ GPU SHARING: Efficient GPU utilization across requests")
        print("✅ FAULT TOLERANCE: Request retries and error handling")
        print("✅ HTTP/REST API: Standard web service interface")
        print("✅ ZERO DOWNTIME UPDATES: Rolling deployments")

        print("\n💡 When to use Ray Serve:")
        print("  - Multiple concurrent users")
        print("  - Production web services")
        print("  - Need high throughput")
        print("  - Want easy scaling")
        print("  - Microservices architecture")


async def main():
    """Main benchmark function."""
    benchmark = ComparisonBenchmark()
    await benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    print("Feature Extraction: Ray Serve vs Local Comparison")
    print(
        "Make sure Ray Serve is running: serve run src.feature_extraction.main:feature_extraction_app"
    )
    print("Press Ctrl+C to stop the benchmark at any time\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark stopped by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
