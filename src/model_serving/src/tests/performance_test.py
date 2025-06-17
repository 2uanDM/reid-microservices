"""
Comprehensive Ray Serve Performance Testing Suite

This test suite compares Ray Serve performance across multiple dimensions:
1. Serving methods: Single request, batch processing, and true batch comparison
2. Request patterns: Concurrent and burst requests
3. Normal inference (for loop) vs Ray Serve comparison
4. Different deployment configurations (1 replica vs 4 replicas)

Metrics tracked:
- Requests per second (RPS)
- Latency (p50, p95, p99)
- GPU utilization and VRAM usage
- Throughput under different load patterns
"""

import asyncio
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.append(os.getcwd())

import aiohttp
import numpy as np
import ray
import torch
from PIL import Image
from ray import serve
from torchvision import transforms

# Import your models (adjust imports based on your actual model structure)
try:
    from src.models.efficientnet import GenderClassificationModel
    from src.models.lightmbn_n import LMBN_n
    from src.models.osnet import osnet_x1_0
except ImportError:
    print("Warning: Could not import models. Some tests may fail.")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    rps: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_latency: float
    total_requests: int
    failed_requests: int
    duration: float
    gpu_utilization: Optional[float] = None
    vram_used: Optional[float] = None
    vram_total: Optional[float] = None


class GPUMonitor:
    """Monitor GPU utilization and VRAM usage"""

    def __init__(self):
        self.monitoring = False
        self.gpu_stats = []

    def start_monitoring(self):
        """Start GPU monitoring in background"""
        self.monitoring = True
        self.gpu_stats = []

    def stop_monitoring(self):
        """Stop GPU monitoring and return average stats"""
        self.monitoring = False
        if not self.gpu_stats:
            return None, None, None

        avg_utilization = np.mean([stat["utilization"] for stat in self.gpu_stats])
        avg_vram_used = np.mean([stat["vram_used"] for stat in self.gpu_stats])
        max_vram_total = max([stat["vram_total"] for stat in self.gpu_stats])

        return avg_utilization, avg_vram_used, max_vram_total

    def _get_gpu_stats(self):
        """Get current GPU stats using nvidia-ml-py or fallback methods"""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu

            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used = mem_info.used / 1024**3  # Convert to GB
            vram_total = mem_info.total / 1024**3  # Convert to GB

            return {
                "utilization": gpu_util,
                "vram_used": vram_used,
                "vram_total": vram_total,
            }
        except ImportError:
            # Fallback to nvidia-smi if pynvml not available
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                line = result.stdout.strip().split("\n")[0]
                util, mem_used, mem_total = line.split(", ")

                return {
                    "utilization": float(util),
                    "vram_used": float(mem_used) / 1024,  # Convert MB to GB
                    "vram_total": float(mem_total) / 1024,  # Convert MB to GB
                }
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

    async def monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            stats = self._get_gpu_stats()
            if stats:
                self.gpu_stats.append(stats)
            await asyncio.sleep(0.5)  # Monitor every 500ms


class NormalInferenceModel:
    """Normal inference model for comparison (without Ray Serve)"""

    def __init__(self, device: str = None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Initializing normal inference model on device: {self.device}")

        # Initialize models (same as Ray Serve version)
        self.os_model = osnet_x1_0(
            num_classes=1000,
            loss="softmax",
            pretrained=True,
        ).to(self.device)
        self.os_model.eval()

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

        # Warmup
        self._warmup()

    def _warmup(self):
        """Warmup the model"""
        print("Warming up normal inference model...")
        dummy_input = torch.randn(1, 3, 256, 128).to(self.device)
        with torch.no_grad():
            _ = self.os_model(dummy_input)
            _ = self.lmbn_model(dummy_input)
        print("Normal inference model warmup completed")

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess image bytes to tensor"""
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0)
        return img_tensor.to(self.device)

    @torch.no_grad()
    def extract_features(self, image_bytes: bytes, model: str = "osnet") -> List[float]:
        """Extract features from image"""
        img_tensor = self.preprocess_image(image_bytes)

        if model == "osnet":
            features = self.os_model(img_tensor)
            return features.detach().cpu().numpy().flatten().tolist()
        else:  # lmbn
            features = self.lmbn_model(img_tensor)
            features_avg = features.mean(dim=2)
            return features_avg.detach().cpu().numpy().flatten().tolist()

    @torch.no_grad()
    def extract_features_batch(
        self, images_bytes: List[bytes], model: str = "osnet"
    ) -> List[List[float]]:
        """Extract features from batch of images"""
        batch_tensors = []
        for img_bytes in images_bytes:
            img_tensor = self.preprocess_image(img_bytes)
            batch_tensors.append(img_tensor)

        batch_input = torch.cat(batch_tensors, dim=0)

        if model == "osnet":
            features = self.os_model(batch_input)
            return features.detach().cpu().numpy().tolist()
        else:  # lmbn
            features = self.lmbn_model(batch_input)
            features_avg = features.mean(dim=2)
            return features_avg.detach().cpu().numpy().tolist()


class RayServePerformanceTester:
    """Main performance testing class"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.gpu_monitor = GPUMonitor()
        self.test_images = []
        self.normal_model = None

    def setup_test_images(self, num_images: int = 100):
        """Generate synthetic test images"""
        print(f"Generating {num_images} test images...")
        self.test_images = []

        for i in range(num_images):
            # Create a random RGB image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)

            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)

            self.test_images.append(img_bytes.getvalue())

        print(f"Generated {len(self.test_images)} test images")

    def calculate_metrics(
        self, latencies: List[float], failed_requests: int, duration: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics from latency data"""
        if not latencies:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, failed_requests, duration)

        total_requests = len(latencies) + failed_requests
        successful_requests = len(latencies)
        rps = successful_requests / duration if duration > 0 else 0

        latencies_np = np.array(latencies)

        # Get GPU stats
        gpu_util, vram_used, vram_total = self.gpu_monitor.stop_monitoring()

        return PerformanceMetrics(
            rps=rps,
            latency_p50=np.percentile(latencies_np, 50),
            latency_p95=np.percentile(latencies_np, 95),
            latency_p99=np.percentile(latencies_np, 99),
            avg_latency=np.mean(latencies_np),
            total_requests=total_requests,
            failed_requests=failed_requests,
            duration=duration,
            gpu_utilization=gpu_util,
            vram_used=vram_used,
            vram_total=vram_total,
        )

    # Test 1: Single Request Performance
    async def test_single_requests(
        self, num_requests: int = 100, endpoint: str = "/embedding"
    ) -> PerformanceMetrics:
        """Test single request performance"""
        print(f"Testing single requests: {num_requests} requests to {endpoint}")

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.gpu_monitor.monitor_loop())

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                request_start = time.time()

                try:
                    data = aiohttp.FormData()
                    data.add_field(
                        "image",
                        self.test_images[i % len(self.test_images)],
                        filename=f"test_{i}.jpg",
                        content_type="image/jpeg",
                    )
                    data.add_field("model", "osnet")

                    async with session.post(
                        f"{self.base_url}{endpoint}", data=data
                    ) as response:
                        if response.status == 200:
                            await response.json()
                            latencies.append(time.time() - request_start)
                        else:
                            failed_requests += 1
                            print(f"Request {i} failed with status {response.status}")

                except Exception as e:
                    failed_requests += 1
                    print(f"Request {i} failed with error: {e}")

        duration = time.time() - start_time
        monitor_task.cancel()

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 2: Batch Processing Performance
    async def test_batch_requests(
        self, num_requests: int = 100, endpoint: str = "/embedding/batch"
    ) -> PerformanceMetrics:
        """Test batch processing performance"""
        print(f"Testing batch requests: {num_requests} requests to {endpoint}")

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.gpu_monitor.monitor_loop())

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                request_start = time.time()

                try:
                    data = aiohttp.FormData()
                    data.add_field(
                        "image",
                        self.test_images[i % len(self.test_images)],
                        filename=f"test_{i}.jpg",
                        content_type="image/jpeg",
                    )
                    data.add_field("model", "osnet")

                    async with session.post(
                        f"{self.base_url}{endpoint}", data=data
                    ) as response:
                        if response.status == 200:
                            await response.json()
                            latencies.append(time.time() - request_start)
                        else:
                            failed_requests += 1
                            print(f"Request {i} failed with status {response.status}")

                except Exception as e:
                    failed_requests += 1
                    print(f"Request {i} failed with error: {e}")

        duration = time.time() - start_time
        monitor_task.cancel()

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 3: True Batch Performance
    async def test_true_batch_requests(
        self, num_batches: int = 10, batch_size: int = 10
    ) -> PerformanceMetrics:
        """Test true batch processing performance"""
        print(
            f"Testing true batch requests: {num_batches} batches of size {batch_size}"
        )

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.gpu_monitor.monitor_loop())

        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            for batch_idx in range(num_batches):
                request_start = time.time()

                try:
                    data = aiohttp.FormData()

                    # Add multiple images to the batch
                    for img_idx in range(batch_size):
                        img_index = (batch_idx * batch_size + img_idx) % len(
                            self.test_images
                        )
                        data.add_field(
                            "images",
                            self.test_images[img_index],
                            filename=f"test_{img_index}.jpg",
                            content_type="image/jpeg",
                        )

                    data.add_field("model", "osnet")

                    async with session.post(
                        f"{self.base_url}/embedding/true-batch", data=data
                    ) as response:
                        if response.status == 200:
                            await response.json()
                            latencies.append(time.time() - request_start)
                        else:
                            failed_requests += 1
                            print(
                                f"Batch {batch_idx} failed with status {response.status}"
                            )

                except Exception as e:
                    failed_requests += 1
                    print(f"Batch {batch_idx} failed with error: {e}")

        duration = time.time() - start_time
        monitor_task.cancel()

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 4: Concurrent Requests
    async def test_concurrent_requests(
        self,
        num_requests: int = 100,
        concurrency: int = 10,
        endpoint: str = "/embedding",
    ) -> PerformanceMetrics:
        """Test concurrent request performance"""
        print(
            f"Testing concurrent requests: {num_requests} requests with concurrency {concurrency} to {endpoint}"
        )

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.gpu_monitor.monitor_loop())

        start_time = time.time()

        async def make_request(session, request_id):
            request_start = time.time()
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "image",
                    self.test_images[request_id % len(self.test_images)],
                    filename=f"test_{request_id}.jpg",
                    content_type="image/jpeg",
                )
                data.add_field("model", "osnet")

                async with session.post(
                    f"{self.base_url}{endpoint}", data=data
                ) as response:
                    if response.status == 200:
                        await response.json()
                        return time.time() - request_start
                    else:
                        print(
                            f"Request {request_id} failed with status {response.status}"
                        )
                        return None
            except Exception as e:
                print(f"Request {request_id} failed with error: {e}")
                return None

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(session, request_id):
            async with semaphore:
                return await make_request(session, request_id)

        async with aiohttp.ClientSession() as session:
            tasks = [bounded_request(session, i) for i in range(num_requests)]
            results = await asyncio.gather(*tasks)

            for result in results:
                if result is not None:
                    latencies.append(result)
                else:
                    failed_requests += 1

        duration = time.time() - start_time
        monitor_task.cancel()

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 5: Burst Requests
    async def test_burst_requests(
        self,
        burst_size: int = 50,
        num_bursts: int = 5,
        burst_interval: int = 2,
        endpoint: str = "/embedding",
    ) -> PerformanceMetrics:
        """Test burst request performance"""
        print(
            f"Testing burst requests: {num_bursts} bursts of {burst_size} requests each"
        )

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()
        monitor_task = asyncio.create_task(self.gpu_monitor.monitor_loop())

        start_time = time.time()

        async def make_burst_request(session, request_id):
            request_start = time.time()
            try:
                data = aiohttp.FormData()
                data.add_field(
                    "image",
                    self.test_images[request_id % len(self.test_images)],
                    filename=f"test_{request_id}.jpg",
                    content_type="image/jpeg",
                )
                data.add_field("model", "osnet")

                async with session.post(
                    f"{self.base_url}{endpoint}", data=data
                ) as response:
                    if response.status == 200:
                        await response.json()
                        return time.time() - request_start
                    else:
                        return None
            except Exception:
                return None

        async with aiohttp.ClientSession() as session:
            for burst_idx in range(num_bursts):
                print(f"Executing burst {burst_idx + 1}/{num_bursts}")

                # Send all requests in the burst simultaneously
                tasks = []
                for i in range(burst_size):
                    request_id = burst_idx * burst_size + i
                    tasks.append(make_burst_request(session, request_id))

                results = await asyncio.gather(*tasks)

                for result in results:
                    if result is not None:
                        latencies.append(result)
                    else:
                        failed_requests += 1

                # Wait before next burst (except for the last one)
                if burst_idx < num_bursts - 1:
                    await asyncio.sleep(burst_interval)

        duration = time.time() - start_time
        monitor_task.cancel()

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 6: Normal Inference vs Ray Serve Comparison
    def test_normal_inference_sequential(
        self, num_requests: int = 100
    ) -> PerformanceMetrics:
        """Test normal inference (for loop) performance"""
        print(f"Testing normal inference: {num_requests} sequential requests")

        if self.normal_model is None:
            self.normal_model = NormalInferenceModel()

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()

        start_time = time.time()

        for i in range(num_requests):
            request_start = time.time()
            try:
                features = self.normal_model.extract_features(
                    self.test_images[i % len(self.test_images)], model="osnet"
                )
                latencies.append(time.time() - request_start)
            except Exception as e:
                failed_requests += 1
                print(f"Normal inference request {i} failed: {e}")

        duration = time.time() - start_time

        return self.calculate_metrics(latencies, failed_requests, duration)

    def test_normal_inference_batch(
        self, num_batches: int = 10, batch_size: int = 10
    ) -> PerformanceMetrics:
        """Test normal inference batch processing"""
        print(
            f"Testing normal inference batch: {num_batches} batches of size {batch_size}"
        )

        if self.normal_model is None:
            self.normal_model = NormalInferenceModel()

        latencies = []
        failed_requests = 0

        self.gpu_monitor.start_monitoring()

        start_time = time.time()

        for batch_idx in range(num_batches):
            request_start = time.time()
            try:
                # Prepare batch
                batch_images = []
                for img_idx in range(batch_size):
                    img_index = (batch_idx * batch_size + img_idx) % len(
                        self.test_images
                    )
                    batch_images.append(self.test_images[img_index])

                features = self.normal_model.extract_features_batch(
                    batch_images, model="osnet"
                )
                latencies.append(time.time() - request_start)
            except Exception as e:
                failed_requests += 1
                print(f"Normal inference batch {batch_idx} failed: {e}")

        duration = time.time() - start_time

        return self.calculate_metrics(latencies, failed_requests, duration)

    # Test 7: Different Deployment Configurations
    async def run_deployment_comparison(self):
        """Compare different Ray Serve deployment configurations"""
        print("Running deployment configuration comparison...")

        results = {}

        # Test with 1 replica
        print("\n=== Testing with 1 replica ===")
        await self.setup_ray_serve_deployment(num_replicas=1)
        await asyncio.sleep(5)  # Wait for deployment to be ready

        results["1_replica"] = {
            "single_requests": await self.test_single_requests(num_requests=50),
            "batch_requests": await self.test_batch_requests(num_requests=50),
            "concurrent_requests": await self.test_concurrent_requests(
                num_requests=50, concurrency=5
            ),
        }

        # Test with 4 replicas
        print("\n=== Testing with 4 replicas ===")
        await self.setup_ray_serve_deployment(num_replicas=4)
        await asyncio.sleep(10)  # Wait for deployment to be ready

        results["4_replicas"] = {
            "single_requests": await self.test_single_requests(num_requests=50),
            "batch_requests": await self.test_batch_requests(num_requests=50),
            "concurrent_requests": await self.test_concurrent_requests(
                num_requests=50, concurrency=20
            ),
        }

        return results

    async def setup_ray_serve_deployment(self, num_replicas: int = 1):
        """Setup Ray Serve deployment with specified number of replicas"""
        print(f"Setting up Ray Serve deployment with {num_replicas} replicas...")

        # This is a placeholder - you'll need to implement the actual deployment logic
        # based on your Ray Serve setup. This might involve:
        # 1. Stopping existing deployment
        # 2. Starting new deployment with specified replica count
        # 3. Waiting for deployment to be ready

        # Example implementation (adjust based on your setup):
        try:
            # Stop existing deployment
            serve.shutdown()
            await asyncio.sleep(2)

            # Start new deployment
            ray.init(ignore_reinit_error=True)

            # Import and deploy your ModelService with specified replicas
            from src.main import ModelService

            serve.run(
                ModelService.bind(),
                name="model_service",
                route_prefix="/",
                num_replicas=num_replicas,
            )

        except Exception as e:
            print(f"Error setting up deployment: {e}")

    def print_metrics(self, name: str, metrics: PerformanceMetrics):
        """Print formatted performance metrics"""
        print(f"\n{'=' * 60}")
        print(f"Performance Metrics: {name}")
        print(f"{'=' * 60}")
        print(f"Total Requests: {metrics.total_requests}")
        print(f"Failed Requests: {metrics.failed_requests}")
        print(
            f"Success Rate: {((metrics.total_requests - metrics.failed_requests) / metrics.total_requests * 100):.2f}%"
        )
        print(f"Duration: {metrics.duration:.2f}s")
        print(f"Requests/sec: {metrics.rps:.2f}")
        print(f"Average Latency: {metrics.avg_latency * 1000:.2f}ms")
        print(f"P50 Latency: {metrics.latency_p50 * 1000:.2f}ms")
        print(f"P95 Latency: {metrics.latency_p95 * 1000:.2f}ms")
        print(f"P99 Latency: {metrics.latency_p99 * 1000:.2f}ms")

        if metrics.gpu_utilization is not None:
            print(f"GPU Utilization: {metrics.gpu_utilization:.1f}%")
        if metrics.vram_used is not None:
            print(f"VRAM Used: {metrics.vram_used:.2f}GB / {metrics.vram_total:.2f}GB")

    def save_results_to_json(
        self, results: Dict, filename: str = "performance_results.json"
    ):
        """Save results to JSON file"""
        # Convert PerformanceMetrics objects to dictionaries
        json_results = {}

        def convert_metrics(obj):
            if isinstance(obj, PerformanceMetrics):
                return obj.__dict__
            elif isinstance(obj, dict):
                return {k: convert_metrics(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_metrics(item) for item in obj]
            else:
                return obj

        json_results = convert_metrics(results)

        with open(filename, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nResults saved to {filename}")

    async def run_comprehensive_test_suite(self):
        """Run the complete performance test suite"""
        print("Starting Comprehensive Ray Serve Performance Test Suite")
        print("=" * 80)

        # Setup test data
        self.setup_test_images(num_images=500)

        results = {}

        # Test 1: Single Request Methods Comparison
        print("\n" + "=" * 80)
        print("TEST 1: Serving Methods Comparison")
        print("=" * 80)

        results["serving_methods"] = {}

        # Single requests
        results["serving_methods"]["single_requests"] = await self.test_single_requests(
            num_requests=500, endpoint="/embedding"
        )
        self.print_metrics(
            "Single Requests", results["serving_methods"]["single_requests"]
        )

        # Batch processing (dynamic batching)
        results["serving_methods"]["batch_processing"] = await self.test_batch_requests(
            num_requests=500, endpoint="/embedding/batch"
        )
        self.print_metrics(
            "Batch Processing", results["serving_methods"]["batch_processing"]
        )

        # True batch processing
        results["serving_methods"]["true_batch"] = await self.test_true_batch_requests(
            num_batches=10, batch_size=50
        )
        self.print_metrics(
            "True Batch Processing", results["serving_methods"]["true_batch"]
        )

        # Test 2: Request Patterns
        print("\n" + "=" * 80)
        print("TEST 2: Request Patterns (Concurrent & Burst)")
        print("=" * 80)

        results["request_patterns"] = {}

        # Concurrent requests
        results["request_patterns"]["concurrent"] = await self.test_concurrent_requests(
            num_requests=500, concurrency=50, endpoint="/embedding"
        )
        self.print_metrics(
            "Concurrent Requests", results["request_patterns"]["concurrent"]
        )

        # Burst requests
        results["request_patterns"]["burst"] = await self.test_burst_requests(
            burst_size=50, num_bursts=5, burst_interval=2, endpoint="/embedding"
        )
        self.print_metrics("Burst Requests", results["request_patterns"]["burst"])

        # Test 3: Normal Inference vs Ray Serve
        print("\n" + "=" * 80)
        print("TEST 3: Normal Inference vs Ray Serve Comparison")
        print("=" * 80)

        results["inference_comparison"] = {}

        # Normal inference (sequential)
        results["inference_comparison"]["normal_sequential"] = (
            self.test_normal_inference_sequential(num_requests=500)
        )
        self.print_metrics(
            "Normal Inference (Sequential)",
            results["inference_comparison"]["normal_sequential"],
        )

        # Normal inference (batch)
        results["inference_comparison"]["normal_batch"] = (
            self.test_normal_inference_batch(num_batches=5, batch_size=50)
        )
        self.print_metrics(
            "Normal Inference (Batch)", results["inference_comparison"]["normal_batch"]
        )

        # Ray Serve comparison (for reference)
        results["inference_comparison"][
            "rayserve_single"
        ] = await self.test_single_requests(num_requests=500, endpoint="/embedding")
        self.print_metrics(
            "Ray Serve (Single)", results["inference_comparison"]["rayserve_single"]
        )

        results["inference_comparison"][
            "rayserve_batch"
        ] = await self.test_batch_requests(
            num_requests=500, endpoint="/embedding/batch"
        )
        self.print_metrics(
            "Ray Serve (Batch)", results["inference_comparison"]["rayserve_batch"]
        )

        # Test 4: Deployment Configuration Comparison
        print("\n" + "=" * 80)
        print("TEST 4: Deployment Configuration Comparison")
        print("=" * 80)

        try:
            results["deployment_comparison"] = await self.run_deployment_comparison()

            for config, metrics in results["deployment_comparison"].items():
                print(f"\n--- {config.upper()} ---")
                for test_name, test_metrics in metrics.items():
                    self.print_metrics(f"{config} - {test_name}", test_metrics)

        except Exception as e:
            print(f"Deployment comparison failed: {e}")
            results["deployment_comparison"] = {"error": str(e)}

        # Save results
        self.save_results_to_json(results, "ray_serve_performance_results.json")

        # Print summary
        self.print_test_summary(results)

        return results

    def print_test_summary(self, results: Dict):
        """Print a summary of all test results"""
        print("\n" + "=" * 80)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 80)

        # Summary table
        summary_data = []

        try:
            # Serving methods comparison
            single_rps = results["serving_methods"]["single_requests"].rps
            batch_rps = results["serving_methods"]["batch_processing"].rps
            true_batch_rps = results["serving_methods"]["true_batch"].rps

            print("\n📊 Serving Methods Comparison:")
            print(f"  Single Requests:    {single_rps:.2f} RPS")
            print(f"  Batch Processing:   {batch_rps:.2f} RPS")
            print(f"  True Batch:         {true_batch_rps:.2f} RPS")

            # Request patterns
            concurrent_rps = results["request_patterns"]["concurrent"].rps
            burst_rps = results["request_patterns"]["burst"].rps

            print("\n🚀 Request Patterns:")
            print(f"  Concurrent (10):    {concurrent_rps:.2f} RPS")
            print(f"  Burst:              {burst_rps:.2f} RPS")

            # Inference comparison
            normal_seq_rps = results["inference_comparison"]["normal_sequential"].rps
            normal_batch_rps = results["inference_comparison"]["normal_batch"].rps
            rayserve_single_rps = results["inference_comparison"]["rayserve_single"].rps
            rayserve_batch_rps = results["inference_comparison"]["rayserve_batch"].rps

            print("\n⚡ Inference Comparison:")
            print(f"  Normal Sequential:  {normal_seq_rps:.2f} RPS")
            print(f"  Normal Batch:       {normal_batch_rps:.2f} RPS")
            print(f"  Ray Serve Single:   {rayserve_single_rps:.2f} RPS")
            print(f"  Ray Serve Batch:    {rayserve_batch_rps:.2f} RPS")

            # Performance insights
            print("\n💡 Key Insights:")
            if batch_rps > single_rps:
                improvement = ((batch_rps - single_rps) / single_rps) * 100
                print(f"  • Batch processing improves throughput by {improvement:.1f}%")

            if rayserve_batch_rps > normal_batch_rps:
                improvement = (
                    (rayserve_batch_rps - normal_batch_rps) / normal_batch_rps
                ) * 100
                print(
                    f"  • Ray Serve batch is {improvement:.1f}% faster than normal batch"
                )

            if concurrent_rps > single_rps:
                improvement = ((concurrent_rps - single_rps) / single_rps) * 100
                print(
                    f"  • Concurrent requests improve throughput by {improvement:.1f}%"
                )

        except Exception as e:
            print(f"Error generating summary: {e}")


# Main execution
async def main():
    """Main function to run the performance tests"""
    # Initialize the tester
    tester = RayServePerformanceTester(base_url="http://localhost:8000")

    # Check if Ray Serve is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{tester.base_url}/health") as response:
                if response.status != 200:
                    print("❌ Ray Serve is not running or not responding")
                    print("Please start your Ray Serve deployment first")
                    return
    except Exception as e:
        print(f"❌ Cannot connect to Ray Serve at {tester.base_url}")
        print(f"Error: {e}")
        print("Please ensure Ray Serve is running and accessible")
        return

    print("✅ Ray Serve is running and accessible")

    # Run the comprehensive test suite
    results = await tester.run_comprehensive_test_suite()

    print("\n🎉 Performance testing completed!")
    print("Check 'ray_serve_performance_results.json' for detailed results")


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import pynvml
    except ImportError:
        print("Installing pynvml for GPU monitoring...")
        subprocess.run(["pip", "install", "pynvml"], check=True)

    # Run the tests
    asyncio.run(main())
