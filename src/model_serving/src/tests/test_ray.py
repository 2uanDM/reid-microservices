import asyncio
import concurrent.futures
import io
import statistics
import time
from typing import Any, Dict

import httpx
import numpy as np
from PIL import Image


class FeatureExtractionBenchmark:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def generate_test_image(self, width: int = 128, height: int = 256) -> bytes:
        """Generate a random test image."""
        # Create random RGB image
        img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, "RGB")

        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return img_bytes.getvalue()

    async def test_single_request(self) -> Dict[str, Any]:
        """Test single image processing latency."""
        image_data = self.generate_test_image()

        start_time = time.perf_counter()

        files = {"image": ("test.jpg", image_data, "image/jpeg")}
        response = await self.client.post(f"{self.base_url}/embedding", files=files)
        result = response.json()

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds

        return {
            "latency_ms": latency,
            "features_count": len(result["features"]),
            "status": "success" if response.status_code == 200 else "failed",
        }

    async def test_batch_request(self, batch_size: int = 8) -> Dict[str, Any]:
        """Test batch processing."""
        images_data = [self.generate_test_image() for _ in range(batch_size)]

        start_time = time.perf_counter()

        files = [
            ("images", (f"test_{i}.jpg", img_data, "image/jpeg"))
            for i, img_data in enumerate(images_data)
        ]
        response = await self.client.post(
            f"{self.base_url}/embedding/batch", files=files
        )
        result = response.json()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        return {
            "total_time_ms": total_time,
            "batch_size": batch_size,
            "time_per_image_ms": total_time / batch_size,
            "images_processed": result["count"],
            "status": "success" if response.status_code == 200 else "failed",
        }

    async def test_true_batch_request(self, batch_size: int = 8) -> Dict[str, Any]:
        """Test true batch processing endpoint."""
        images_data = [self.generate_test_image() for _ in range(batch_size)]

        start_time = time.perf_counter()

        files = [
            ("images", (f"test_{i}.jpg", img_data, "image/jpeg"))
            for i, img_data in enumerate(images_data)
        ]
        response = await self.client.post(
            f"{self.base_url}/embedding/true-batch", files=files
        )
        result = response.json()

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        return {
            "total_time_ms": total_time,
            "batch_size": batch_size,
            "time_per_image_ms": total_time / batch_size,
            "images_processed": result["count"],
            "status": "success" if response.status_code == 200 else "failed",
        }

    async def test_concurrent_requests(self, num_requests: int = 10) -> Dict[str, Any]:
        """Test concurrent single requests."""
        start_time = time.perf_counter()

        tasks = [self.test_single_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        total_time = (end_time - start_time) * 1000  # Convert to milliseconds

        latencies = [r["latency_ms"] for r in results if r["status"] == "success"]

        return {
            "total_time_ms": total_time,
            "num_requests": num_requests,
            "successful_requests": len(latencies),
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "throughput_rps": num_requests / (total_time / 1000)
            if total_time > 0
            else 0,
        }

    async def run_benchmark_suite(self):
        """Run complete benchmark suite."""
        print("🚀 Starting Feature Extraction Service Benchmark")
        print("=" * 60)

        # Health check
        try:
            response = await self.client.get(f"{self.base_url}/health")
            health = response.json()
            print(f"✅ Service health: {health}")
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return

        print("\n1. Single Request Latency Test")
        print("-" * 40)
        single_results = []
        for i in range(10):
            result = await self.test_single_request()
            single_results.append(result)
            print(f"Request {i + 1}: {result['latency_ms']:.2f}ms")

        latencies = [r["latency_ms"] for r in single_results]
        print(f"Average latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")
        print(f"Min latency: {min(latencies):.2f}ms")
        print(f"Max latency: {max(latencies):.2f}ms")

        print("\n2. Batch Processing Test")
        print("-" * 40)
        for batch_size in [2, 4, 8, 16]:
            result = await self.test_batch_request(batch_size)
            print(
                f"Batch size {batch_size}: {result['total_time_ms']:.2f}ms total, "
                f"{result['time_per_image_ms']:.2f}ms per image"
            )

        print("\n3. True Batch Processing Test")
        print("-" * 40)
        for batch_size in [2, 4, 8, 16]:
            result = await self.test_true_batch_request(batch_size)
            print(
                f"True batch size {batch_size}: {result['total_time_ms']:.2f}ms total, "
                f"{result['time_per_image_ms']:.2f}ms per image"
            )

        print("\n4. Concurrent Requests Test")
        print("-" * 40)
        for num_concurrent in [2, 5, 10, 20]:
            result = await self.test_concurrent_requests(num_concurrent)
            print(
                f"Concurrent {num_concurrent}: {result['throughput_rps']:.2f} RPS, "
                f"avg latency: {result['avg_latency_ms']:.2f}ms"
            )

        print("\n5. Stress Test")
        print("-" * 40)
        stress_result = await self.test_concurrent_requests(50)
        print(f"Stress test (50 concurrent): {stress_result['throughput_rps']:.2f} RPS")
        print(
            f"Success rate: {stress_result['successful_requests']}/50 "
            f"({stress_result['successful_requests'] / 50 * 100:.1f}%)"
        )
        print(
            f"Latency stats: min={stress_result['min_latency_ms']:.2f}ms, "
            f"avg={stress_result['avg_latency_ms']:.2f}ms, "
            f"max={stress_result['max_latency_ms']:.2f}ms"
        )


def run_threaded_benchmark(num_threads: int = 4, requests_per_thread: int = 10):
    """Run benchmark with multiple threads for maximum load testing."""
    print(
        f"\n6. Multi-threaded Load Test ({num_threads} threads, {requests_per_thread} requests each)"
    )
    print("-" * 60)

    def thread_worker(thread_id: int):
        async def worker():
            async with FeatureExtractionBenchmark() as benchmark:
                results = []
                for i in range(requests_per_thread):
                    result = await benchmark.test_single_request()
                    results.append(result)
                return results

        return asyncio.run(worker())

    start_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(thread_worker, i) for i in range(num_threads)]
        all_results = []
        for future in concurrent.futures.as_completed(futures):
            all_results.extend(future.result())

    end_time = time.perf_counter()
    total_time = end_time - start_time

    successful_results = [r for r in all_results if r["status"] == "success"]
    latencies = [r["latency_ms"] for r in successful_results]

    total_requests = num_threads * requests_per_thread
    print(f"Total requests: {total_requests}")
    print(f"Successful requests: {len(successful_results)}")
    print(f"Success rate: {len(successful_results) / total_requests * 100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(successful_results) / total_time:.2f} RPS")
    if latencies:
        print(
            f"Latency - Min: {min(latencies):.2f}ms, "
            f"Avg: {statistics.mean(latencies):.2f}ms, "
            f"Max: {max(latencies):.2f}ms"
        )


async def main():
    """Main benchmark function."""
    async with FeatureExtractionBenchmark() as benchmark:
        await benchmark.run_benchmark_suite()

    # Run threaded load test
    run_threaded_benchmark()

    print("\n🎉 Benchmark completed!")


if __name__ == "__main__":
    # Install required packages if not available:
    # pip install httpx pillow numpy

    print("Feature Extraction Service Benchmark Tool")
    print("Make sure your service is running on http://localhost:8000")
    print("Press Ctrl+C to stop the benchmark at any time")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark stopped by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
