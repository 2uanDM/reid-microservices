import asyncio
import io
import os
import statistics
import sys
import time
from typing import Any, Dict, List

import httpx
import numpy as np
from PIL import Image

# Add the current working directory to Python path
sys.path.append(os.getcwd())


class RayServeReplicaClient:
    """Ray Serve client for replica comparison testing."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=120.0)  # Longer timeout for heavy loads
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


class ReplicaComparisonBenchmark:
    """Benchmark to compare Ray Serve with different replica configurations."""

    def __init__(self):
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

    async def benchmark_concurrent_requests(
        self, images: List[bytes], num_concurrent: int, num_rounds: int = 3
    ) -> Dict[str, Any]:
        """Benchmark concurrent request processing."""
        all_latencies = []
        all_throughputs = []

        async with RayServeReplicaClient() as client:
            for round_num in range(num_rounds):
                print(f"    Round {round_num + 1}/{num_rounds}...")

                start_time = time.perf_counter()

                # Create concurrent tasks
                tasks = [
                    client.extract_features_single(images[i % len(images)])
                    for i in range(num_concurrent)
                ]

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = time.perf_counter()
                total_time = (end_time - start_time) * 1000

                # Count successful requests
                successful_requests = sum(
                    1 for result in results if not isinstance(result, Exception)
                )

                if successful_requests > 0:
                    avg_latency = total_time / successful_requests
                    throughput = successful_requests / (total_time / 1000)

                    all_latencies.append(avg_latency)
                    all_throughputs.append(throughput)

        return {
            "avg_latency_ms": statistics.mean(all_latencies)
            if all_latencies
            else float("inf"),
            "throughput_rps": statistics.mean(all_throughputs)
            if all_throughputs
            else 0,
            "std_latency_ms": statistics.stdev(all_latencies)
            if len(all_latencies) > 1
            else 0,
            "std_throughput_rps": statistics.stdev(all_throughputs)
            if len(all_throughputs) > 1
            else 0,
            "num_concurrent": num_concurrent,
            "successful_rounds": len(all_latencies),
        }

    async def benchmark_burst_traffic(
        self, images: List[bytes], burst_size: int, num_bursts: int = 5
    ) -> Dict[str, Any]:
        """Benchmark burst traffic patterns."""
        burst_results = []

        async with RayServeReplicaClient() as client:
            for burst_num in range(num_bursts):
                print(
                    f"    Burst {burst_num + 1}/{num_bursts} ({burst_size} requests)..."
                )

                start_time = time.perf_counter()

                # Create burst of requests
                tasks = [
                    client.extract_features_single(images[i % len(images)])
                    for i in range(burst_size)
                ]

                results = await asyncio.gather(*tasks, return_exceptions=True)

                end_time = time.perf_counter()
                burst_time = (end_time - start_time) * 1000

                successful_requests = sum(
                    1 for result in results if not isinstance(result, Exception)
                )

                burst_results.append(
                    {
                        "time_ms": burst_time,
                        "successful_requests": successful_requests,
                        "throughput_rps": successful_requests / (burst_time / 1000),
                    }
                )

                # Small delay between bursts
                await asyncio.sleep(0.5)

        return {
            "avg_burst_time_ms": statistics.mean([b["time_ms"] for b in burst_results]),
            "avg_throughput_rps": statistics.mean(
                [b["throughput_rps"] for b in burst_results]
            ),
            "total_successful_requests": sum(
                [b["successful_requests"] for b in burst_results]
            ),
            "burst_size": burst_size,
            "num_bursts": num_bursts,
            "burst_results": burst_results,
        }

    async def benchmark_sustained_load(
        self, images: List[bytes], rps_target: int, duration_seconds: int = 30
    ) -> Dict[str, Any]:
        """Benchmark sustained load over time."""
        print(f"    Sustaining {rps_target} RPS for {duration_seconds} seconds...")

        request_interval = 1.0 / rps_target
        end_time = time.time() + duration_seconds
        request_times = []
        successful_requests = 0
        failed_requests = 0

        async with RayServeReplicaClient() as client:
            start_time = time.time()

            while time.time() < end_time:
                request_start = time.perf_counter()

                try:
                    _ = await client.extract_features_single(images[0])
                    successful_requests += 1
                except Exception:
                    failed_requests += 1

                request_end = time.perf_counter()
                request_times.append((request_end - request_start) * 1000)

                # Wait for next request based on target RPS
                elapsed = time.time() - start_time
                expected_requests = elapsed * rps_target
                actual_requests = successful_requests + failed_requests

                if actual_requests < expected_requests:
                    # We're behind, don't wait
                    continue
                else:
                    # Wait for next interval
                    await asyncio.sleep(request_interval)

        actual_duration = time.time() - start_time
        actual_rps = successful_requests / actual_duration

        return {
            "target_rps": rps_target,
            "actual_rps": actual_rps,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "avg_latency_ms": statistics.mean(request_times) if request_times else 0,
            "p95_latency_ms": np.percentile(request_times, 95) if request_times else 0,
            "p99_latency_ms": np.percentile(request_times, 99) if request_times else 0,
            "duration_seconds": actual_duration,
        }

    async def run_replica_comparison(self):
        """Run comprehensive replica comparison benchmark."""
        print("🔄 RAY SERVE REPLICA COMPARISON BENCHMARK")
        print("=" * 70)
        print("📝 Instructions:")
        print("   1. First, run with num_replicas=1 in main.py")
        print("   2. Then change to num_replicas=4 (or your desired number)")
        print("   3. Compare the results below")
        print("=" * 70)

        # Check Ray Serve availability
        print("Checking Ray Serve availability...")
        async with RayServeReplicaClient() as client:
            ray_available = await client.health_check()

        if not ray_available:
            print("❌ Ray Serve is not available. Please start the service first.")
            print("Run: serve run src.feature_extraction.main:feature_extraction_app")
            return

        print("✅ Ray Serve is available")

        # Generate test data
        test_images = self.generate_test_images(20)

        print("\n📊 BENCHMARK RESULTS")
        print("=" * 70)

        # 1. Concurrent Request Scaling Test
        print("\n1. 🚀 CONCURRENT REQUEST SCALING")
        print("-" * 50)

        concurrent_loads = [1, 5, 10, 20, 40, 80, 100, 120, 140, 160, 180, 200]
        concurrent_results = {}

        for load in concurrent_loads:
            print(f"\nTesting {load} concurrent requests:")
            result = await self.benchmark_concurrent_requests(test_images, load)
            concurrent_results[load] = result

            print(
                f"  Avg Latency: {result['avg_latency_ms']:.2f}ms (±{result['std_latency_ms']:.2f})"
            )
            print(
                f"  Throughput:  {result['throughput_rps']:.1f} RPS (±{result['std_throughput_rps']:.1f})"
            )

        # 2. Burst Traffic Test
        print("\n2. 💥 BURST TRAFFIC HANDLING")
        print("-" * 50)

        burst_sizes = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        burst_results = {}

        for burst_size in burst_sizes:
            print(f"\nTesting bursts of {burst_size} requests:")
            result = await self.benchmark_burst_traffic(test_images, burst_size)
            burst_results[burst_size] = result

            print(f"  Avg Burst Time: {result['avg_burst_time_ms']:.0f}ms")
            print(f"  Avg Throughput: {result['avg_throughput_rps']:.1f} RPS")

        # 3. Sustained Load Test
        print("\n3. ⏱️  SUSTAINED LOAD TEST")
        print("-" * 50)

        rps_targets = [5, 10, 20, 30]
        sustained_results = {}

        for target_rps in rps_targets:
            print(f"\nTesting sustained {target_rps} RPS:")
            result = await self.benchmark_sustained_load(test_images, target_rps, 20)
            sustained_results[target_rps] = result

            print(
                f"  Target: {result['target_rps']} RPS, Actual: {result['actual_rps']:.1f} RPS"
            )
            print(
                f"  Success Rate: {result['successful_requests']}/{result['successful_requests'] + result['failed_requests']}"
            )
            print(
                f"  Avg Latency: {result['avg_latency_ms']:.2f}ms, P95: {result['p95_latency_ms']:.2f}ms"
            )

        # 4. Summary and Analysis
        print("\n📈 ANALYSIS & INSIGHTS")
        print("=" * 70)

        # Find optimal concurrent load
        best_concurrent = max(
            concurrent_results.keys(),
            key=lambda x: concurrent_results[x]["throughput_rps"],
        )

        print("🎯 Peak Performance:")
        print(f"   Best Concurrent Load: {best_concurrent} requests")
        print(
            f"   Peak Throughput: {concurrent_results[best_concurrent]['throughput_rps']:.1f} RPS"
        )
        print(
            f"   Latency at Peak: {concurrent_results[best_concurrent]['avg_latency_ms']:.2f}ms"
        )

        # Analyze scaling efficiency
        single_rps = concurrent_results[1]["throughput_rps"]
        peak_rps = concurrent_results[best_concurrent]["throughput_rps"]
        scaling_factor = peak_rps / single_rps

        print("\n📊 Scaling Analysis:")
        print(f"   Single Request RPS: {single_rps:.1f}")
        print(f"   Peak Concurrent RPS: {peak_rps:.1f}")
        print(f"   Scaling Factor: {scaling_factor:.1f}x")

        # Latency analysis
        latencies = [result["avg_latency_ms"] for result in concurrent_results.values()]
        min_latency = min(latencies)
        max_latency = max(latencies)

        print("\n⏱️  Latency Analysis:")
        print(f"   Best Latency: {min_latency:.2f}ms")
        print(f"   Worst Latency: {max_latency:.2f}ms")
        print(f"   Latency Increase: {(max_latency / min_latency - 1) * 100:.1f}%")

        print("\n💡 REPLICA CONFIGURATION GUIDELINES")
        print("=" * 70)
        print("🔍 What to look for when comparing replica configurations:")
        print("   • Higher throughput with more replicas")
        print("   • Better handling of concurrent requests")
        print("   • More stable latencies under load")
        print("   • Better burst traffic handling")
        print("   • Higher sustained RPS capability")
        print()
        print("⚖️  Trade-offs:")
        print("   • More replicas = Higher memory usage")
        print("   • More replicas = Better fault tolerance")
        print("   • More replicas = Better load distribution")
        print("   • Optimal replica count depends on your hardware")
        print()
        print("🎯 Recommended replica configuration:")
        print("   • Start with num_replicas = number of CPU cores")
        print("   • Adjust based on your specific workload")
        print("   • Monitor GPU utilization if using GPUs")
        print("   • Consider ray_actor_options for resource allocation")


async def main():
    """Main benchmark function."""
    benchmark = ReplicaComparisonBenchmark()
    await benchmark.run_replica_comparison()


if __name__ == "__main__":
    print("Ray Serve Replica Configuration Comparison")
    print(
        "Make sure Ray Serve is running: serve run src.feature_extraction.main:feature_extraction_app"
    )
    print("Change num_replicas in main.py between tests")
    print("Press Ctrl+C to stop the benchmark at any time\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark stopped by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
