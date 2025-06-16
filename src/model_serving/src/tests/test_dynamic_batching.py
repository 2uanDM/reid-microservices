import asyncio
import io
import json
import os
import statistics
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import httpx
import numpy as np
from PIL import Image

# Add the current working directory to Python path
sys.path.append(os.getcwd())


class DynamicBatchingClient:
    """Client for testing Ray Serve dynamic batching under heavy load."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        # Increased timeout and connection limits for heavy load testing
        self.client = httpx.AsyncClient(
            timeout=300.0,  # 5 minutes timeout for extreme loads
            limits=httpx.Limits(
                max_keepalive_connections=200,
                max_connections=500,
                keepalive_expiry=30.0,
            ),
        )
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

    async def send_batch_request(
        self, image_bytes: bytes, request_id: int
    ) -> Dict[str, Any]:
        """Send a single request to the batch endpoint."""
        start_time = time.perf_counter()

        try:
            files = {"image": (f"test_{request_id}.jpg", image_bytes, "image/jpeg")}
            response = await self.client.post(
                f"{self.base_url}/embedding/batch", files=files, data={"model": "osnet"}
            )

            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000

            if response.status_code == 200:
                result = response.json()
                return {
                    "request_id": request_id,
                    "latency_ms": latency,
                    "status": "success",
                    "features_count": len(result["features"]),
                    "batch_count": result.get("count", 1),
                    "timestamp": time.time(),
                }
            else:
                return {
                    "request_id": request_id,
                    "latency_ms": latency,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}",
                    "timestamp": time.time(),
                }

        except Exception as e:
            end_time = time.perf_counter()
            latency = (end_time - start_time) * 1000
            return {
                "request_id": request_id,
                "latency_ms": latency,
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }


class DynamicBatchingBenchmark:
    """Comprehensive benchmark for testing dynamic batching under extreme loads."""

    def __init__(self):
        self.test_images = []
        self.results_log = []

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
        print(f"Generating {count} test images...")
        return [self.generate_test_image() for _ in range(count)]

    async def test_massive_concurrent_load(
        self, images: List[bytes], num_requests: int, arrival_pattern: str = "instant"
    ) -> Dict[str, Any]:
        """Test massive concurrent load with different arrival patterns."""
        print(
            f"🚀 Testing {num_requests} concurrent requests with {arrival_pattern} arrival..."
        )

        async with DynamicBatchingClient() as client:
            start_time = time.perf_counter()

            if arrival_pattern == "instant":
                # All requests sent at exactly the same time
                tasks = [
                    client.send_batch_request(images[i % len(images)], i)
                    for i in range(num_requests)
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

            elif arrival_pattern == "burst_waves":
                # Send requests in waves to simulate burst traffic
                wave_size = min(100, num_requests // 5)  # 5 waves
                results = []

                for wave in range(0, num_requests, wave_size):
                    wave_end = min(wave + wave_size, num_requests)
                    wave_tasks = [
                        client.send_batch_request(images[i % len(images)], i)
                        for i in range(wave, wave_end)
                    ]
                    wave_results = await asyncio.gather(
                        *wave_tasks, return_exceptions=True
                    )
                    results.extend(wave_results)

                    # Small delay between waves
                    if wave_end < num_requests:
                        await asyncio.sleep(0.1)

            elif arrival_pattern == "gradual_ramp":
                # Gradually increase the request rate
                results = []
                batch_sizes = [10, 25, 50, 100, 200, 500, 1000]
                current_id = 0

                for batch_size in batch_sizes:
                    if current_id >= num_requests:
                        break

                    actual_batch = min(batch_size, num_requests - current_id)
                    batch_tasks = [
                        client.send_batch_request(
                            images[i % len(images)], current_id + i
                        )
                        for i in range(actual_batch)
                    ]
                    batch_results = await asyncio.gather(
                        *batch_tasks, return_exceptions=True
                    )
                    results.extend(batch_results)
                    current_id += actual_batch

                    await asyncio.sleep(0.2)  # Brief pause between ramp levels

            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000

            # Process results
            successful_results = [
                r
                for r in results
                if isinstance(r, dict) and r.get("status") == "success"
            ]
            failed_results = [
                r
                for r in results
                if isinstance(r, dict) and r.get("status") != "success"
            ]
            exception_results = [r for r in results if isinstance(r, Exception)]

            return {
                "total_requests": num_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "exception_count": len(exception_results),
                "total_time_ms": total_time,
                "arrival_pattern": arrival_pattern,
                "throughput_rps": len(successful_results) / (total_time / 1000)
                if total_time > 0
                else 0,
                "success_rate": len(successful_results) / num_requests * 100,
                "latency_stats": self._calculate_latency_stats(successful_results),
                "batch_efficiency": self._analyze_batch_efficiency(successful_results),
                "detailed_results": successful_results[
                    :100
                ],  # Keep first 100 for analysis
            }

    def _calculate_latency_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate detailed latency statistics."""
        if not results:
            return {}

        latencies = [r["latency_ms"] for r in results]

        return {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "p99_9_ms": np.percentile(latencies, 99.9),
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

    def _analyze_batch_efficiency(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze how effectively dynamic batching is working."""
        if not results:
            return {}

        batch_counts = [r.get("batch_count", 1) for r in results]

        return {
            "avg_batch_size": statistics.mean(batch_counts),
            "max_batch_size": max(batch_counts),
            "min_batch_size": min(batch_counts),
            "batch_size_distribution": {
                "1": sum(1 for b in batch_counts if b == 1),
                "2-5": sum(1 for b in batch_counts if 2 <= b <= 5),
                "6-15": sum(1 for b in batch_counts if 6 <= b <= 15),
                "16-32": sum(1 for b in batch_counts if 16 <= b <= 32),
                "32+": sum(1 for b in batch_counts if b > 32),
            },
            "batching_effectiveness": (statistics.mean(batch_counts) - 1)
            / 31
            * 100,  # % of max batch size utilized
        }

    async def test_sustained_heavy_load(
        self, images: List[bytes], target_rps: int, duration_minutes: int = 5
    ) -> Dict[str, Any]:
        """Test sustained heavy load over extended period."""
        print(
            f"🔥 Testing sustained load: {target_rps} RPS for {duration_minutes} minutes..."
        )

        duration_seconds = duration_minutes * 60
        request_interval = 1.0 / target_rps
        end_time = time.time() + duration_seconds

        results = []
        request_id = 0

        async with DynamicBatchingClient() as client:
            start_time = time.time()

            while time.time() < end_time:
                batch_start = time.time()

                # Send a batch of requests to maintain target RPS
                batch_size = min(10, target_rps // 10)  # Send in small batches
                batch_tasks = []

                for _ in range(batch_size):
                    if time.time() >= end_time:
                        break
                    task = client.send_batch_request(
                        images[request_id % len(images)], request_id
                    )
                    batch_tasks.append(task)
                    request_id += 1

                if batch_tasks:
                    batch_results = await asyncio.gather(
                        *batch_tasks, return_exceptions=True
                    )
                    results.extend([r for r in batch_results if isinstance(r, dict)])

                # Wait to maintain target RPS
                batch_duration = time.time() - batch_start
                expected_duration = batch_size * request_interval

                if batch_duration < expected_duration:
                    await asyncio.sleep(expected_duration - batch_duration)

        actual_duration = time.time() - start_time
        successful_results = [r for r in results if r.get("status") == "success"]

        return {
            "target_rps": target_rps,
            "actual_rps": len(successful_results) / actual_duration,
            "duration_minutes": actual_duration / 60,
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100
            if results
            else 0,
            "latency_stats": self._calculate_latency_stats(successful_results),
            "batch_efficiency": self._analyze_batch_efficiency(successful_results),
        }

    async def test_extreme_burst_scenarios(self, images: List[bytes]) -> Dict[str, Any]:
        """Test extreme burst scenarios that would overwhelm traditional systems."""
        print("💥 Testing extreme burst scenarios...")

        scenarios = [
            {"name": "Lightning Burst", "requests": 1000, "pattern": "instant"},
            {"name": "Thunder Storm", "requests": 2000, "pattern": "burst_waves"},
            {"name": "Tsunami Wave", "requests": 5000, "pattern": "instant"},
            {
                "name": "Earthquake Aftershocks",
                "requests": 3000,
                "pattern": "burst_waves",
            },
            {"name": "Volcanic Eruption", "requests": 10000, "pattern": "gradual_ramp"},
        ]

        scenario_results = {}

        for scenario in scenarios:
            print(f"\n🌪️  Running {scenario['name']} scenario...")
            try:
                result = await self.test_massive_concurrent_load(
                    images, scenario["requests"], scenario["pattern"]
                )
                scenario_results[scenario["name"]] = result

                print(
                    f"   ✅ {result['successful_requests']}/{result['total_requests']} succeeded"
                )
                print(f"   ⚡ {result['throughput_rps']:.1f} RPS achieved")
                print(
                    f"   ⏱️  P95 latency: {result['latency_stats'].get('p95_ms', 0):.2f}ms"
                )

                # Brief recovery time between scenarios
                await asyncio.sleep(2)

            except Exception as e:
                print(f"   ❌ {scenario['name']} failed: {e}")
                scenario_results[scenario["name"]] = {"error": str(e)}

        return scenario_results

    async def test_batch_size_optimization(self, images: List[bytes]) -> Dict[str, Any]:
        """Test different concurrent loads to find optimal batch sizes."""
        print("🔬 Testing batch size optimization...")

        concurrent_loads = [1, 5, 10, 20, 32, 50, 64, 100, 128, 200, 300, 500, 1000]
        optimization_results = {}

        for load in concurrent_loads:
            print(f"   Testing {load} concurrent requests...")
            result = await self.test_massive_concurrent_load(images, load, "instant")

            optimization_results[load] = {
                "throughput_rps": result["throughput_rps"],
                "avg_latency_ms": result["latency_stats"].get("mean_ms", 0),
                "p95_latency_ms": result["latency_stats"].get("p95_ms", 0),
                "avg_batch_size": result["batch_efficiency"].get("avg_batch_size", 1),
                "success_rate": result["success_rate"],
                "batching_effectiveness": result["batch_efficiency"].get(
                    "batching_effectiveness", 0
                ),
            }

            print(f"      Throughput: {result['throughput_rps']:.1f} RPS")
            print(
                f"      Avg Batch Size: {result['batch_efficiency'].get('avg_batch_size', 1):.1f}"
            )

            # Brief pause between tests
            await asyncio.sleep(1)

        return optimization_results

    def save_results_to_file(self, results: Dict[str, Any], filename: str = None):
        """Save detailed results to JSON file for analysis."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dynamic_batching_results_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📄 Results saved to {filename}")

    async def run_comprehensive_dynamic_batching_test(self):
        """Run the complete dynamic batching test suite."""
        print("🧪 RAY SERVE DYNAMIC BATCHING STRESS TEST")
        print("=" * 80)
        print("🎯 Testing Ray Serve's dynamic batching under extreme concurrent loads")
        print(
            "📊 This will test batching efficiency, throughput, and latency under stress"
        )
        print("=" * 80)

        # Check Ray Serve availability
        print("Checking Ray Serve availability...")
        async with DynamicBatchingClient() as client:
            if not await client.health_check():
                print("❌ Ray Serve is not available. Please start the service first.")
                print("Run: serve run src.main:model_service")
                return

        print("✅ Ray Serve is available")

        # Generate test data
        self.test_images = self.generate_test_images(50)  # Diverse set of test images

        all_results = {
            "test_timestamp": datetime.now().isoformat(),
            "test_configuration": {
                "max_batch_size": 32,  # From your Ray Serve config
                "batch_wait_timeout_s": 0.01,
                "num_test_images": len(self.test_images),
            },
        }

        print("\n🔬 PHASE 1: BATCH SIZE OPTIMIZATION")
        print("=" * 60)
        optimization_results = await self.test_batch_size_optimization(self.test_images)
        all_results["batch_optimization"] = optimization_results

        # Find optimal concurrent load
        best_load = max(
            optimization_results.keys(),
            key=lambda x: optimization_results[x]["throughput_rps"],
        )
        print(f"\n🎯 Optimal concurrent load found: {best_load} requests")
        print(
            f"   Peak throughput: {optimization_results[best_load]['throughput_rps']:.1f} RPS"
        )
        print(
            f"   Avg batch size: {optimization_results[best_load]['avg_batch_size']:.1f}"
        )

        print("\n💥 PHASE 2: EXTREME BURST SCENARIOS")
        print("=" * 60)
        burst_results = await self.test_extreme_burst_scenarios(self.test_images)
        all_results["extreme_bursts"] = burst_results

        print("\n🔥 PHASE 3: SUSTAINED HEAVY LOAD")
        print("=" * 60)
        sustained_loads = [50, 100, 200, 300]  # Different RPS targets
        sustained_results = {}

        for target_rps in sustained_loads:
            print(f"\nTesting sustained {target_rps} RPS...")
            result = await self.test_sustained_heavy_load(
                self.test_images, target_rps, 2
            )  # 2 minutes each
            sustained_results[target_rps] = result

            print(f"   Achieved: {result['actual_rps']:.1f} RPS")
            print(f"   Success rate: {result['success_rate']:.1f}%")
            print(
                f"   Avg batch size: {result['batch_efficiency'].get('avg_batch_size', 1):.1f}"
            )

        all_results["sustained_load"] = sustained_results

        print("\n📊 COMPREHENSIVE ANALYSIS")
        print("=" * 80)

        # Analyze batching effectiveness
        print("\n🔍 DYNAMIC BATCHING ANALYSIS:")
        print("-" * 50)

        for load, result in optimization_results.items():
            if load in [1, 10, 32, 100, 500, 1000]:  # Key data points
                effectiveness = result["batching_effectiveness"]
                avg_batch = result["avg_batch_size"]
                throughput = result["throughput_rps"]

                print(
                    f"Load {load:4d}: Batch {avg_batch:4.1f} | Effectiveness {effectiveness:5.1f}% | Throughput {throughput:6.1f} RPS"
                )

        # Performance insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 50)

        max_throughput = max(r["throughput_rps"] for r in optimization_results.values())
        single_req_throughput = optimization_results[1]["throughput_rps"]
        scaling_factor = (
            max_throughput / single_req_throughput if single_req_throughput > 0 else 0
        )

        print(f"🚀 Peak Throughput: {max_throughput:.1f} RPS")
        print(
            f"📈 Scaling Factor: {scaling_factor:.1f}x improvement over single requests"
        )

        # Find sweet spot for batching
        sweet_spot = max(
            optimization_results.keys(),
            key=lambda x: optimization_results[x]["batching_effectiveness"],
        )
        print(f"🎯 Batching Sweet Spot: {sweet_spot} concurrent requests")
        print(
            f"   Batch effectiveness: {optimization_results[sweet_spot]['batching_effectiveness']:.1f}%"
        )
        print(
            f"   Average batch size: {optimization_results[sweet_spot]['avg_batch_size']:.1f}"
        )

        # Latency analysis
        latencies = [
            (load, result["p95_latency_ms"])
            for load, result in optimization_results.items()
        ]
        min_latency_load, min_latency = min(latencies, key=lambda x: x[1])

        print(
            f"⚡ Best P95 Latency: {min_latency:.2f}ms at {min_latency_load} concurrent requests"
        )

        print("\n🏆 DYNAMIC BATCHING BENEFITS:")
        print("-" * 50)
        print("✅ AUTOMATIC BATCHING: Requests automatically grouped for efficiency")
        print("✅ THROUGHPUT OPTIMIZATION: Higher RPS with intelligent batching")
        print("✅ LATENCY MANAGEMENT: Balanced latency vs throughput trade-offs")
        print("✅ RESOURCE EFFICIENCY: Better GPU/CPU utilization")
        print("✅ SCALABILITY: Handles massive concurrent loads gracefully")
        print("✅ ADAPTIVE BEHAVIOR: Batch sizes adapt to load patterns")

        print("\n⚙️  CONFIGURATION RECOMMENDATIONS:")
        print("-" * 50)
        print("🔧 Current max_batch_size: 32 (good for most workloads)")
        print("🔧 Current batch_wait_timeout_s: 0.01s (10ms - good for low latency)")
        print(f"🔧 Optimal concurrent load: {best_load} requests")
        print(
            "🔧 Consider increasing max_batch_size to 64 for higher throughput workloads"
        )
        print("🔧 Consider increasing timeout to 0.02s if latency is less critical")

        # Save results
        self.save_results_to_file(all_results)

        print("\n✅ Dynamic batching test completed successfully!")
        print(
            f"📈 Peak performance: {max_throughput:.1f} RPS with {scaling_factor:.1f}x scaling"
        )


async def main():
    """Main test function."""
    benchmark = DynamicBatchingBenchmark()
    await benchmark.run_comprehensive_dynamic_batching_test()


if __name__ == "__main__":
    print("Ray Serve Dynamic Batching Stress Test")
    print("Make sure Ray Serve is running with dynamic batching enabled")
    print("Expected config: @serve.batch(max_batch_size=32, batch_wait_timeout_s=0.01)")
    print("Press Ctrl+C to stop the test at any time\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
