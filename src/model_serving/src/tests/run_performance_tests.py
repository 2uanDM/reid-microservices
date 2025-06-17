#!/usr/bin/env python3
"""
Performance Test Runner Script

Usage:
    python run_performance_tests.py --help
    python run_performance_tests.py --all
    python run_performance_tests.py --test single_requests
    python run_performance_tests.py --test batch_requests --num-requests 50
    python run_performance_tests.py --test concurrent_requests --concurrency 20
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import performance_test
sys.path.append(str(Path(__file__).parent.parent))

from tests.performance_test import RayServePerformanceTester


async def run_single_test(test_name: str, **kwargs):
    """Run a single performance test"""
    tester = RayServePerformanceTester(
        base_url=kwargs.get("base_url", "http://localhost:8000")
    )

    # Setup test images
    tester.setup_test_images(num_images=kwargs.get("num_images", 100))

    print(f"Running {test_name} test...")

    if test_name == "single_requests":
        metrics = await tester.test_single_requests(
            num_requests=kwargs.get("num_requests", 100),
            endpoint=kwargs.get("endpoint", "/embedding"),
        )
    elif test_name == "batch_requests":
        metrics = await tester.test_batch_requests(
            num_requests=kwargs.get("num_requests", 100),
            endpoint=kwargs.get("endpoint", "/embedding/batch"),
        )
    elif test_name == "true_batch":
        metrics = await tester.test_true_batch_requests(
            num_batches=kwargs.get("num_batches", 10),
            batch_size=kwargs.get("batch_size", 10),
        )
    elif test_name == "concurrent_requests":
        metrics = await tester.test_concurrent_requests(
            num_requests=kwargs.get("num_requests", 100),
            concurrency=kwargs.get("concurrency", 10),
            endpoint=kwargs.get("endpoint", "/embedding"),
        )
    elif test_name == "burst_requests":
        metrics = await tester.test_burst_requests(
            burst_size=kwargs.get("burst_size", 20),
            num_bursts=kwargs.get("num_bursts", 5),
            burst_interval=kwargs.get("burst_interval", 2),
            endpoint=kwargs.get("endpoint", "/embedding"),
        )
    elif test_name == "normal_inference_sequential":
        metrics = tester.test_normal_inference_sequential(
            num_requests=kwargs.get("num_requests", 100)
        )
    elif test_name == "normal_inference_batch":
        metrics = tester.test_normal_inference_batch(
            num_batches=kwargs.get("num_batches", 10),
            batch_size=kwargs.get("batch_size", 10),
        )
    else:
        print(f"Unknown test: {test_name}")
        return

    tester.print_metrics(test_name.replace("_", " ").title(), metrics)

    # Save individual test result
    filename = f"{test_name}_results.json"
    tester.save_results_to_json({test_name: metrics}, filename)


async def main():
    parser = argparse.ArgumentParser(description="Ray Serve Performance Test Runner")

    # Main test options
    parser.add_argument("--all", action="store_true", help="Run all performance tests")
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "single_requests",
            "batch_requests",
            "true_batch",
            "concurrent_requests",
            "burst_requests",
            "normal_inference_sequential",
            "normal_inference_batch",
        ],
        help="Run a specific test",
    )

    # Configuration options
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for Ray Serve (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests for single/batch/concurrent tests (default: 100)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="Number of batches for batch tests (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for batch tests (default: 10)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrency level for concurrent tests (default: 10)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=20,
        help="Burst size for burst tests (default: 20)",
    )
    parser.add_argument(
        "--num-bursts",
        type=int,
        default=5,
        help="Number of bursts for burst tests (default: 5)",
    )
    parser.add_argument(
        "--burst-interval",
        type=int,
        default=2,
        help="Interval between bursts in seconds (default: 2)",
    )
    parser.add_argument(
        "--endpoint",
        default="/embedding",
        help="Endpoint to test (default: /embedding)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of test images to generate (default: 100)",
    )

    # Quick test presets
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests with reduced parameters"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress tests with increased parameters",
    )

    args = parser.parse_args()

    # Adjust parameters for quick/stress tests
    if args.quick:
        args.num_requests = 20
        args.num_batches = 3
        args.batch_size = 5
        args.concurrency = 5
        args.burst_size = 10
        args.num_bursts = 2
        args.num_images = 50
        print("🚀 Running in QUICK mode with reduced parameters")

    elif args.stress:
        args.num_requests = 500
        args.num_batches = 50
        args.batch_size = 20
        args.concurrency = 50
        args.burst_size = 100
        args.num_bursts = 10
        args.num_images = 500
        print("💪 Running in STRESS mode with increased parameters")

    # Convert args to kwargs
    kwargs = {
        "base_url": args.base_url,
        "num_requests": args.num_requests,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "burst_size": args.burst_size,
        "num_bursts": args.num_bursts,
        "burst_interval": args.burst_interval,
        "endpoint": args.endpoint,
        "num_images": args.num_images,
    }

    if args.all:
        print("Running comprehensive performance test suite...")
        tester = RayServePerformanceTester(base_url=args.base_url)
        await tester.run_comprehensive_test_suite()

    elif args.test:
        await run_single_test(args.test, **kwargs)

    else:
        print("Please specify --all or --test <test_name>")
        print("Use --help for more information")


if __name__ == "__main__":
    asyncio.run(main())
