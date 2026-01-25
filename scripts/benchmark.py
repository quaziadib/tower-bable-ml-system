#!/usr/bin/env python3
"""
Performance Benchmarking Script

Compares PyTorch vs TensorRT-LLM inference speed to validate the 5-10x improvement.

Usage:
    python scripts/benchmark.py [--num-runs 10] [--warmup 3]

Output:
    - Latency statistics (avg, p50, p95, min, max)
    - Speedup calculation
    - JSON report saved to benchmark_results.json
"""

import os
import sys
import time
import statistics
import json
import argparse
from typing import List, Dict, Tuple

# Add parent directory to path to import main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_backend(
    backend_name: str,
    use_tensorrt: bool,
    test_inputs: List[Tuple[str, str, str]],
    num_runs: int = 10,
    warmup: int = 3
) -> Dict:
    """
    Benchmark a specific backend (PyTorch or TensorRT).

    Args:
        backend_name: Human-readable name ("PyTorch" or "TensorRT")
        use_tensorrt: Whether to use TensorRT (True) or PyTorch (False)
        test_inputs: List of (text, src_lang, tgt_lang) tuples
        num_runs: Number of runs per test case
        warmup: Number of warmup runs before measurement

    Returns:
        Dictionary with latency statistics
    """
    # Set environment variable
    os.environ["USE_TENSORRT"] = "1" if use_tensorrt else "0"

    # Import translate function (this will load the model with correct backend)
    print(f"\n{'='*70}")
    print(f"Benchmarking {backend_name} Backend")
    print(f"{'='*70}\n")

    # Force reload of main module to pick up new USE_TENSORRT value
    if 'main' in sys.modules:
        del sys.modules['main']

    from main import translate

    results = []

    for idx, (text, src, tgt) in enumerate(test_inputs, 1):
        print(f"\nTest case {idx}/{len(test_inputs)}: {len(text)} characters")
        print(f"Direction: {src} → {tgt}")

        # Warmup runs
        print(f"Warmup ({warmup} runs)...")
        for _ in range(warmup):
            try:
                translate(text, src, tgt)
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                return None

        # Measurement runs
        print(f"Measuring ({num_runs} runs)...")
        latencies = []

        for run in range(num_runs):
            start_time = time.time()

            try:
                result = translate(text, src, tgt)
                elapsed_ms = (time.time() - start_time) * 1000
                latencies.append(elapsed_ms)

                if (run + 1) % 3 == 0 or run == num_runs - 1:
                    print(f"  Run {run+1}/{num_runs}: {elapsed_ms:.1f}ms")

            except Exception as e:
                print(f"❌ Run {run+1} failed: {e}")
                continue

        if not latencies:
            print(f"❌ All runs failed for test case {idx}")
            continue

        # Calculate statistics
        stats = {
            "text_length": len(text),
            "src_lang": src,
            "tgt_lang": tgt,
            "num_runs": len(latencies),
            "avg_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0],
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "all_latencies": latencies
        }

        results.append(stats)

        print(f"\nResults:")
        print(f"  Average: {stats['avg_ms']:.1f}ms")
        print(f"  Median:  {stats['median_ms']:.1f}ms")
        print(f"  P95:     {stats['p95_ms']:.1f}ms")
        print(f"  Range:   {stats['min_ms']:.1f}ms - {stats['max_ms']:.1f}ms")

    return {
        "backend": backend_name,
        "use_tensorrt": use_tensorrt,
        "test_cases": results,
        "overall_avg_ms": statistics.mean([r["avg_ms"] for r in results]) if results else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT-LLM")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs per test case")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch benchmark")
    parser.add_argument("--skip-tensorrt", action="store_true", help="Skip TensorRT benchmark")

    args = parser.parse_args()

    print("="*70)
    print("Translation API Performance Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Runs per test: {args.num_runs}")
    print(f"  Warmup runs: {args.warmup}")

    # Test inputs (short, medium, long)
    test_inputs = [
        ("আমি ভাত খাই", "Bangla", "English"),  # Short (~15 chars)

        ("The Bishop of Ramsbury was an episcopal title used by medieval English-Catholic diocesan bishops in the Anglo-Saxon English church.", "English", "Bangla"),  # Medium (~140 chars)

        ("রামসবারির বিশপ মধ্যযুগীয় ইংলিশ-ক্যাথলিক ডায়োসেসান বিশপদের দ্বারা ব্যবহৃত একটি এপিস্কোপাল শিরোনাম ছিল অ্যাংলো-স্যাক্সন ইংলিশ চার্চে। শিরোনামটি উইল্টশায়ারের রামসবারি গ্রাম থেকে এর নাম নিয়েছে এবং প্রথম ব্যবহার করা হয়েছিল ১০ম এবং ১১তম শতাব্দীতে অ্যাংলো-স্যাক্সন বিশপ অফ রামসবারি দ্বারা। স্যাক্সন সময়ে, রামসবারি চার্চের জন্য একটি গুরুত্বপূর্ণ অবস্থান ছিল এবং বেশ কিছু প্রাথমিক বিশপ ক্যান্টারবারির আর্চবিশপ হয়েছিলেন।", "Bangla", "English"),  # Long (~400 chars)
    ]

    print(f"\nTest cases: {len(test_inputs)}")
    for i, (text, src, tgt) in enumerate(test_inputs, 1):
        print(f"  {i}. {len(text)} chars ({src} → {tgt})")

    # Benchmark PyTorch
    pytorch_results = None
    if not args.skip_pytorch:
        pytorch_results = benchmark_backend(
            backend_name="PyTorch FP16",
            use_tensorrt=False,
            test_inputs=test_inputs,
            num_runs=args.num_runs,
            warmup=args.warmup
        )

    # Benchmark TensorRT
    tensorrt_results = None
    if not args.skip_tensorrt:
        # Check if TensorRT engine exists
        engine_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "trt_engines",
            "babel-9b-int8"
        )

        if not os.path.exists(engine_path):
            print(f"\n❌ TensorRT engine not found at: {engine_path}")
            print("   Please run the TensorRT setup first:")
            print("   1. python scripts/merge_lora.py")
            print("   2. Convert to TensorRT checkpoint")
            print("   3. Build TensorRT engine")
            print("\nSkipping TensorRT benchmark.")
        else:
            tensorrt_results = benchmark_backend(
                backend_name="TensorRT INT8",
                use_tensorrt=True,
                test_inputs=test_inputs,
                num_runs=args.num_runs,
                warmup=args.warmup
            )

    # Compare results
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)

    if pytorch_results and tensorrt_results:
        print("\n📊 Latency Comparison:")
        print(f"\n{'Test Case':<20} {'PyTorch':<15} {'TensorRT':<15} {'Speedup':<10}")
        print("-" * 70)

        for i, (pt, trt) in enumerate(zip(pytorch_results["test_cases"], tensorrt_results["test_cases"])):
            speedup = pt["avg_ms"] / trt["avg_ms"] if trt["avg_ms"] > 0 else 0
            print(f"{i+1}. {pt['text_length']} chars{'':<8} "
                  f"{pt['avg_ms']:.1f}ms{'':<8} "
                  f"{trt['avg_ms']:.1f}ms{'':<8} "
                  f"{speedup:.2f}x")

        # Overall speedup
        overall_speedup = pytorch_results["overall_avg_ms"] / tensorrt_results["overall_avg_ms"]
        print("-" * 70)
        print(f"{'OVERALL':<20} "
              f"{pytorch_results['overall_avg_ms']:.1f}ms{'':<8} "
              f"{tensorrt_results['overall_avg_ms']:.1f}ms{'':<8} "
              f"{overall_speedup:.2f}x")

        print(f"\n🎯 Target achieved: {'✓ YES' if overall_speedup >= 5.0 else '✗ NO'} "
              f"(target: 5-10x, achieved: {overall_speedup:.1f}x)")

    elif pytorch_results:
        print(f"\nPyTorch average latency: {pytorch_results['overall_avg_ms']:.1f}ms")

    elif tensorrt_results:
        print(f"\nTensorRT average latency: {tensorrt_results['overall_avg_ms']:.1f}ms")

    # Save results
    output_file = "benchmark_results.json"
    results_data = {
        "pytorch": pytorch_results,
        "tensorrt": tensorrt_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "num_runs": args.num_runs,
            "warmup": args.warmup
        }
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
