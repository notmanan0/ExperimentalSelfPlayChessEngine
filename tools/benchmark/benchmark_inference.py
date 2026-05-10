from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chessmoe batched inference.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def providers(provider: str, fp16: bool) -> list:
    if provider == "tensorrt":
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": fp16,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "python-test-output/trt-cache",
                    "trt_profile_min_shapes": "board:1x18x8x8",
                    "trt_profile_opt_shapes": "board:8x18x8x8",
                    "trt_profile_max_shapes": "board:64x18x8x8",
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return sorted(values)[index]


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("batch-size and iterations must be positive")

    try:
        import onnxruntime as ort
    except Exception as exc:
        print(f"onnxruntime is not available: {exc}")
        return 2

    session = ort.InferenceSession(
        str(args.onnx), providers=providers(args.provider, args.fp16)
    )
    board = np.zeros((args.batch_size, 18, 8, 8), dtype=np.float32)

    for _ in range(args.warmup):
        session.run(None, {"board": board})

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        session.run(None, {"board": board})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    positions_per_second = args.batch_size / (mean_ms / 1000.0)
    print(f"provider={args.provider}")
    print(f"batch_size={args.batch_size}")
    print(f"mean_latency_ms={mean_ms:.3f}")
    print(f"p50_latency_ms={percentile(latencies_ms, 50):.3f}")
    print(f"p95_latency_ms={percentile(latencies_ms, 95):.3f}")
    print(f"throughput_positions_per_second={positions_per_second:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
