from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report student latency/strength tradeoffs")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda", "tensorrt"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--arena-student-vs-teacher", type=Path)
    parser.add_argument("--arena-student-vs-best", type=Path)
    parser.add_argument("--report", type=Path, default=Path("data/benchmark/student_tradeoff.json"))
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


def load_arena_summary(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary": payload.get("summary"),
        "decision": payload.get("promotion", {}).get("decision"),
        "config": payload.get("config"),
    }


def run_latency_benchmark(args: argparse.Namespace) -> dict:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit(f"onnxruntime is not available: {exc}")

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
    return {
        "provider": args.provider,
        "batch_size": args.batch_size,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": percentile(latencies_ms, 50),
        "p95_latency_ms": percentile(latencies_ms, 95),
        "throughput_positions_per_second": args.batch_size / (mean_ms / 1000.0),
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("batch-size and iterations must be positive")

    latency = run_latency_benchmark(args)
    arena_teacher = load_arena_summary(args.arena_student_vs_teacher)
    arena_best = load_arena_summary(args.arena_student_vs_best)

    report = {
        "created_at_ms": int(time.time() * 1000),
        "student": {
            "onnx": str(args.onnx),
            "latency": latency,
        },
        "arena": {
            "vs_teacher": arena_teacher,
            "vs_best": arena_best,
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"tradeoff report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
