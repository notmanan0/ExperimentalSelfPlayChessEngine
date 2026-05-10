"""MoE vs Dense inference benchmark.

Compares inference latency and throughput between dense transformer and
MoE transformer evaluators at matched parameter counts.

Usage:
    python tools/benchmark/moe_inference_benchmark.py [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import time

import torch

from chessmoe.models.dense_transformer import DenseTransformerConfig, DenseTransformerEvaluator
from chessmoe.models.moe_module import MoEConfig
from chessmoe.models.moe_transformer import MoETransformerConfig, MoETransformerEvaluator
from chessmoe.models.encoding import BOARD_SHAPE


def make_dense_model(d_model: int = 64, num_layers: int = 4, ffn_dim: int = 256) -> DenseTransformerEvaluator:
    config = DenseTransformerConfig(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8,
        ffn_dim=ffn_dim,
        dropout=0.0,
    )
    return DenseTransformerEvaluator(config)


def make_moe_model(
    d_model: int = 64,
    num_layers: int = 4,
    ffn_dim: int = 256,
    num_experts: int = 8,
    moe_layers: tuple[int, ...] = (1, 3),
) -> MoETransformerEvaluator:
    config = MoETransformerConfig(
        d_model=d_model,
        num_layers=num_layers,
        num_heads=8,
        ffn_dim=ffn_dim,
        dropout=0.0,
        moe_layers=moe_layers,
        moe=MoEConfig(
            num_experts=num_experts,
            top_k_training=2,
            top_k_inference=1,
            capacity_factor=1.0,
        ),
    )
    return MoETransformerEvaluator(config)


def benchmark_model(
    model: torch.nn.Module,
    batch_size: int,
    num_warmup: int,
    num_iterations: int,
    device: torch.device,
) -> dict[str, float]:
    model = model.to(device)
    model.eval()
    dummy = torch.zeros((batch_size, *BOARD_SHAPE), dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(num_warmup):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(num_iterations):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_iterations) * 1000
    throughput = (batch_size * num_iterations) / elapsed

    return {
        "avg_ms": avg_ms,
        "throughput_positions_per_sec": throughput,
        "total_time_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE vs Dense inference benchmark")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
    print()

    dense = make_dense_model(args.d_model, args.num_layers, args.ffn_dim)
    moe = make_moe_model(args.d_model, args.num_layers, args.ffn_dim, args.num_experts)

    dense_params = sum(p.numel() for p in dense.parameters())
    moe_params = sum(p.numel() for p in moe.parameters())
    print(f"Dense parameters:  {dense_params:,}")
    print(f"MoE parameters:    {moe_params:,}")
    print(f"MoE/Dense ratio:   {moe_params / dense_params:.2f}x")
    print()

    print("Benchmarking dense model...")
    dense_results = benchmark_model(dense, args.batch_size, args.warmup, args.iterations, device)
    print(f"  Avg latency: {dense_results['avg_ms']:.2f} ms")
    print(f"  Throughput:  {dense_results['throughput_positions_per_sec']:.0f} positions/sec")
    print()

    print("Benchmarking MoE model (inference, top-1)...")
    moe_results = benchmark_model(moe, args.batch_size, args.warmup, args.iterations, device)
    print(f"  Avg latency: {moe_results['avg_ms']:.2f} ms")
    print(f"  Throughput:  {moe_results['throughput_positions_per_sec']:.0f} positions/sec")
    print()

    overhead_pct = ((moe_results["avg_ms"] / dense_results["avg_ms"]) - 1) * 100
    print(f"MoE overhead: {overhead_pct:+.1f}%")

    print("\nBenchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
