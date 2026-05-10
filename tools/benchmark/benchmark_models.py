from __future__ import annotations

import argparse
import statistics
import time

import torch

from chessmoe.models.dense_transformer import DenseTransformerConfig
from chessmoe.models.dense_transformer import parameter_count
from chessmoe.models.factory import build_model
from chessmoe.models.encoding import BOARD_SHAPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tiny CNN and dense transformer forward-pass speed."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def benchmark(model: torch.nn.Module, batch: torch.Tensor, warmup: int, iterations: int) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        model(batch)
    if batch.device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(batch)
        if batch.device.type == "cuda":
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.fmean(latencies_ms)
    return {
        "mean_latency_ms": mean_ms,
        "positions_per_second": batch.shape[0] / (mean_ms / 1000.0),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    batch = torch.zeros((args.batch_size, *BOARD_SHAPE), dtype=torch.float32, device=device)
    tiny = build_model("tiny_cnn", tiny_channels=32, tiny_hidden=128).to(device)
    transformer = build_model(
        "dense_transformer",
        transformer_config=DenseTransformerConfig(
            d_model=args.d_model,
            num_layers=args.layers,
            num_heads=args.heads,
            ffn_dim=args.ffn_dim,
            dropout=0.0,
        ),
    ).to(device)

    tiny_result = benchmark(tiny, batch, args.warmup, args.iterations)
    transformer_result = benchmark(transformer, batch, args.warmup, args.iterations)
    print(f"device={device}")
    print(f"batch_size={args.batch_size}")
    print(f"tiny_cnn_parameters={parameter_count(tiny)}")
    print(f"tiny_cnn_mean_latency_ms={tiny_result['mean_latency_ms']:.3f}")
    print(f"tiny_cnn_positions_per_second={tiny_result['positions_per_second']:.2f}")
    print(f"dense_transformer_parameters={parameter_count(transformer)}")
    print(f"dense_transformer_mean_latency_ms={transformer_result['mean_latency_ms']:.3f}")
    print(
        "dense_transformer_positions_per_second="
        f"{transformer_result['positions_per_second']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
