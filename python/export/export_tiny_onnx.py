from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chessmoe.export.onnx_export import export_tiny_onnx
from chessmoe.models.dense_transformer import DenseTransformerConfig
from chessmoe.models.factory import build_model
from chessmoe.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export tiny chessmoe model to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-kind", choices=["tiny_cnn", "dense_transformer"], default="tiny_cnn")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--static-batch", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--opset", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = (
        load_checkpoint(args.checkpoint)
        if args.checkpoint
        else build_model(
            args.model_kind,
            transformer_config=DenseTransformerConfig(
                d_model=args.d_model,
                num_layers=args.layers,
                num_heads=args.heads,
                ffn_dim=args.ffn_dim,
                dropout=0.0,
            ),
        )
    )
    with torch.no_grad():
        result = export_tiny_onnx(
            model,
            args.output,
            verify=args.verify,
            dynamic_batch=not args.static_batch,
            opset_version=args.opset,
        )
    if result.status != "exported":
        print(f"ONNX export skipped: {result.reason}")
        return 2
    print(f"Exported ONNX model: {result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
