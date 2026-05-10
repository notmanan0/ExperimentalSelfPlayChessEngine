from __future__ import annotations

import argparse
from pathlib import Path

from chessmoe.export.onnx_export import export_policy_value_onnx
from chessmoe.training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export distilled student to ONNX")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dynamic-batch", dest="dynamic_batch", action="store_true")
    parser.add_argument("--static-batch", dest="dynamic_batch", action="store_false")
    parser.set_defaults(dynamic_batch=True)
    parser.add_argument("--opset-version", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = load_checkpoint(args.checkpoint, map_location="cpu")
    result = export_policy_value_onnx(
        model,
        args.output,
        verify=args.verify,
        dynamic_batch=args.dynamic_batch,
        opset_version=args.opset_version,
    )
    print(f"student export status={result.status} path={result.path}")
    if result.reason:
        print(f"reason={result.reason}")
    return 0 if result.status == "exported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
