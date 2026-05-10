from __future__ import annotations

import argparse
from pathlib import Path


def parse_shape(text: str) -> tuple[int, int, int, int]:
    values = tuple(int(part) for part in text.lower().replace("x", ",").split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("shape must be N,18,8,8 or Nx18x8x8")
    if values[1:] != (18, 8, 8):
        raise argparse.ArgumentTypeError("tiny baseline input shape must be N,18,8,8")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TensorRT engine from an exported chessmoe ONNX model."
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--min-shape", type=parse_shape, default=(1, 18, 8, 8))
    parser.add_argument("--opt-shape", type=parse_shape, default=(8, 18, 8, 8))
    parser.add_argument("--max-shape", type=parse_shape, default=(64, 18, 8, 8))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import tensorrt as trt
    except Exception as exc:
        print(f"TensorRT Python package is not available: {exc}")
        return 2

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = args.onnx.read_bytes()
    if not parser.parse(onnx_bytes):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        return 1

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024
    )
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("board", args.min_shape, args.opt_shape, args.max_shape)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("TensorRT failed to build a serialized engine")
        return 1

    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f"Built TensorRT engine: {args.engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
