from __future__ import annotations

from torch import nn

from chessmoe.models.dense_transformer import DenseTransformerConfig, DenseTransformerEvaluator
from chessmoe.models.tiny_model import TinyChessNet


def build_model(
    kind: str,
    *,
    tiny_channels: int = 32,
    tiny_hidden: int = 128,
    transformer_config: DenseTransformerConfig | None = None,
) -> nn.Module:
    if kind == "tiny_cnn":
        return TinyChessNet(channels=tiny_channels, hidden=tiny_hidden)
    if kind == "dense_transformer":
        return DenseTransformerEvaluator(transformer_config)
    raise ValueError(f"unsupported model kind: {kind}")


def model_kind(model: nn.Module) -> str:
    source = model._orig_mod if hasattr(model, "_orig_mod") else model
    if isinstance(source, TinyChessNet):
        return "tiny_cnn"
    if isinstance(source, DenseTransformerEvaluator):
        return "dense_transformer"
    raise ValueError(f"unsupported model type: {type(source).__name__}")
