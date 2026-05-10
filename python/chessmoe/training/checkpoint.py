from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from chessmoe.models.dense_transformer import DenseTransformerConfig, DenseTransformerEvaluator
from chessmoe.models.factory import build_model, model_kind as infer_model_kind
from chessmoe.models.moe_module import MoEConfig
from chessmoe.models.moe_transformer import MoETransformerConfig, MoETransformerEvaluator
from chessmoe.models.student_hybrid import StudentHybridConfig, StudentHybridEvaluator
from chessmoe.models.tiny_model import TinyChessNet


class TrainingCheckpoint:
    def __init__(
        self,
        model: nn.Module,
        optimizer_state: dict[str, Any],
        scheduler_state: dict[str, Any] | None,
        scaler_state: dict[str, Any] | None,
        epoch: int,
        metadata: dict[str, Any],
    ) -> None:
        self.model = model
        self.optimizer_state = optimizer_state
        self.scheduler_state = scheduler_state
        self.scaler_state = scaler_state
        self.epoch = epoch
        self.metadata = metadata


def save_checkpoint(model: nn.Module, path: str | Path, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = infer_model_kind(model)
    torch.save(
        {
            "model": _checkpoint_model_name(kind),
            "model_kind": kind,
            "model_kwargs": _model_kwargs(model),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> nn.Module:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    kind = _checkpoint_kind(checkpoint)
    model = _build_from_checkpoint(kind, checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    path: str | Path,
    epoch: int,
    **metadata: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = infer_model_kind(model)
    torch.save(
        {
            "model": _checkpoint_model_name(kind),
            "model_kind": kind,
            "model_kwargs": _model_kwargs(model),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "scaler_state_dict": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "metadata": metadata,
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model_kind: str = "tiny_cnn",
    model_channels: int = 32,
    model_hidden: int = 128,
    transformer_config: DenseTransformerConfig | None = None,
    moe_transformer_config: MoETransformerConfig | None = None,
    student_hybrid_config: StudentHybridConfig | None = None,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    kind = checkpoint.get("model_kind", model_kind)
    model_kwargs = checkpoint.get("model_kwargs", {})
    if not model_kwargs and kind == "tiny_cnn":
        model_kwargs = {"channels": model_channels, "hidden": model_hidden}
    if not model_kwargs and kind == "dense_transformer" and transformer_config is not None:
        model_kwargs = transformer_config.to_dict()
    if not model_kwargs and kind == "moe_transformer" and moe_transformer_config is not None:
        model_kwargs = moe_transformer_config.to_dict()
    if not model_kwargs and kind == "student_hybrid" and student_hybrid_config is not None:
        model_kwargs = student_hybrid_config.to_dict()
    model = _build_from_checkpoint(kind, model_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(map_location)
    return TrainingCheckpoint(
        model=model,
        optimizer_state=checkpoint["optimizer_state_dict"],
        scheduler_state=checkpoint.get("scheduler_state_dict"),
        scaler_state=checkpoint.get("scaler_state_dict"),
        epoch=int(checkpoint["epoch"]),
        metadata=checkpoint.get("metadata", {}),
    )


def _tiny_model_channels(model: nn.Module) -> int:
    source = model._orig_mod if hasattr(model, "_orig_mod") else model
    return int(source.trunk[0].out_channels)


def _tiny_model_hidden(model: nn.Module) -> int:
    source = model._orig_mod if hasattr(model, "_orig_mod") else model
    return int(source.policy_head.in_features)


def _checkpoint_model_name(kind: str) -> str:
    if kind == "tiny_cnn":
        return "TinyChessNet"
    if kind == "dense_transformer":
        return "DenseTransformerEvaluator"
    if kind == "moe_transformer":
        return "MoETransformerEvaluator"
    if kind == "student_hybrid":
        return "StudentHybridEvaluator"
    raise ValueError(f"unsupported model kind: {kind}")


def _checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    if "model_kind" in checkpoint:
        return str(checkpoint["model_kind"])
    if checkpoint.get("model") == "TinyChessNet":
        return "tiny_cnn"
    if checkpoint.get("model") == "DenseTransformerEvaluator":
        return "dense_transformer"
    if checkpoint.get("model") == "MoETransformerEvaluator":
        return "moe_transformer"
    if checkpoint.get("model") == "StudentHybridEvaluator":
        return "student_hybrid"
    raise ValueError("checkpoint does not contain a supported model")


def _model_kwargs(model: nn.Module) -> dict[str, Any]:
    source = model._orig_mod if hasattr(model, "_orig_mod") else model
    if isinstance(source, TinyChessNet):
        return {
            "channels": _tiny_model_channels(source),
            "hidden": _tiny_model_hidden(source),
        }
    if isinstance(source, DenseTransformerEvaluator):
        return source.config.to_dict()
    if isinstance(source, MoETransformerEvaluator):
        return source.config.to_dict()
    if isinstance(source, StudentHybridEvaluator):
        return source.config.to_dict()
    raise ValueError(f"unsupported model type: {type(source).__name__}")


def _build_from_checkpoint(kind: str, model_kwargs: dict[str, Any]) -> nn.Module:
    if kind == "tiny_cnn":
        return build_model(
            "tiny_cnn",
            tiny_channels=int(model_kwargs.get("channels", 32)),
            tiny_hidden=int(model_kwargs.get("hidden", 128)),
        )
    if kind == "dense_transformer":
        return build_model(
            "dense_transformer",
            transformer_config=DenseTransformerConfig(
                d_model=int(model_kwargs.get("d_model", 128)),
                num_layers=int(model_kwargs.get("num_layers", 4)),
                num_heads=int(model_kwargs.get("num_heads", 8)),
                ffn_dim=int(model_kwargs.get("ffn_dim", 512)),
                dropout=float(model_kwargs.get("dropout", 0.1)),
                layer_norm_eps=float(model_kwargs.get("layer_norm_eps", 1.0e-5)),
                uncertainty_head=bool(model_kwargs.get("uncertainty_head", False)),
            ),
        )
    if kind == "moe_transformer":
        moe_kwargs = model_kwargs.get("moe", {})
        moe_config = MoEConfig(
            num_experts=int(moe_kwargs.get("num_experts", 8)),
            top_k_training=int(moe_kwargs.get("top_k_training", 2)),
            top_k_inference=int(moe_kwargs.get("top_k_inference", 1)),
            capacity_factor=float(moe_kwargs.get("capacity_factor", 1.25)),
            load_balance_coeff=float(moe_kwargs.get("load_balance_coeff", 0.01)),
            router_entropy_coeff=float(moe_kwargs.get("router_entropy_coeff", 0.001)),
            router_noise_std=float(moe_kwargs.get("router_noise_std", 0.1)),
            dense_fallback=bool(moe_kwargs.get("dense_fallback", False)),
            expert_dropout=float(moe_kwargs.get("expert_dropout", 0.0)),
        )
        moe_layers_raw = model_kwargs.get("moe_layers", [1, 3])
        moe_layers = tuple(int(x) for x in moe_layers_raw)
        return build_model(
            "moe_transformer",
            moe_transformer_config=MoETransformerConfig(
                d_model=int(model_kwargs.get("d_model", 128)),
                num_layers=int(model_kwargs.get("num_layers", 4)),
                num_heads=int(model_kwargs.get("num_heads", 8)),
                ffn_dim=int(model_kwargs.get("ffn_dim", 512)),
                dropout=float(model_kwargs.get("dropout", 0.1)),
                layer_norm_eps=float(model_kwargs.get("layer_norm_eps", 1.0e-5)),
                uncertainty_head=bool(model_kwargs.get("uncertainty_head", False)),
                moe_layers=moe_layers,
                moe=moe_config,
                dense_fallback_config=bool(model_kwargs.get("dense_fallback_config", False)),
            ),
        )
    if kind == "student_hybrid":
        return build_model(
            "student_hybrid",
            student_hybrid_config=StudentHybridConfig(
                conv_channels=int(model_kwargs.get("conv_channels", 32)),
                d_model=int(model_kwargs.get("d_model", 96)),
                num_layers=int(model_kwargs.get("num_layers", 2)),
                num_heads=int(model_kwargs.get("num_heads", 4)),
                ffn_dim=int(model_kwargs.get("ffn_dim", 192)),
                dropout=float(model_kwargs.get("dropout", 0.1)),
                layer_norm_eps=float(model_kwargs.get("layer_norm_eps", 1.0e-5)),
            ),
        )
    raise ValueError(f"unsupported model kind: {kind}")
