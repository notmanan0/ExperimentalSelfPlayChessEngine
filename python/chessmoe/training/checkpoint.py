from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from chessmoe.models.tiny_model import TinyChessNet


class TrainingCheckpoint:
    def __init__(
        self,
        model: TinyChessNet,
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


def save_checkpoint(model: TinyChessNet, path: str | Path, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "TinyChessNet",
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> TinyChessNet:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("model") != "TinyChessNet":
        raise ValueError("checkpoint does not contain a TinyChessNet model")
    model = TinyChessNet()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def save_training_checkpoint(
    model: TinyChessNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    path: str | Path,
    epoch: int,
    **metadata: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "TinyChessNet",
            "model_kwargs": {
                "channels": _tiny_model_channels(model),
                "hidden": _tiny_model_hidden(model),
            },
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
    model_channels: int = 32,
    model_hidden: int = 128,
    map_location: str | torch.device = "cpu",
) -> TrainingCheckpoint:
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("model") != "TinyChessNet":
        raise ValueError("checkpoint does not contain a TinyChessNet model")
    model_kwargs = checkpoint.get("model_kwargs", {})
    model = TinyChessNet(
        channels=int(model_kwargs.get("channels", model_channels)),
        hidden=int(model_kwargs.get("hidden", model_hidden)),
    )
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
