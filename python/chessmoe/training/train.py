from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
from typing import Iterable

import torch
from torch.utils.data import DataLoader

from chessmoe.models.tiny_model import TinyChessNet
from chessmoe.training.checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
)
from chessmoe.training.config import (
    TrainingConfig,
    load_training_config,
    training_config_to_dict,
)
from chessmoe.training.data import (
    ReplayDataset,
    TrainingBatch,
    collate_replay_samples,
    split_dataset,
)
from chessmoe.training.losses import TinyLossTargets, compute_tiny_loss


@dataclass(frozen=True)
class TrainingResult:
    start_epoch: int
    epochs_completed: int
    train_losses: list[float]
    validation_losses: list[float]


def run_training(config: TrainingConfig) -> TrainingResult:
    _set_seed(config.seed)
    device = _resolve_device(config.device)

    dataset = ReplayDataset.from_index(config.replay_index)
    train_data, validation_data = split_dataset(
        dataset,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_replay_samples,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_replay_samples,
    )

    model = TinyChessNet(channels=config.model_channels, hidden=config.model_hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer)
    scaler = _build_scaler(device, config.amp)
    start_epoch = 0

    if config.resume_checkpoint is not None:
        restored = load_training_checkpoint(
            config.resume_checkpoint,
            model_channels=config.model_channels,
            model_hidden=config.model_hidden,
            map_location=device,
        )
        model.load_state_dict(restored.model.state_dict())
        optimizer.load_state_dict(restored.optimizer_state)
        if restored.scheduler_state is not None and scheduler is not None:
            scheduler.load_state_dict(restored.scheduler_state)
        if restored.scaler_state is not None:
            scaler.load_state_dict(restored.scaler_state)
        start_epoch = restored.epoch

    train_model = model
    if config.compile_model:
        train_model = torch.compile(model)

    metrics_path = Path(config.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(start_epoch, config.epochs):
        train_metrics = _train_one_epoch(
            train_model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
        )
        if scheduler is not None:
            scheduler.step()
        validation_metrics = _evaluate(model, validation_loader, device, config)
        train_losses.append(train_metrics["loss"])
        validation_losses.append(validation_metrics["loss"])

        _append_metrics(
            metrics_path,
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": validation_metrics,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )
        save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            path=config.checkpoint_path,
            epoch=epoch + 1,
            config=training_config_to_dict(config),
        )

    return TrainingResult(
        start_epoch=start_epoch,
        epochs_completed=config.epochs,
        train_losses=train_losses,
        validation_losses=validation_losses,
    )


def _train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[TrainingBatch],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    totals = _metric_totals()
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, config.amp):
            output = model(batch.features)
            losses = compute_tiny_loss(
                output,
                TinyLossTargets(
                    policy=batch.policy,
                    wdl=batch.wdl,
                    moves_left=batch.moves_left,
                ),
                moves_left_weight=config.moves_left_weight,
            )

        scaler.scale(losses.total).backward()
        if config.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        _accumulate(totals, losses, batch.features.shape[0])
    return _averages(totals)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: Iterable[TrainingBatch],
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, float]:
    if len(loader) == 0:
        return {"loss": 0.0, "policy": 0.0, "wdl": 0.0, "moves_left": 0.0}
    model.eval()
    totals = _metric_totals()
    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast_context(device, config.amp):
            output = model(batch.features)
            losses = compute_tiny_loss(
                output,
                TinyLossTargets(
                    policy=batch.policy,
                    wdl=batch.wdl,
                    moves_left=batch.moves_left,
                ),
                moves_left_weight=config.moves_left_weight,
            )
        _accumulate(totals, losses, batch.features.shape[0])
    return _averages(totals)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_scheduler(
    config: TrainingConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler == "none":
        return None
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, config.epochs),
        )
    raise ValueError(f"unsupported scheduler: {config.scheduler}")


def _build_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    enabled = amp and device.type == "cuda"
    return torch.amp.GradScaler(device.type, enabled=enabled)


def _autocast_context(device: torch.device, amp: bool):
    enabled = amp and torch.amp.autocast_mode.is_autocast_available(device.type)
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _to_device(batch: TrainingBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        features=batch.features.to(device),
        policy=batch.policy.to(device),
        wdl=batch.wdl.to(device),
        moves_left=batch.moves_left.to(device),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_totals() -> dict[str, float]:
    return {"loss": 0.0, "policy": 0.0, "wdl": 0.0, "moves_left": 0.0, "count": 0.0}


def _accumulate(totals: dict[str, float], losses, batch_size: int) -> None:
    totals["loss"] += float(losses.total.detach().cpu()) * batch_size
    totals["policy"] += float(losses.policy.detach().cpu()) * batch_size
    totals["wdl"] += float(losses.wdl.detach().cpu()) * batch_size
    totals["moves_left"] += float(losses.moves_left.detach().cpu()) * batch_size
    totals["count"] += batch_size


def _averages(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "policy": totals["policy"] / count,
        "wdl": totals["wdl"] / count,
        "moves_left": totals["moves_left"] / count,
    }


def _append_metrics(path: Path, metrics: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metrics, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TinyChessNet from replay chunks")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_training(load_training_config(args.config))
    print(
        f"training complete: epochs={result.epochs_completed}, "
        f"last_train_loss={result.train_losses[-1]:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
