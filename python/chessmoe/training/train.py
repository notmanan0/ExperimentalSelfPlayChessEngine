from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import random
import time
from typing import Iterable

import torch
from torch.utils.data import DataLoader, DistributedSampler

from chessmoe.models.dense_transformer import DenseTransformerConfig
from chessmoe.models.factory import build_model
from chessmoe.models.moe_module import MoEConfig
from chessmoe.models.moe_transformer import MoETransformerConfig
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
from chessmoe.training.distributed import (
    DistributedContext,
    barrier,
    ddp_wrap,
    destroy_distributed,
    init_distributed,
    reduce_metric_totals,
    should_log,
    unwrap_model,
)
from chessmoe.training.losses import TinyLossTargets, compute_tiny_loss, compute_moe_aware_loss


@dataclass(frozen=True)
class TrainingResult:
    start_epoch: int
    epochs_completed: int
    train_losses: list[float]
    validation_losses: list[float]


def run_training(config: TrainingConfig) -> TrainingResult:
    base_device = _resolve_device(config.device)
    dist_context = init_distributed(config, base_device)
    try:
        _configure_reproducibility(config)
        _set_seed(config.seed)

        dataset = ReplayDataset.from_index(
            config.replay_index,
            target_policy=config.target_policy,
            reanalysis_fraction=config.reanalysis_fraction,
            reanalysis_seed=config.reanalysis_seed,
        )
        train_data, validation_data = split_dataset(
            dataset,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )

        train_loader, train_sampler = _build_loader(
            train_data,
            config,
            dist_context,
            shuffle=True,
        )
        validation_loader, _ = _build_loader(
            validation_data,
            config,
            dist_context,
            shuffle=False,
        )

        model = _build_configured_model(config).to(dist_context.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_scheduler(config, optimizer)
        scaler = _build_scaler(dist_context.device, config.amp)
        start_epoch = 0

        if config.resume_checkpoint is not None:
            restored = load_training_checkpoint(
                config.resume_checkpoint,
                model_kind=config.model_kind,
                model_channels=config.model_channels,
                model_hidden=config.model_hidden,
                transformer_config=_transformer_config(config),
                moe_transformer_config=_moe_transformer_config(config)
                if config.model_kind == "moe_transformer"
                else None,
                map_location=dist_context.device,
            )
            model.load_state_dict(restored.model.state_dict())
            optimizer.load_state_dict(restored.optimizer_state)
            if restored.scheduler_state is not None and scheduler is not None:
                scheduler.load_state_dict(restored.scheduler_state)
            if restored.scaler_state is not None:
                scaler.load_state_dict(restored.scaler_state)
            start_epoch = restored.epoch

        if config.fsdp_enabled and should_log(dist_context, config.log_all_ranks):
            print("fsdp_enabled requested; using DDP baseline until FSDP is wired")
        ddp_model = ddp_wrap(model, dist_context, config)
        train_model = ddp_model
        if config.compile_model and dist_context.enabled:
            if should_log(dist_context, config.log_all_ranks):
                print("compile_model is disabled for distributed training")
        elif config.compile_model:
            train_model = torch.compile(ddp_model)

        metrics_path = Path(config.metrics_path)
        if should_log(dist_context, config.log_all_ranks):
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
        train_losses: list[float] = []
        validation_losses: list[float] = []

        if should_log(dist_context, config.log_all_ranks):
            print(
                "training start: "
                f"epochs={config.epochs} batch_size={config.batch_size} "
                f"checkpoint={config.checkpoint_path} metrics={metrics_path} "
                f"device={dist_context.device} amp={config.amp and dist_context.device.type == 'cuda'} "
                f"model_kind={config.model_kind}"
            )

        for epoch in range(start_epoch, config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = _train_one_epoch(
                train_model,
                train_loader,
                optimizer,
                scaler,
                dist_context,
                config,
            )
            if scheduler is not None:
                scheduler.step()
            validation_metrics = _evaluate(
                train_model,
                validation_loader,
                dist_context,
                config,
            )
            train_losses.append(train_metrics["loss"])
            validation_losses.append(validation_metrics["loss"])

            if should_log(dist_context, config.log_all_ranks):
                print(
                    "training epoch complete: "
                    f"epoch={epoch + 1}/{config.epochs} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"validation_loss={validation_metrics['loss']:.6f} "
                    f"checkpoint={config.checkpoint_path} metrics={metrics_path}"
                )
                _append_metrics(
                    metrics_path,
                    {
                        "epoch": epoch + 1,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                        "rank": dist_context.rank,
                        "world_size": dist_context.world_size,
                        "grad_accum_steps": config.grad_accum_steps,
                    },
                )
            if dist_context.is_rank0:
                save_training_checkpoint(
                    model=unwrap_model(ddp_model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    path=config.checkpoint_path,
                    epoch=epoch + 1,
                    config=training_config_to_dict(config),
                )
            barrier(dist_context)

        return TrainingResult(
            start_epoch=start_epoch,
            epochs_completed=config.epochs,
            train_losses=train_losses,
            validation_losses=validation_losses,
        )
    finally:
        destroy_distributed(dist_context)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[TrainingBatch],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context: DistributedContext,
    config: TrainingConfig,
) -> dict[str, float]:
    model.train()
    totals = _metric_totals()
    is_moe = config.model_kind == "moe_transformer"
    accum_steps = _resolve_grad_accum_steps(config.grad_accum_steps)
    num_batches = len(loader)
    started = time.monotonic()
    processed_samples = 0
    optimizer.zero_grad(set_to_none=True)
    for step_index, batch in enumerate(loader):
        batch = _to_device(batch, context.device)
        batch_size = batch.features.shape[0]
        is_last_micro = (step_index + 1) % accum_steps == 0 or (step_index + 1) == num_batches
        with _no_sync_context(model, context.enabled and not is_last_micro):
            with _autocast_context(context.device, config.amp):
                output = model(batch.features)
                if is_moe:
                    losses = compute_moe_aware_loss(
                        output,
                        TinyLossTargets(
                            policy=batch.policy,
                            wdl=batch.wdl,
                            moves_left=batch.moves_left,
                        ),
                        moves_left_weight=config.moves_left_weight,
                        moe_load_balance_coeff=config.moe_load_balance_coeff,
                        moe_router_entropy_coeff=config.moe_router_entropy_coeff,
                    )
                else:
                    losses = compute_tiny_loss(
                        output,
                        TinyLossTargets(
                            policy=batch.policy,
                            wdl=batch.wdl,
                            moves_left=batch.moves_left,
                        ),
                        moves_left_weight=config.moves_left_weight,
                    )
                loss_value = losses.total / accum_steps

            scaler.scale(loss_value).backward()
        if is_last_micro:
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        _accumulate(totals, losses, batch_size)
        processed_samples += int(batch_size)
        if should_log(context, config.log_all_ranks):
            elapsed = time.monotonic() - started
            samples_per_second = processed_samples / elapsed if elapsed > 0 else 0.0
            remaining_batches = max(0, num_batches - (step_index + 1))
            eta = (
                remaining_batches * elapsed / (step_index + 1)
                if step_index + 1 > 0
                else 0.0
            )
            print(
                "training progress: "
                f"batch={step_index + 1}/{num_batches} "
                f"train_loss={float(losses.total.detach().cpu()):.6f} "
                f"samples/sec={samples_per_second:.2f} "
                f"elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                f"checkpoint={config.checkpoint_path} metrics={config.metrics_path} "
                f"device={context.device} amp={config.amp and context.device.type == 'cuda'} "
                f"model_kind={config.model_kind}"
            )
    totals = reduce_metric_totals(totals, context)
    return _averages(totals)


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: Iterable[TrainingBatch],
    context: DistributedContext,
    config: TrainingConfig,
) -> dict[str, float]:
    if len(loader) == 0:
        return {"loss": 0.0, "policy": 0.0, "wdl": 0.0, "moves_left": 0.0}
    model.eval()
    totals = _metric_totals()
    is_moe = config.model_kind == "moe_transformer"
    for batch in loader:
        batch = _to_device(batch, context.device)
        with _autocast_context(context.device, config.amp):
            output = model(batch.features)
            if is_moe:
                losses = compute_moe_aware_loss(
                    output,
                    TinyLossTargets(
                        policy=batch.policy,
                        wdl=batch.wdl,
                        moves_left=batch.moves_left,
                    ),
                    moves_left_weight=config.moves_left_weight,
                    moe_load_balance_coeff=config.moe_load_balance_coeff,
                    moe_router_entropy_coeff=config.moe_router_entropy_coeff,
                )
            else:
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
    totals = reduce_metric_totals(totals, context)
    return _averages(totals)


def _build_loader(
    dataset,
    config: TrainingConfig,
    context: DistributedContext,
    *,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    if context.enabled and context.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            seed=config.seed,
            drop_last=False,
        )
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        collate_fn=collate_replay_samples,
        generator=generator if sampler is None else None,
        worker_init_fn=_seed_worker,
        pin_memory=context.device.type == "cuda",
    )
    return loader, sampler


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def _configure_reproducibility(config: TrainingConfig) -> None:
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)


def _resolve_grad_accum_steps(value: int) -> int:
    steps = int(value)
    if steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    return steps


def _no_sync_context(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_configured_model(config: TrainingConfig) -> torch.nn.Module:
    if config.model_kind == "moe_transformer":
        return build_model(
            config.model_kind,
            moe_transformer_config=_moe_transformer_config(config),
        )
    return build_model(
        config.model_kind,
        tiny_channels=config.model_channels,
        tiny_hidden=config.model_hidden,
        transformer_config=_transformer_config(config),
    )


def _transformer_config(config: TrainingConfig) -> DenseTransformerConfig:
    return DenseTransformerConfig(
        d_model=config.transformer_d_model,
        num_layers=config.transformer_layers,
        num_heads=config.transformer_heads,
        ffn_dim=config.transformer_ffn_dim,
        dropout=config.transformer_dropout,
        uncertainty_head=config.transformer_uncertainty_head,
    )


def _moe_transformer_config(config: TrainingConfig) -> MoETransformerConfig:
    moe_config = MoEConfig(
        num_experts=config.moe_num_experts,
        top_k_training=config.moe_top_k_training,
        top_k_inference=config.moe_top_k_inference,
        capacity_factor=config.moe_capacity_factor,
        load_balance_coeff=config.moe_load_balance_coeff,
        router_entropy_coeff=config.moe_router_entropy_coeff,
        router_noise_std=config.moe_router_noise_std,
        dense_fallback=config.moe_dense_fallback,
        expert_dropout=config.moe_expert_dropout,
    )
    return MoETransformerConfig(
        d_model=config.transformer_d_model,
        num_layers=config.transformer_layers,
        num_heads=config.transformer_heads,
        ffn_dim=config.transformer_ffn_dim,
        dropout=config.transformer_dropout,
        uncertainty_head=config.transformer_uncertainty_head,
        moe_layers=tuple(config.moe_layers),
        moe=moe_config,
        dense_fallback_config=config.moe_dense_fallback,
    )


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
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    config = load_training_config(args.config)
    if args.local_rank is not None:
        config = TrainingConfig(**{**config.__dict__, "local_rank": args.local_rank})
    result = run_training(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 or config.log_all_ranks:
        print(
            f"training complete: epochs={result.epochs_completed}, "
            f"last_train_loss={result.train_losses[-1]:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
