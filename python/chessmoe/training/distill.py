from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import random
from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from chessmoe.models.dense_transformer import DenseTransformerConfig
from chessmoe.models.factory import build_model
from chessmoe.models.student_hybrid import StudentHybridConfig
from chessmoe.training.checkpoint import (
    load_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from chessmoe.training.distill_config import (
    DistillationConfig,
    distillation_config_to_dict,
    load_distillation_config,
)
from chessmoe.training.distill_data import (
    DistillationBatch,
    DistillationDataset,
    collate_distillation_samples,
    split_distillation_dataset,
)
from chessmoe.training.distill_losses import (
    HardTargets,
    TeacherTargets,
    compute_distillation_loss,
    teacher_targets_from_output,
)
from chessmoe.training.distributed import (
    DistributedContext,
    barrier,
    ddp_wrap,
    destroy_distributed,
    init_distributed,
    should_log,
    unwrap_model,
)


@dataclass(frozen=True)
class DistillationResult:
    start_epoch: int
    epochs_completed: int
    train_losses: list[float]
    validation_losses: list[float]


def run_distillation(config: DistillationConfig) -> DistillationResult:
    base_device = _resolve_device(config.device)
    dist_context = init_distributed(config, base_device)
    try:
        _configure_reproducibility(config)
        _set_seed(config.seed)

        dataset = DistillationDataset.from_index(
            config.replay_index,
            target_policy=config.target_policy,
            reanalysis_fraction=config.reanalysis_fraction,
            reanalysis_seed=config.reanalysis_seed,
        )
        train_data, validation_data = split_distillation_dataset(
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

        student = _build_student_model(config).to(dist_context.device)
        teacher_device = _resolve_teacher_device(config, dist_context.device)
        teacher = load_checkpoint(config.teacher_checkpoint, map_location=teacher_device)
        teacher.to(teacher_device)
        teacher.eval()

        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = _build_scheduler(config, optimizer)
        scaler = _build_scaler(dist_context.device, config.amp)
        start_epoch = 0

        if config.resume_checkpoint is not None:
            restored = load_training_checkpoint(
                config.resume_checkpoint,
                map_location=dist_context.device,
            )
            student.load_state_dict(restored.model.state_dict())
            optimizer.load_state_dict(restored.optimizer_state)
            if restored.scheduler_state is not None and scheduler is not None:
                scheduler.load_state_dict(restored.scheduler_state)
            if restored.scaler_state is not None:
                scaler.load_state_dict(restored.scaler_state)
            start_epoch = restored.epoch

        if config.fsdp_enabled and should_log(dist_context, config.log_all_ranks):
            print("fsdp_enabled requested; using DDP baseline until FSDP is wired")
        ddp_model = ddp_wrap(student, dist_context, config)
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

        for epoch in range(start_epoch, config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = _train_one_epoch(
                train_model,
                teacher,
                train_loader,
                optimizer,
                scaler,
                dist_context,
                teacher_device,
                config,
            )
            if scheduler is not None:
                scheduler.step()
            validation_metrics = _evaluate(
                train_model,
                teacher,
                validation_loader,
                dist_context,
                teacher_device,
                config,
            )
            train_losses.append(train_metrics["loss"])
            validation_losses.append(validation_metrics["loss"])

            if should_log(dist_context, config.log_all_ranks):
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
                    config=distillation_config_to_dict(config),
                )
            barrier(dist_context)

        return DistillationResult(
            start_epoch=start_epoch,
            epochs_completed=config.epochs,
            train_losses=train_losses,
            validation_losses=validation_losses,
        )
    finally:
        destroy_distributed(dist_context)


def _train_one_epoch(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Iterable[DistillationBatch],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context: DistributedContext,
    teacher_device: torch.device,
    config: DistillationConfig,
) -> dict[str, float]:
    student.train()
    totals = _metric_totals()
    accum_steps = _resolve_grad_accum_steps(config.grad_accum_steps)
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for step_index, batch in enumerate(loader):
        batch = _to_device(batch, context.device)
        is_last_micro = (step_index + 1) % accum_steps == 0 or (step_index + 1) == num_batches
        with _no_sync_context(student, context.enabled and not is_last_micro):
            with _autocast_context(context.device, config.amp):
                teacher_targets = _teacher_targets(
                    teacher,
                    batch.features,
                    teacher_device,
                    config,
                )
                hard_targets = HardTargets(
                    policy=batch.hard_policy,
                    wdl=batch.hard_wdl,
                    value=batch.hard_value,
                    moves_left=batch.hard_moves_left,
                )
                losses = compute_distillation_loss(
                    student(batch.features),
                    teacher_targets,
                    hard_targets,
                    temperature=config.temperature,
                    policy_kl_weight=config.policy_kl_weight,
                    wdl_kl_weight=config.wdl_kl_weight,
                    value_weight=config.value_weight,
                    moves_left_weight=config.moves_left_weight,
                    hard_target_weight=config.hard_target_weight,
                    hard_value_weight=config.hard_value_weight,
                    hard_moves_left_weight=config.hard_moves_left_weight,
                )
                loss_value = losses.total / accum_steps

            scaler.scale(loss_value).backward()
        if is_last_micro:
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        _accumulate(totals, losses, batch.features.shape[0])
    totals = _reduce_metric_totals(totals, context)
    return _averages(totals)


@torch.no_grad()
def _evaluate(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: Iterable[DistillationBatch],
    context: DistributedContext,
    teacher_device: torch.device,
    config: DistillationConfig,
) -> dict[str, float]:
    if len(loader) == 0:
        return {
            "loss": 0.0,
            "policy_kl": 0.0,
            "wdl_kl": 0.0,
            "value": 0.0,
            "moves_left": 0.0,
        }
    student.eval()
    totals = _metric_totals()
    for batch in loader:
        batch = _to_device(batch, context.device)
        with _autocast_context(context.device, config.amp):
            teacher_targets = _teacher_targets(
                teacher,
                batch.features,
                teacher_device,
                config,
            )
            hard_targets = HardTargets(
                policy=batch.hard_policy,
                wdl=batch.hard_wdl,
                value=batch.hard_value,
                moves_left=batch.hard_moves_left,
            )
            losses = compute_distillation_loss(
                student(batch.features),
                teacher_targets,
                hard_targets,
                temperature=config.temperature,
                policy_kl_weight=config.policy_kl_weight,
                wdl_kl_weight=config.wdl_kl_weight,
                value_weight=config.value_weight,
                moves_left_weight=config.moves_left_weight,
                hard_target_weight=config.hard_target_weight,
                hard_value_weight=config.hard_value_weight,
                hard_moves_left_weight=config.hard_moves_left_weight,
            )
        _accumulate(totals, losses, batch.features.shape[0])
    totals = _reduce_metric_totals(totals, context)
    return _averages(totals)


def _teacher_targets(
    teacher: torch.nn.Module,
    features: torch.Tensor,
    teacher_device: torch.device,
    config: DistillationConfig,
) -> TeacherTargets:
    if teacher_device == features.device:
        teacher_features = features
    else:
        teacher_features = features.to(teacher_device)
    with _autocast_context(teacher_device, config.teacher_amp), torch.no_grad():
        output = teacher(teacher_features)
    targets = teacher_targets_from_output(output, config.temperature)
    if teacher_device != features.device:
        return targets.to(features.device)
    return targets


def _build_loader(
    dataset,
    config: DistillationConfig,
    context: DistributedContext,
    *,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    if context.enabled:
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
        collate_fn=collate_distillation_samples,
        generator=generator if sampler is None else None,
        worker_init_fn=_seed_worker,
        pin_memory=context.device.type == "cuda",
    )
    return loader, sampler


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    torch.manual_seed(seed)


def _configure_reproducibility(config: DistillationConfig) -> None:
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


def _resolve_teacher_device(config: DistillationConfig, student_device: torch.device) -> torch.device:
    if config.teacher_device == "auto":
        return student_device
    return torch.device(config.teacher_device)


def _build_student_model(config: DistillationConfig) -> torch.nn.Module:
    if config.student_kind == "tiny_cnn":
        return build_model(
            "tiny_cnn",
            tiny_channels=config.student_tiny_channels,
            tiny_hidden=config.student_tiny_hidden,
        )
    if config.student_kind == "dense_transformer":
        return build_model(
            "dense_transformer",
            transformer_config=_student_transformer_config(config),
        )
    if config.student_kind == "student_hybrid":
        return build_model(
            "student_hybrid",
            student_hybrid_config=_student_hybrid_config(config),
        )
    raise ValueError(f"unsupported student kind: {config.student_kind}")


def _student_transformer_config(config: DistillationConfig) -> DenseTransformerConfig:
    return DenseTransformerConfig(
        d_model=config.student_transformer_d_model,
        num_layers=config.student_transformer_layers,
        num_heads=config.student_transformer_heads,
        ffn_dim=config.student_transformer_ffn_dim,
        dropout=config.student_transformer_dropout,
        uncertainty_head=config.student_transformer_uncertainty_head,
    )


def _student_hybrid_config(config: DistillationConfig) -> StudentHybridConfig:
    return StudentHybridConfig(
        conv_channels=config.student_hybrid_conv_channels,
        d_model=config.student_hybrid_d_model,
        num_layers=config.student_hybrid_layers,
        num_heads=config.student_hybrid_heads,
        ffn_dim=config.student_hybrid_ffn_dim,
        dropout=config.student_hybrid_dropout,
        layer_norm_eps=config.student_hybrid_layer_norm_eps,
    )


def _build_scheduler(
    config: DistillationConfig,
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


def _to_device(batch: DistillationBatch, device: torch.device) -> DistillationBatch:
    return DistillationBatch(
        features=batch.features.to(device),
        hard_policy=batch.hard_policy.to(device),
        hard_wdl=batch.hard_wdl.to(device),
        hard_moves_left=batch.hard_moves_left.to(device),
        hard_value=batch.hard_value.to(device),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric_totals() -> dict[str, float]:
    return {
        "loss": 0.0,
        "policy_kl": 0.0,
        "wdl_kl": 0.0,
        "value": 0.0,
        "moves_left": 0.0,
        "count": 0.0,
    }


def _accumulate(totals: dict[str, float], losses, batch_size: int) -> None:
    totals["loss"] += float(losses.total.detach().cpu()) * batch_size
    totals["policy_kl"] += float(losses.policy_kl.detach().cpu()) * batch_size
    totals["wdl_kl"] += float(losses.wdl_kl.detach().cpu()) * batch_size
    totals["value"] += float(losses.value.detach().cpu()) * batch_size
    totals["moves_left"] += float(losses.moves_left.detach().cpu()) * batch_size
    totals["count"] += batch_size


def _reduce_metric_totals(
    totals: dict[str, float],
    context: DistributedContext,
) -> dict[str, float]:
    if not context.enabled or not dist.is_initialized():
        return totals
    values = torch.tensor(
        [
            totals["loss"],
            totals["policy_kl"],
            totals["wdl_kl"],
            totals["value"],
            totals["moves_left"],
            totals["count"],
        ],
        device=context.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "loss": float(values[0].item()),
        "policy_kl": float(values[1].item()),
        "wdl_kl": float(values[2].item()),
        "value": float(values[3].item()),
        "moves_left": float(values[4].item()),
        "count": float(values[5].item()),
    }


def _averages(totals: dict[str, float]) -> dict[str, float]:
    count = max(1.0, totals["count"])
    return {
        "loss": totals["loss"] / count,
        "policy_kl": totals["policy_kl"] / count,
        "wdl_kl": totals["wdl_kl"] / count,
        "value": totals["value"] / count,
        "moves_left": totals["moves_left"] / count,
    }


def _append_metrics(path: Path, metrics: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(metrics, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill a teacher into a fast student")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()
    config = load_distillation_config(args.config)
    if args.local_rank is not None:
        config = DistillationConfig(**{**config.__dict__, "local_rank": args.local_rank})
    result = run_distillation(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0 or config.log_all_ranks:
        print(
            f"distillation complete: epochs={result.epochs_completed}, "
            f"last_train_loss={result.train_losses[-1]:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
