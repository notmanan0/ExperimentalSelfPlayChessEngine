from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from chessmoe.training.config import TrainingConfig, load_training_config
from chessmoe.training.data import ReplayDataset, collate_replay_samples, split_dataset
from chessmoe.training.distributed import (
    barrier,
    ddp_wrap,
    destroy_distributed,
    init_distributed,
    reduce_scalar,
    should_log,
)
from chessmoe.training.losses import TinyLossTargets, compute_moe_aware_loss, compute_tiny_loss
from chessmoe.training.train import (
    _autocast_context,
    _build_configured_model,
    _build_scaler,
    _resolve_grad_accum_steps,
    _resolve_device,
    _to_device,
)


def _build_loader(
    dataset,
    config: TrainingConfig,
    context,
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
        collate_fn=collate_replay_samples,
        generator=generator if sampler is None else None,
        pin_memory=context.device.type == "cuda",
    )
    return loader, sampler


def _no_sync_context(model: torch.nn.Module, enabled: bool):
    if enabled and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()


def _train_steps(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    context,
    config: TrainingConfig,
    steps: int,
) -> int:
    is_moe = config.model_kind == "moe_transformer"
    accum_steps = _resolve_grad_accum_steps(config.grad_accum_steps)
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    processed = 0
    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = _to_device(batch, context.device)
        is_last_micro = (step + 1) % accum_steps == 0
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
        processed += batch.features.shape[0]
    return processed


def run_benchmark(config: TrainingConfig, steps: int, warmup: int) -> None:
    base_device = _resolve_device(config.device)
    context = init_distributed(config, base_device)
    try:
        dataset = ReplayDataset.from_index(
            config.replay_index,
            target_policy=config.target_policy,
            reanalysis_fraction=config.reanalysis_fraction,
            reanalysis_seed=config.reanalysis_seed,
        )
        train_data, _ = split_dataset(
            dataset,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )

        loader, sampler = _build_loader(train_data, config, context, shuffle=True)
        if sampler is not None:
            sampler.set_epoch(0)

        model = _build_configured_model(config).to(context.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scaler = _build_scaler(context.device, config.amp)
        model = ddp_wrap(model, context, config)
        model.train()

        if warmup > 0:
            _train_steps(model, loader, optimizer, scaler, context, config, warmup)

        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        barrier(context)

        start = time.perf_counter()
        samples = _train_steps(model, loader, optimizer, scaler, context, config, steps)
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        barrier(context)
        elapsed = time.perf_counter() - start

        total_samples = reduce_scalar(float(samples), context, op=dist.ReduceOp.SUM)
        max_elapsed = reduce_scalar(float(elapsed), context, op=dist.ReduceOp.MAX)
        if should_log(context, config.log_all_ranks):
            throughput = total_samples / max_elapsed if max_elapsed > 0 else 0.0
            print(
                "throughput: "
                f"samples={int(total_samples)}, "
                f"elapsed_s={max_elapsed:.3f}, "
                f"samples_per_s={throughput:.2f}"
            )
    finally:
        destroy_distributed(context)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark training throughput")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.local_rank is not None:
        config = TrainingConfig(**{**config.__dict__, "local_rank": args.local_rank})

    run_benchmark(config, steps=args.steps, warmup=args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
