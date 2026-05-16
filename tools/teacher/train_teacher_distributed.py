"""Distributed training script for teacher bootstrap targets.

Supports multi-GPU training with DistributedDataParallel (DDP).

Usage:
    # Single GPU
    python tools/teacher/train_teacher_distributed.py \
        --teacher-targets data/teacher/pesto_ab_targets.jsonl \
        --config configs/training/dense_teacher_bootstrap.json

    # Multi-GPU (torchrun)
    torchrun --nproc_per_node=2 tools/teacher/train_teacher_distributed.py \
        --teacher-targets data/teacher/pesto_ab_targets.jsonl \
        --config configs/training/dense_teacher_bootstrap.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from chessmoe.models.factory import build_model
from chessmoe.training.teacher_data import TeacherTargetDataset, collate_teacher_samples


def get_device(distributed: bool) -> torch.device:
    if distributed and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def init_distributed() -> tuple[bool, int, int, int]:
    """Initialize distributed training. Returns (enabled, rank, world_size, local_rank)."""
    if not dist.is_available():
        return False, 0, 1, 0

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank

    return False, 0, 1, 0


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def compute_loss(model, batch, device, config):
    """Compute training loss for a batch."""
    features = batch["features"].to(device)
    policy_target = batch["policy"].to(device)
    wdl_target = batch["wdl"].to(device)
    moves_left_target = batch["moves_left"].to(device)

    output = model(features)

    # Policy loss: KL divergence
    log_probs = torch.nn.functional.log_softmax(output.policy_logits, dim=-1)
    policy_loss = -(policy_target * log_probs).sum(dim=-1).mean()

    # WDL loss: cross-entropy
    wdl_loss = torch.nn.functional.cross_entropy(output.wdl_logits, wdl_target)

    # Total loss
    policy_weight = config.get("policy_weight", 1.0)
    wdl_weight = config.get("wdl_weight", 1.0)
    total_loss = policy_weight * policy_loss + wdl_weight * wdl_loss

    return total_loss, {
        "policy": policy_loss.item(),
        "wdl": wdl_loss.item(),
        "total": total_loss.item(),
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_policy = 0.0
    total_wdl = 0.0
    num_batches = 0
    log_interval = config.get("log_interval", 10)

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        loss, details = compute_loss(model, batch, device, config)
        loss.backward()

        # Gradient clipping
        grad_clip = config.get("grad_clip", 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += details["total"]
        total_policy += details["policy"]
        total_wdl += details["wdl"]
        num_batches += 1

        if rank == 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            print(f"  epoch {epoch} batch {batch_idx + 1}: "
                  f"loss={avg_loss:.4f} policy={total_policy/num_batches:.4f} "
                  f"wdl={total_wdl/num_batches:.4f}")

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": total_loss / max(num_batches, 1),
        "policy": total_policy / max(num_batches, 1),
        "wdl": total_wdl / max(num_batches, 1),
    }


def save_checkpoint(model, optimizer, scheduler, epoch, path, rank):
    """Save training checkpoint (rank 0 only)."""
    if rank != 0:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP model
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"  Checkpoint saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed teacher bootstrap training")
    parser.add_argument("--teacher-targets", required=True, help="JSONL teacher targets file")
    parser.add_argument("--config", required=True, help="Training config JSON")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--output", default=None, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")

    args = parser.parse_args()

    # Load config
    config = json.loads(Path(args.config).read_text())

    # Override config with CLI args
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    # Init distributed
    distributed, rank, world_size, local_rank = init_distributed()
    device = get_device(distributed)

    if rank == 0:
        print(f"Teacher bootstrap training")
        print(f"  Distributed: {distributed} (rank={rank}, world_size={world_size})")
        print(f"  Device: {device}")
        print(f"  Config: {args.config}")

    # Load dataset
    dataset = TeacherTargetDataset(args.teacher_targets)
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 64),
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_teacher_samples,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if rank == 0:
        print(f"  Dataset: {len(dataset)} samples")

    # Build model
    model_kind = config.get("model_kind", "dense_transformer")
    model_params = config.get("model_params", {})
    model = build_model(model_kind, **model_params)
    model = model.to(device)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Model: {model_kind} ({param_count:,} parameters)")

    # Wrap with DDP
    if distributed:
        find_unused = model_kind == "moe_transformer"
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    # Optimizer and scheduler
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    scheduler_type = config.get("scheduler", "cosine")
    epochs = config.get("epochs", 10)
    warmup_epochs = config.get("warmup_epochs", 1)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        step_size = config.get("step_size", 5)
        gamma = config.get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Resume from checkpoint
    start_epoch = 0
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            if rank == 0:
                print(f"  Resumed from epoch {start_epoch}")

    # Training loop
    output_path = args.output or config.get("checkpoint_path", "weights/teacher_bootstrap.pt")

    if rank == 0:
        print(f"\nTraining for {epochs} epochs...")
        start_time = time.time()

    for epoch in range(start_epoch, epochs):
        if sampler:
            sampler.set_epoch(epoch)

        epoch_start = time.time()
        metrics = train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch, rank)
        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"loss={metrics['loss']:.4f} "
                  f"policy={metrics['policy']:.4f} "
                  f"wdl={metrics['wdl']:.4f} "
                  f"({epoch_time:.1f}s)")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            save_checkpoint(model, optimizer, scheduler, epoch, output_path, rank)

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"  Output: {output_path}")

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
