from __future__ import annotations

from dataclasses import dataclass
import datetime
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    backend: str
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def _env_int(key: str, default: int) -> int:
    value = os.environ.get(key)
    if value is None or value == "":
        return default
    return int(value)


def _resolve_backend(config, device: torch.device) -> str:
    backend = str(config.distributed_backend)
    if backend == "auto":
        return "nccl" if device.type == "cuda" else "gloo"
    return backend


def init_distributed(config, device: torch.device) -> DistributedContext:
    if not config.distributed:
        return DistributedContext(False, "", 0, 1, 0, device)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")

    rank = _env_int("RANK", int(config.rank) if config.rank is not None else 0)
    world_size = _env_int(
        "WORLD_SIZE",
        int(config.world_size) if config.world_size is not None else 1,
    )
    local_rank = _env_int(
        "LOCAL_RANK",
        int(config.local_rank) if config.local_rank is not None else 0,
    )
    backend = _resolve_backend(config, device)

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method=str(config.distributed_init_method),
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=int(config.distributed_timeout_sec)),
        )

    return DistributedContext(True, backend, rank, world_size, local_rank, device)


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.enabled and dist.is_initialized():
        dist.barrier()


def should_log(context: DistributedContext, log_all_ranks: bool) -> bool:
    return log_all_ranks or context.rank == 0


def ddp_wrap(model: torch.nn.Module, context: DistributedContext, config) -> torch.nn.Module:
    if not context.enabled:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    output_device = context.local_rank if context.device.type == "cuda" else None
    find_unused = config.ddp_find_unused_parameters
    if find_unused is None:
        find_unused = config.model_kind == "moe_transformer"
    return DDP(
        model,
        device_ids=device_ids,
        output_device=output_device,
        find_unused_parameters=find_unused,
        static_graph=bool(config.ddp_static_graph),
        bucket_cap_mb=config.ddp_bucket_cap_mb,
    )


def reduce_metric_totals(
    totals: dict[str, float],
    context: DistributedContext,
) -> dict[str, float]:
    if not context.enabled or not dist.is_initialized():
        return totals
    values = torch.tensor(
        [
            totals["loss"],
            totals["policy"],
            totals["wdl"],
            totals["moves_left"],
            totals["count"],
        ],
        device=context.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "loss": float(values[0].item()),
        "policy": float(values[1].item()),
        "wdl": float(values[2].item()),
        "moves_left": float(values[3].item()),
        "count": float(values[4].item()),
    }


def reduce_scalar(
    value: float,
    context: DistributedContext,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> float:
    if not context.enabled or not dist.is_initialized():
        return value
    tensor = torch.tensor([value], device=context.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    source = model
    if hasattr(source, "module"):
        source = source.module
    if hasattr(source, "_orig_mod"):
        source = source._orig_mod
    return source
