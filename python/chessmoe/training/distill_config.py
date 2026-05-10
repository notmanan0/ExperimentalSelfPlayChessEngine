from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DistillationConfig:
    replay_index: str | Path
    checkpoint_path: str | Path
    metrics_path: str | Path
    teacher_checkpoint: str | Path
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    train_fraction: float = 0.9
    validation_fraction: float = 0.1
    seed: int = 1
    device: str = "auto"
    teacher_device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    teacher_amp: bool = False
    compile_model: bool = False
    grad_accum_steps: int = 1
    deterministic: bool = False
    cudnn_benchmark: bool = False
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    target_policy: str = "latest_reanalysis"
    reanalysis_fraction: float = 1.0
    reanalysis_seed: int = 1
    distributed: bool = False
    distributed_backend: str = "auto"
    distributed_init_method: str = "env://"
    distributed_timeout_sec: int = 1800
    rank: int | None = None
    world_size: int | None = None
    local_rank: int | None = None
    log_all_ranks: bool = False
    ddp_find_unused_parameters: bool | None = None
    ddp_static_graph: bool = False
    ddp_bucket_cap_mb: int | None = None
    fsdp_enabled: bool = False
    student_kind: str = "student_hybrid"
    student_tiny_channels: int = 32
    student_tiny_hidden: int = 128
    student_transformer_d_model: int = 64
    student_transformer_layers: int = 2
    student_transformer_heads: int = 4
    student_transformer_ffn_dim: int = 128
    student_transformer_dropout: float = 0.1
    student_transformer_uncertainty_head: bool = False
    student_hybrid_conv_channels: int = 32
    student_hybrid_d_model: int = 96
    student_hybrid_layers: int = 2
    student_hybrid_heads: int = 4
    student_hybrid_ffn_dim: int = 192
    student_hybrid_dropout: float = 0.1
    student_hybrid_layer_norm_eps: float = 1.0e-5
    temperature: float = 2.0
    policy_kl_weight: float = 1.0
    wdl_kl_weight: float = 1.0
    value_weight: float = 0.25
    moves_left_weight: float = 0.05
    hard_target_weight: float = 0.1
    hard_value_weight: float = 0.1
    hard_moves_left_weight: float = 0.0
    resume_checkpoint: str | Path | None = None


def load_distillation_config(path: str | Path) -> DistillationConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DistillationConfig(**raw)


def distillation_config_to_dict(config: DistillationConfig) -> dict[str, Any]:
    result = dict(config.__dict__)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result
