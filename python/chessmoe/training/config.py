from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class TrainingConfig:
    replay_index: str | Path
    checkpoint_path: str | Path
    metrics_path: str | Path
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    train_fraction: float = 0.9
    validation_fraction: float = 0.1
    seed: int = 1
    device: str = "auto"
    num_workers: int = 0
    amp: bool = False
    compile_model: bool = False
    moves_left_weight: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    target_policy: str = "original"
    reanalysis_fraction: float = 0.0
    reanalysis_seed: int = 1
    model_kind: str = "tiny_cnn"
    model_channels: int = 32
    model_hidden: int = 128
    transformer_d_model: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1
    transformer_uncertainty_head: bool = False
    resume_checkpoint: str | Path | None = None
    moe_num_experts: int = 8
    moe_top_k_training: int = 2
    moe_top_k_inference: int = 1
    moe_capacity_factor: float = 1.25
    moe_load_balance_coeff: float = 0.01
    moe_router_entropy_coeff: float = 0.001
    moe_router_noise_std: float = 0.1
    moe_dense_fallback: bool = False
    moe_expert_dropout: float = 0.0
    moe_layers: list[int] = field(default_factory=lambda: [1, 3])


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(**raw)


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    result = dict(config.__dict__)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result
