from __future__ import annotations

from dataclasses import dataclass
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
    model_channels: int = 32
    model_hidden: int = 128
    resume_checkpoint: str | Path | None = None


def load_training_config(path: str | Path) -> TrainingConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(**raw)


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    result = dict(config.__dict__)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result
