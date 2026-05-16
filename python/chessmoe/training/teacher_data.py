"""Dataset for loading teacher-generated targets (PeSTO alpha-beta, Stockfish, etc)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from chessmoe.models.encoding import (
    BOARD_SHAPE,
    NUM_MOVE_BUCKETS,
    encode_fen,
    move_to_index,
)


@dataclass(frozen=True)
class TeacherTarget:
    fen: str
    legal_moves: list[str]
    soft_policy: dict[str, float]
    value: float
    depth: int
    nodes: int
    teacher: str
    teacher_version: int


@dataclass(frozen=True)
class TeacherSample:
    features: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor


class TeacherTargetDataset(Dataset[TeacherSample]):
    def __init__(self, jsonl_path: str | Path) -> None:
        self._path = Path(jsonl_path)
        self._targets: list[TeacherTarget] = []
        self._load()

    def _load(self) -> None:
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._targets.append(TeacherTarget(
                    fen=data["fen"],
                    legal_moves=data["legal_moves"],
                    soft_policy=data["soft_policy"],
                    value=data["value"],
                    depth=data.get("depth", 0),
                    nodes=data.get("nodes", 0),
                    teacher=data.get("teacher", "unknown"),
                    teacher_version=data.get("teacher_version", 0),
                ))

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> TeacherSample:
        target = self._targets[index]
        features = encode_fen(target.fen)

        policy = torch.zeros(NUM_MOVE_BUCKETS, dtype=torch.float32)
        total = 0.0
        for uci, prob in target.soft_policy.items():
            idx = move_to_index(uci)
            policy[idx] = prob
            total += prob
        if total > 0:
            policy /= total

        value = max(-1.0, min(1.0, target.value))
        if value > 0:
            wdl = torch.tensor([value, 1.0 - value, 0.0], dtype=torch.float32)
        elif value < 0:
            wdl = torch.tensor([0.0, 1.0 - abs(value), abs(value)], dtype=torch.float32)
        else:
            wdl = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        wdl = wdl / wdl.sum()

        return TeacherSample(
            features=features,
            policy=policy,
            wdl=wdl,
            moves_left=torch.tensor(40.0, dtype=torch.float32),
        )


def collate_teacher_samples(samples: list[TeacherSample]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.stack([s.features for s in samples]),
        "policy": torch.stack([s.policy for s in samples]),
        "wdl": torch.stack([s.wdl for s in samples]),
        "moves_left": torch.stack([s.moves_left for s in samples]),
    }
