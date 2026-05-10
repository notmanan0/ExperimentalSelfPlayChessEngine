from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sqlite3

import torch
from torch.utils.data import Dataset, Subset

from chessmoe.models.encoding import (
    BOARD_SHAPE,
    NUM_MOVE_BUCKETS,
    PIECE_TO_CHANNEL,
    move_to_index,
)
from replay.reader import ReplayReader, ReplaySample


@dataclass(frozen=True)
class TrainingSample:
    features: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor
    root_value: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainingBatch:
    features: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor


class ReplayDataset(Dataset[TrainingSample]):
    def __init__(
        self,
        samples: list[ReplaySample],
        *,
        sample_sources: list[Path] | None = None,
        replay_index: str | Path | None = None,
        target_policy: str = "original",
        reanalysis_fraction: float = 0.0,
        reanalysis_seed: int = 1,
    ) -> None:
        self._samples = samples
        self._sample_sources = sample_sources
        self._replay_index = Path(replay_index) if replay_index is not None else None
        self._target_policy = str(target_policy)
        self._reanalysis_fraction = reanalysis_fraction
        self._reanalysis_seed = reanalysis_seed
        self._moves_left = _moves_left_targets(samples)
        if self._sample_sources is not None and len(self._sample_sources) != len(samples):
            raise ValueError("sample_sources length must match samples length")
        if not 0.0 <= reanalysis_fraction <= 1.0:
            raise ValueError("reanalysis_fraction must be in [0, 1]")

    @classmethod
    def from_index(
        cls,
        db_path: str | Path,
        *,
        target_policy: str = "original",
        reanalysis_fraction: float = 0.0,
        reanalysis_seed: int = 1,
    ) -> "ReplayDataset":
        paths = _chunk_paths_from_index(db_path)
        samples: list[ReplaySample] = []
        sample_sources: list[Path] = []
        for path in paths:
            chunk_samples = ReplayReader.read_file(path).samples
            samples.extend(chunk_samples)
            sample_sources.extend([Path(path)] * len(chunk_samples))
        return cls(
            samples,
            sample_sources=sample_sources,
            replay_index=db_path,
            target_policy=target_policy,
            reanalysis_fraction=reanalysis_fraction,
            reanalysis_seed=reanalysis_seed,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> TrainingSample:
        sample = self._samples[index]
        reanalysis_target = self._target_for_index(index)
        policy = (
            policy_target_from_reanalysis(reanalysis_target)
            if reanalysis_target is not None
            else policy_target_from_replay(sample)
        )
        wdl = (
            reanalysis_wdl_target(reanalysis_target)
            if reanalysis_target is not None
            else wdl_target_from_replay(sample)
        )
        return TrainingSample(
            features=encode_replay_sample(sample),
            policy=policy,
            wdl=torch.tensor(wdl, dtype=torch.long),
            moves_left=torch.tensor(self._moves_left[index], dtype=torch.float32),
            root_value=(
                torch.tensor(reanalysis_target.root_value, dtype=torch.float32)
                if reanalysis_target is not None
                else torch.tensor(sample.root_value, dtype=torch.float32)
            ),
        )

    def _target_for_index(self, index: int):
        policy = self._target_policy
        if policy == "original":
            return None
        if self._replay_index is None or self._sample_sources is None:
            return None

        sample = self._samples[index]
        use_reanalysis = policy == "latest_reanalysis"
        if policy == "mix":
            rng = random.Random(
                f"{self._reanalysis_seed}:{self._sample_sources[index]}:{sample.game_id}:{sample.ply_index}"
            )
            use_reanalysis = rng.random() < self._reanalysis_fraction
        if not use_reanalysis:
            return None

        from replay.reanalysis import load_latest_reanalysis_target

        return load_latest_reanalysis_target(
            self._replay_index,
            self._sample_sources[index],
            game_id=sample.game_id,
            ply_index=sample.ply_index,
        )


def collate_replay_samples(samples: list[TrainingSample]) -> TrainingBatch:
    return TrainingBatch(
        features=torch.stack([sample.features for sample in samples]),
        policy=torch.stack([sample.policy for sample in samples]),
        wdl=torch.stack([sample.wdl for sample in samples]),
        moves_left=torch.stack([sample.moves_left for sample in samples]),
    )


def policy_target_from_reanalysis(target) -> torch.Tensor:
    from replay.reanalysis import reanalysis_policy_target

    return reanalysis_policy_target(target)


def reanalysis_wdl_target(target) -> int:
    from replay.reanalysis import reanalysis_wdl_target as convert

    return convert(target)


def split_dataset(
    dataset: ReplayDataset,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset[TrainingSample], Subset[TrainingSample]]:
    if len(dataset) == 0:
        raise ValueError("replay dataset is empty")
    if train_fraction <= 0.0 or train_fraction > 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if validation_fraction < 0.0:
        raise ValueError("validation_fraction cannot be negative")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    train_count = max(1, int(len(indices) * train_fraction))
    train_count = min(train_count, len(indices))
    validation_count = min(len(indices) - train_count, int(len(indices) * validation_fraction))

    train_indices = indices[:train_count]
    validation_indices = indices[train_count : train_count + validation_count]
    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def encode_replay_sample(sample: ReplaySample) -> torch.Tensor:
    tensor = torch.zeros(BOARD_SHAPE, dtype=torch.float32)
    for square, piece in enumerate(sample.board):
        if piece is None:
            continue
        channel = PIECE_TO_CHANNEL[piece]
        tensor[channel, square // 8, square % 8] = 1.0

    if sample.side_to_move == "white":
        tensor[12].fill_(1.0)
    if sample.castling_rights & 0b0001:
        tensor[13].fill_(1.0)
    if sample.castling_rights & 0b0010:
        tensor[14].fill_(1.0)
    if sample.castling_rights & 0b0100:
        tensor[15].fill_(1.0)
    if sample.castling_rights & 0b1000:
        tensor[16].fill_(1.0)
    if sample.en_passant_square is not None:
        square = _square_to_index(sample.en_passant_square)
        tensor[17, square // 8, square % 8] = 1.0
    return tensor


def policy_target_from_replay(sample: ReplaySample) -> torch.Tensor:
    target = torch.zeros(NUM_MOVE_BUCKETS, dtype=torch.float32)
    total_visits = sum(max(0, entry.visit_count) for entry in sample.policy)
    if total_visits > 0:
        for entry in sample.policy:
            target[move_to_index(entry.move)] = max(0, entry.visit_count) / total_visits
        return target

    total_probability = sum(max(0.0, entry.probability) for entry in sample.policy)
    if total_probability <= 0.0:
        raise ValueError("replay sample has no positive policy mass")
    for entry in sample.policy:
        target[move_to_index(entry.move)] = max(0.0, entry.probability) / total_probability
    return target


def wdl_target_from_replay(sample: ReplaySample) -> int:
    if sample.final_wdl == "draw":
        return 1
    side_won = (
        sample.final_wdl == "white_win"
        and sample.side_to_move == "white"
        or sample.final_wdl == "black_win"
        and sample.side_to_move == "black"
    )
    return 0 if side_won else 2


def _chunk_paths_from_index(db_path: str | Path) -> list[Path]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "select path from chunks order by creation_timestamp_ms, path"
        ).fetchall()
    if not rows:
        raise ValueError("replay index contains no chunks")
    return [Path(row[0]) for row in rows]


def _moves_left_targets(samples: list[ReplaySample]) -> list[float]:
    last_ply_by_game: dict[int, int] = {}
    for sample in samples:
        last_ply_by_game[sample.game_id] = max(
            sample.ply_index, last_ply_by_game.get(sample.game_id, sample.ply_index)
        )
    return [
        float(max(1, last_ply_by_game[sample.game_id] - sample.ply_index + 1))
        for sample in samples
    ]


def _square_to_index(square: str) -> int:
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"invalid square: {square}")
    return (int(square[1]) - 1) * 8 + (ord(square[0]) - ord("a"))
