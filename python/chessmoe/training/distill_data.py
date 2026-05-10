from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, Subset

from chessmoe.training.data import ReplayDataset, TrainingSample


@dataclass(frozen=True)
class DistillationSample:
    features: torch.Tensor
    hard_policy: torch.Tensor
    hard_wdl: torch.Tensor
    hard_moves_left: torch.Tensor
    hard_value: torch.Tensor


@dataclass(frozen=True)
class DistillationBatch:
    features: torch.Tensor
    hard_policy: torch.Tensor
    hard_wdl: torch.Tensor
    hard_moves_left: torch.Tensor
    hard_value: torch.Tensor


class DistillationDataset(Dataset[DistillationSample]):
    def __init__(self, dataset: ReplayDataset) -> None:
        self._dataset = dataset

    @classmethod
    def from_index(
        cls,
        db_path: str | Path,
        *,
        target_policy: str = "latest_reanalysis",
        reanalysis_fraction: float = 1.0,
        reanalysis_seed: int = 1,
    ) -> "DistillationDataset":
        base = ReplayDataset.from_index(
            db_path,
            target_policy=target_policy,
            reanalysis_fraction=reanalysis_fraction,
            reanalysis_seed=reanalysis_seed,
        )
        return cls(base)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> DistillationSample:
        sample: TrainingSample = self._dataset[index]
        root_value = sample.root_value
        if root_value is None:
            root_value = torch.tensor(0.0, dtype=torch.float32)
        return DistillationSample(
            features=sample.features,
            hard_policy=sample.policy,
            hard_wdl=sample.wdl,
            hard_moves_left=sample.moves_left,
            hard_value=root_value,
        )


def collate_distillation_samples(samples: list[DistillationSample]) -> DistillationBatch:
    return DistillationBatch(
        features=torch.stack([sample.features for sample in samples]),
        hard_policy=torch.stack([sample.hard_policy for sample in samples]),
        hard_wdl=torch.stack([sample.hard_wdl for sample in samples]),
        hard_moves_left=torch.stack([sample.hard_moves_left for sample in samples]),
        hard_value=torch.stack([sample.hard_value for sample in samples]),
    )


def split_distillation_dataset(
    dataset: DistillationDataset,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset[DistillationSample], Subset[DistillationSample]]:
    from chessmoe.training.data import split_dataset

    return split_dataset(dataset, train_fraction, validation_fraction, seed)
