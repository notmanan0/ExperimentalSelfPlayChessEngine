"""Tests for teacher target generation and training support."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from chessmoe.models.encoding import move_to_index, NUM_MOVE_BUCKETS
from chessmoe.training.teacher_data import TeacherTargetDataset, collate_teacher_samples


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_JSONL = _REPO_ROOT / "data" / "teacher" / "test_targets.jsonl"


class TestTeacherTargetDataset:
    def test_loads_jsonl(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        assert len(ds) == 10

    def test_sample_shapes(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        sample = ds[0]
        assert sample.features.shape == (18, 8, 8)
        assert sample.policy.shape == (NUM_MOVE_BUCKETS,)
        assert sample.wdl.shape == (3,)
        assert sample.moves_left.shape == ()

    def test_policy_sums_near_one(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        for i in range(len(ds)):
            sample = ds[i]
            total = sample.policy.sum().item()
            assert abs(total - 1.0) < 0.05, f"Sample {i}: policy sum = {total}"

    def test_wdl_sums_to_one(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        for i in range(len(ds)):
            sample = ds[i]
            assert abs(sample.wdl.sum().item() - 1.0) < 1e-5

    def test_value_in_range(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample.wdl.min() >= 0.0
            assert sample.wdl.max() <= 1.0

    def test_collate(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        batch = collate_teacher_samples([ds[0], ds[1]])
        assert batch["features"].shape == (2, 18, 8, 8)
        assert batch["policy"].shape == (2, NUM_MOVE_BUCKETS)
        assert batch["wdl"].shape == (2, 3)


class TestTeacherTargetValidation:
    def test_policy_indices_are_valid(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        with open(_SAMPLE_JSONL) as f:
            for line in f:
                data = json.loads(line)
                for uci in data["soft_policy"]:
                    idx = move_to_index(uci)
                    assert 0 <= idx < NUM_MOVE_BUCKETS

    def test_policy_values_non_negative(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        with open(_SAMPLE_JSONL) as f:
            for line in f:
                data = json.loads(line)
                for uci, prob in data["soft_policy"].items():
                    assert prob >= 0.0, f"Negative policy for {uci}"

    def test_value_in_valid_range(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        with open(_SAMPLE_JSONL) as f:
            for line in f:
                data = json.loads(line)
                assert -1.0 <= data["value"] <= 1.0

    def test_legal_moves_present(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        with open(_SAMPLE_JSONL) as f:
            for line in f:
                data = json.loads(line)
                assert len(data["legal_moves"]) > 0
                assert len(data["soft_policy"]) > 0


class TestTeacherTrainingStep:
    def test_one_training_step(self) -> None:
        if not _SAMPLE_JSONL.exists():
            pytest.skip("Run generate_teacher_targets.py first")
        from chessmoe.models.tiny_model import TinyChessNet
        from chessmoe.training.losses import compute_tiny_loss

        ds = TeacherTargetDataset(_SAMPLE_JSONL)
        batch = collate_teacher_samples([ds[0], ds[1], ds[2], ds[3]])

        model = TinyChessNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        output = model(batch["features"])

        class Targets:
            pass

        targets = Targets()
        targets.policy = batch["policy"]
        targets.wdl = batch["wdl"]
        targets.moves_left = batch["moves_left"]
        loss_result = compute_tiny_loss(output, targets, moves_left_weight=0.0)

        optimizer.zero_grad()
        loss_result.total.backward()
        optimizer.step()

        assert loss_result.total.item() > 0
        assert torch.isfinite(loss_result.total)
