from __future__ import annotations

from pathlib import Path
import json
import pytest

from chessmoe.models.registry import (
    ModelRegistry,
    PromotionStatus,
    RegistryEntry,
    promote_candidate,
)


def test_registry_entry_roundtrip():
    entry = RegistryEntry(
        model_version=1,
        checkpoint_path="weights/test.pt",
        notes="test",
    )
    d = entry.to_dict()
    restored = RegistryEntry.from_dict(d)
    assert restored.model_version == 1
    assert restored.checkpoint_path == "weights/test.pt"
    assert restored.notes == "test"


def test_registry_register_and_get(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    entry = RegistryEntry(model_version=1, checkpoint_path="test.pt")
    reg.register(entry)
    got = reg.get_entry(1)
    assert got is not None
    assert got.model_version == 1


def test_registry_set_best(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.register(RegistryEntry(model_version=2, checkpoint_path="v2.pt"))
    reg.set_best(2)
    assert reg.get_best_version() == 2
    best = reg.get_best()
    assert best is not None
    assert best.model_version == 2


def test_registry_promote(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.set_best(1)
    reg.register(RegistryEntry(model_version=2, checkpoint_path="v2.pt"))
    reg.promote(2, {"score_rate": 0.6, "games": 128})
    assert reg.get_best_version() == 2
    entry = reg.get_entry(2)
    assert entry is not None
    assert entry.promotion_status == "promoted"
    assert entry.arena_result is not None


def test_registry_reject(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.reject(1, {"score_rate": 0.4, "games": 128})
    entry = reg.get_entry(1)
    assert entry is not None
    assert entry.promotion_status == "rejected"


def test_registry_refuse_without_arena(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    with pytest.raises(RuntimeError, match="no arena result"):
        reg.refuse_promotion_without_arena(1)


def test_registry_refuse_with_force(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.refuse_promotion_without_arena(1, force=True)


def test_registry_refuse_rejected(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.reject(1, {"score_rate": 0.4})
    with pytest.raises(RuntimeError, match="previously rejected"):
        reg.refuse_promotion_without_arena(1)


def test_registry_list_entries(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1))
    reg.register(RegistryEntry(model_version=2))
    reg.register(RegistryEntry(model_version=3))
    entries = reg.list_entries()
    assert len(entries) == 3


def test_registry_format(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.register(RegistryEntry(model_version=1, checkpoint_path="v1.pt"))
    reg.set_best(1)
    text = reg.format_registry()
    assert "Model Registry" in text
    assert "version 1" in text
    assert "[BEST]" in text


def test_promote_candidate(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    candidate = weights / "candidate.pt"
    candidate.write_text("test")
    copied = promote_candidate(candidate, 1, weights_dir=weights)
    assert len(copied) >= 2
    assert (weights / "best.pt").exists()
    assert (weights / "history" / "model_000001.pt").exists()


def test_promote_candidate_no_existing_best(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    candidate = weights / "candidate.pt"
    candidate.write_text("test")
    copied = promote_candidate(candidate, 1, weights_dir=weights)
    assert (weights / "best.pt").exists()


def test_promote_candidate_archives_old(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "best.pt").write_text("old")
    candidate = weights / "candidate.pt"
    candidate.write_text("new")
    copied = promote_candidate(candidate, 2, weights_dir=weights)
    assert (weights / "history" / "model_000001.pt").exists()
    assert (weights / "history" / "model_000002.pt").exists()
