from __future__ import annotations

from pathlib import Path
import json
import pytest

from chessmoe.pipeline.config import (
    BUILTIN_HARDWARE,
    BUILTIN_QUALITY,
    HardwareProfile,
    QualityProfile,
    load_hardware_profile,
    load_quality_profile,
    list_hardware_profiles,
    list_quality_profiles,
)


def test_builtin_hardware_profiles_exist():
    assert len(BUILTIN_HARDWARE) >= 6
    assert "cpu_bootstrap_debug" in BUILTIN_HARDWARE
    assert "gpu_midrange" in BUILTIN_HARDWARE
    assert "gpu_datacenter" in BUILTIN_HARDWARE


def test_builtin_quality_profiles_exist():
    assert len(BUILTIN_QUALITY) >= 5
    assert "fast_bootstrap" in BUILTIN_QUALITY
    assert "balanced_generation" in BUILTIN_QUALITY
    assert "debug_smoke" in BUILTIN_QUALITY


def test_load_hardware_profile_builtin():
    p = load_hardware_profile("cpu_bootstrap_debug")
    assert p.name == "cpu_bootstrap_debug"
    assert p.evaluator == "material"
    assert p.concurrent_games == 4


def test_load_hardware_profile_gpu():
    p = load_hardware_profile("gpu_midrange")
    assert p.evaluator == "tensorrt"
    assert p.fixed_batch == 64
    assert p.visits == 128


def test_load_hardware_profile_unknown():
    with pytest.raises(ValueError, match="unknown hardware profile"):
        load_hardware_profile("nonexistent_profile")


def test_load_quality_profile_builtin():
    q = load_quality_profile("balanced_generation")
    assert q.visits == 128
    assert q.max_plies == 200
    assert q.root_dirichlet_noise is True


def test_load_quality_profile_debug():
    q = load_quality_profile("debug_smoke")
    assert q.visits == 4
    assert q.games == 4


def test_load_quality_profile_unknown():
    with pytest.raises(ValueError, match="unknown quality profile"):
        load_quality_profile("nonexistent_profile")


def test_list_hardware_profiles():
    names = list_hardware_profiles()
    assert "cpu_bootstrap_debug" in names
    assert "gpu_midrange" in names


def test_list_quality_profiles():
    names = list_quality_profiles()
    assert "balanced_generation" in names
    assert "debug_smoke" in names


def test_load_from_json_file(tmp_path: Path):
    config_dir = tmp_path / "profiles"
    config_dir.mkdir()
    hw_data = {
        "custom_profile": {
            "evaluator": "material",
            "concurrent_games": 8,
            "fixed_batch": 16,
            "visits": 16,
            "max_plies": 64,
            "description": "custom test profile",
        }
    }
    (config_dir / "hardware.json").write_text(json.dumps(hw_data))
    p = load_hardware_profile("custom_profile", config_dir)
    assert p.name == "custom_profile"
    assert p.concurrent_games == 8


def test_hardware_profile_dataclass():
    p = HardwareProfile(name="test", evaluator="material")
    assert p.name == "test"
    assert p.evaluator == "material"
    assert p.concurrent_games == 1


def test_quality_profile_dataclass():
    q = QualityProfile(name="test", visits=64)
    assert q.name == "test"
    assert q.visits == 64
    assert q.root_dirichlet_noise is True
