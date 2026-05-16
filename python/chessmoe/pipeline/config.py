from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class HardwareProfile:
    name: str
    evaluator: str = "material"
    concurrent_games: int = 1
    fixed_batch: int = 64
    max_pending_requests: int = 4096
    flush_ms: int = 2
    visits: int = 64
    max_plies: int = 128
    precision: str = "fp32"
    cpu_workers: int = 0
    replay_chunk_games: int = 64
    progress_interval: int = 10
    description: str = ""


@dataclass
class QualityProfile:
    name: str
    visits: int = 64
    max_plies: int = 128
    root_dirichlet_noise: bool = True
    root_dirichlet_alpha: float = 0.3
    root_dirichlet_epsilon: float = 0.25
    temperature_initial: float = 1.0
    temperature_final: float = 0.0
    temperature_cutoff_ply: int = 30
    games: int = 256
    arena_games: int = 64
    promotion_threshold: float = 0.55
    resignation_enabled: bool = False
    description: str = ""


BUILTIN_HARDWARE: dict[str, HardwareProfile] = {
    "cpu_bootstrap_debug": HardwareProfile(
        name="cpu_bootstrap_debug",
        evaluator="material",
        concurrent_games=4,
        fixed_batch=1,
        max_pending_requests=64,
        flush_ms=10,
        visits=8,
        max_plies=32,
        precision="fp32",
        cpu_workers=2,
        replay_chunk_games=16,
        progress_interval=4,
        description="Minimal CPU-only debug run with material evaluator",
    ),
    "cpu_bootstrap_fast": HardwareProfile(
        name="cpu_bootstrap_fast",
        evaluator="material",
        concurrent_games=16,
        fixed_batch=1,
        max_pending_requests=256,
        flush_ms=5,
        visits=32,
        max_plies=128,
        precision="fp32",
        cpu_workers=0,
        replay_chunk_games=64,
        progress_interval=16,
        description="Fast CPU-only bootstrap with material evaluator",
    ),
    "cpu_pesto_bootstrap": HardwareProfile(
        name="cpu_pesto_bootstrap",
        evaluator="pesto",
        concurrent_games=16,
        fixed_batch=1,
        max_pending_requests=256,
        flush_ms=5,
        visits=32,
        max_plies=160,
        precision="fp32",
        cpu_workers=0,
        replay_chunk_games=64,
        progress_interval=16,
        description="CPU bootstrap with PeSTO tapered evaluator for stronger initial data",
    ),
    "gpu_low_vram": HardwareProfile(
        name="gpu_low_vram",
        evaluator="tensorrt",
        concurrent_games=32,
        fixed_batch=32,
        max_pending_requests=2048,
        flush_ms=2,
        visits=64,
        max_plies=160,
        precision="fp16",
        cpu_workers=0,
        replay_chunk_games=128,
        progress_interval=16,
        description="Low VRAM GPU (4 GB or less)",
    ),
    "gpu_midrange": HardwareProfile(
        name="gpu_midrange",
        evaluator="tensorrt",
        concurrent_games=96,
        fixed_batch=64,
        max_pending_requests=4096,
        flush_ms=2,
        visits=128,
        max_plies=200,
        precision="fp16",
        cpu_workers=0,
        replay_chunk_games=256,
        progress_interval=25,
        description="Mid-range GPU (6-10 GB VRAM, e.g. RTX 4060)",
    ),
    "gpu_highend": HardwareProfile(
        name="gpu_highend",
        evaluator="tensorrt",
        concurrent_games=128,
        fixed_batch=128,
        max_pending_requests=8192,
        flush_ms=2,
        visits=200,
        max_plies=256,
        precision="fp16",
        cpu_workers=0,
        replay_chunk_games=512,
        progress_interval=32,
        description="High-end GPU (12-24 GB VRAM, e.g. RTX 4080/4090)",
    ),
    "gpu_datacenter": HardwareProfile(
        name="gpu_datacenter",
        evaluator="tensorrt",
        concurrent_games=256,
        fixed_batch=256,
        max_pending_requests=16384,
        flush_ms=1,
        visits=400,
        max_plies=300,
        precision="fp16",
        cpu_workers=0,
        replay_chunk_games=1024,
        progress_interval=64,
        description="Datacenter GPU (40+ GB VRAM, e.g. A100/H100)",
    ),
}

BUILTIN_QUALITY: dict[str, QualityProfile] = {
    "fast_bootstrap": QualityProfile(
        name="fast_bootstrap",
        visits=32,
        max_plies=128,
        root_dirichlet_noise=True,
        root_dirichlet_alpha=0.3,
        root_dirichlet_epsilon=0.25,
        temperature_initial=1.0,
        temperature_final=0.0,
        temperature_cutoff_ply=20,
        games=256,
        arena_games=32,
        promotion_threshold=0.55,
        resignation_enabled=False,
        description="Fast bootstrap generation for initial training data",
    ),
    "balanced_generation": QualityProfile(
        name="balanced_generation",
        visits=128,
        max_plies=200,
        root_dirichlet_noise=True,
        root_dirichlet_alpha=0.3,
        root_dirichlet_epsilon=0.25,
        temperature_initial=1.0,
        temperature_final=0.0,
        temperature_cutoff_ply=30,
        games=2048,
        arena_games=128,
        promotion_threshold=0.55,
        resignation_enabled=False,
        description="Balanced generation for steady improvement",
    ),
    "high_quality_generation": QualityProfile(
        name="high_quality_generation",
        visits=400,
        max_plies=300,
        root_dirichlet_noise=True,
        root_dirichlet_alpha=0.3,
        root_dirichlet_epsilon=0.25,
        temperature_initial=1.0,
        temperature_final=0.0,
        temperature_cutoff_ply=40,
        games=4096,
        arena_games=256,
        promotion_threshold=0.55,
        resignation_enabled=True,
        description="High-quality generation for strong models",
    ),
    "arena_eval": QualityProfile(
        name="arena_eval",
        visits=200,
        max_plies=300,
        root_dirichlet_noise=False,
        root_dirichlet_alpha=0.0,
        root_dirichlet_epsilon=0.0,
        temperature_initial=0.0,
        temperature_final=0.0,
        temperature_cutoff_ply=0,
        games=0,
        arena_games=256,
        promotion_threshold=0.55,
        resignation_enabled=False,
        description="Arena evaluation settings with deterministic play",
    ),
    "debug_smoke": QualityProfile(
        name="debug_smoke",
        visits=4,
        max_plies=20,
        root_dirichlet_noise=False,
        root_dirichlet_alpha=0.0,
        root_dirichlet_epsilon=0.0,
        temperature_initial=1.0,
        temperature_final=0.0,
        temperature_cutoff_ply=10,
        games=4,
        arena_games=4,
        promotion_threshold=0.55,
        resignation_enabled=False,
        description="Minimal smoke test for pipeline validation",
    ),
}


def _load_json_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def load_hardware_profile(name: str, config_dir: Path | None = None) -> HardwareProfile:
    if name in BUILTIN_HARDWARE:
        return BUILTIN_HARDWARE[name]
    config_dir = config_dir or Path("configs/profiles")
    file_profiles = _load_json_profiles(config_dir / "hardware.json")
    if name in file_profiles:
        return HardwareProfile(name=name, **file_profiles[name])
    available = list(BUILTIN_HARDWARE.keys())
    raise ValueError(
        f"unknown hardware profile: {name}; available: {available}"
    )


def load_quality_profile(name: str, config_dir: Path | None = None) -> QualityProfile:
    if name in BUILTIN_QUALITY:
        return BUILTIN_QUALITY[name]
    config_dir = config_dir or Path("configs/profiles")
    file_profiles = _load_json_profiles(config_dir / "quality.json")
    if name in file_profiles:
        return QualityProfile(name=name, **file_profiles[name])
    available = list(BUILTIN_QUALITY.keys())
    raise ValueError(
        f"unknown quality profile: {name}; available: {available}"
    )


def list_hardware_profiles() -> list[str]:
    return list(BUILTIN_HARDWARE.keys())


def list_quality_profiles() -> list[str]:
    return list(BUILTIN_QUALITY.keys())
