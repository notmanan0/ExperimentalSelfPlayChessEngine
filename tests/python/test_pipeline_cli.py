from __future__ import annotations

import argparse
from pathlib import Path
import pytest

from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile
import tools.chessmoe as cli


def test_pipeline_config_creation():
    hw = load_hardware_profile("cpu_bootstrap_debug")
    q = load_quality_profile("debug_smoke")
    config = PipelineConfig(phase=0, hardware_profile=hw, quality_profile=q)
    assert config.phase == 0
    assert config.hardware_profile.name == "cpu_bootstrap_debug"
    assert config.quality_profile.name == "debug_smoke"


def test_pipeline_runner_creation():
    hw = load_hardware_profile("cpu_bootstrap_debug")
    q = load_quality_profile("debug_smoke")
    config = PipelineConfig(phase=0, hardware_profile=hw, quality_profile=q)
    runner = PipelineRunner(config)
    assert len(runner.results) == 0


def test_pipeline_summary_empty():
    hw = load_hardware_profile("cpu_bootstrap_debug")
    q = load_quality_profile("debug_smoke")
    config = PipelineConfig(phase=0, hardware_profile=hw, quality_profile=q)
    runner = PipelineRunner(config)
    summary = runner.summary()
    assert "Pipeline Summary" in summary


def test_full_cycle_passes_all_cli_flags_to_pipeline_config(monkeypatch):
    captured: dict[str, PipelineConfig] = {}

    class CapturingRunner:
        def __init__(self, config: PipelineConfig):
            captured["config"] = config

        def run_full_cycle(self):
            return []

        def summary(self):
            return "captured"

    monkeypatch.setattr("chessmoe.pipeline.runner.PipelineRunner", CapturingRunner)

    args = argparse.Namespace(
        phase=18,
        hardware_profile="cpu_bootstrap_debug",
        quality="debug_smoke",
        engine="weights/engine.plan",
        train_config="configs/training/tiny_replay.json",
        checkpoint="weights/checkpoint.pt",
        onnx_output="weights/candidate.onnx",
        engine_output="weights/candidate.engine",
        arena_config="configs/arena/tiny_arena.json",
        candidate="weights/candidate.pt",
        best="weights/best.pt",
        skip_engine_build=True,
        skip_promotion=True,
        allow_debug=True,
    )

    assert cli.cmd_full_cycle(args) == 0

    config = captured["config"]
    assert config.phase == 18
    assert config.hardware_profile.name == "cpu_bootstrap_debug"
    assert config.quality_profile.name == "debug_smoke"
    assert config.engine_path == Path("weights/engine.plan")
    assert config.train_config == "configs/training/tiny_replay.json"
    assert config.checkpoint == Path("weights/checkpoint.pt")
    assert config.onnx_output == Path("weights/candidate.onnx")
    assert config.engine_output == Path("weights/candidate.engine")
    assert config.arena_config == "configs/arena/tiny_arena.json"
    assert config.candidate == Path("weights/candidate.pt")
    assert config.best == Path("weights/best.pt")
    assert config.skip_engine_build is True
    assert config.skip_promotion is True
    assert config.allow_debug is True
