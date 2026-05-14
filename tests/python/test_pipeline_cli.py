from __future__ import annotations

from pathlib import Path
import pytest

from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile


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
