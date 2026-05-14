from __future__ import annotations

from pathlib import Path
import json
import pytest

from chessmoe.pipeline.report import generate_run_report, generate_registry_report


def test_generate_run_report_no_summary(tmp_path: Path):
    report = generate_run_report(tmp_path)
    assert "No summary found" in report


def test_generate_run_report_with_summary(tmp_path: Path):
    summary = {
        "hardware_profile": "gpu_midrange",
        "quality_profile": "balanced_generation",
        "evaluator": "tensorrt",
        "build_type": "Release",
        "gpu": "RTX 4060",
        "debug_build": False,
        "games_completed": 100,
        "samples_written": 5000,
        "games_per_second": 2.5,
        "positions_per_second": 160.0,
        "average_plies_per_game": 50.0,
        "elapsed_ms": 40000.0,
        "checkmate_count": 30,
        "stalemate_count": 5,
        "repetition_count": 10,
        "fifty_move_count": 5,
        "max_plies_count": 50,
        "batch_fill_ratio": 0.85,
        "padding_ratio": 0.15,
        "avg_inference_latency_ms": 3.5,
        "replay_chunks": 4,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    report = generate_run_report(tmp_path)
    assert "gpu_midrange" in report
    assert "100" in report
    assert "5000" in report


def test_generate_run_report_with_health(tmp_path: Path):
    summary = {"games_completed": 10, "evaluator": "material"}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    health = {
        "passed": True,
        "total_games": 10,
        "total_samples": 500,
        "average_plies": 50.0,
        "draw_rate": 0.3,
        "warnings": ["test warning"],
    }
    (tmp_path / "replay_health.json").write_text(json.dumps(health))
    report = generate_run_report(tmp_path)
    assert "PASSED" in report
    assert "test warning" in report
