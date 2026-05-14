from __future__ import annotations

from pathlib import Path
import json
import pytest

from chessmoe.pipeline.report import generate_run_report, generate_html_report


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


def test_generate_html_report_no_data(tmp_path: Path):
    html = generate_html_report(tmp_path)
    assert "<html" in html
    assert "No run data" in html


def test_generate_html_report_with_data(tmp_path: Path):
    summary = {
        "hardware_profile": "gpu_midrange",
        "quality_profile": "balanced_generation",
        "evaluator": "tensorrt",
        "build_type": "Release",
        "gpu": "RTX 4060",
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
        "hardware_profile": "gpu_midrange",
        "quality_profile": "balanced_generation",
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    html = generate_html_report(tmp_path)
    assert "<!DOCTYPE html>" in html
    assert "chessmoe Run Report" in html
    assert "gpu_midrange" in html
    assert "100" in html
    assert "Terminal Distribution" in html


def test_generate_html_report_with_health(tmp_path: Path):
    summary = {"games_completed": 10, "evaluator": "material",
               "checkmate_count": 5, "stalemate_count": 1,
               "repetition_count": 2, "fifty_move_count": 1,
               "max_plies_count": 1}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    health = {"passed": True, "total_games": 10, "total_samples": 500,
              "average_plies": 50.0, "draw_rate": 0.3, "warnings": ["test warning"]}
    (tmp_path / "replay_health.json").write_text(json.dumps(health))
    html = generate_html_report(tmp_path)
    assert "PASSED" in html
    assert "test warning" in html


def test_generate_html_report_with_profile(tmp_path: Path):
    summary = {"games_completed": 10, "evaluator": "material",
               "checkmate_count": 5, "stalemate_count": 1,
               "repetition_count": 2, "fifty_move_count": 1,
               "max_plies_count": 1}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    profile = {"total_ms": 1000.0, "positions_per_second": 500.0}
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    html = generate_html_report(tmp_path)
    assert "Profile Breakdown" in html
    assert "positions_per_second" in html
