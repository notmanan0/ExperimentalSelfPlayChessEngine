from __future__ import annotations

from pathlib import Path
import json
import pytest

from chessmoe.analysis.calibration import (
    CalibrationPoint,
    CalibrationResult,
    print_calibration_summary,
    save_calibration_result,
)


def test_calibration_point():
    p = CalibrationPoint(
        concurrent_games=64, fixed_batch=32, flush_ms=2,
        positions_per_sec=500.0, games_per_sec=2.0,
        samples_per_sec=100.0, batch_fill_ratio=0.85,
        avg_inference_latency_ms=3.5,
    )
    assert p.concurrent_games == 64
    assert p.positions_per_sec == 500.0


def test_calibration_result():
    points = [
        CalibrationPoint(64, 32, 2, 500.0, 2.0, 100.0, 0.85, 3.5),
        CalibrationPoint(96, 64, 1, 600.0, 2.5, 120.0, 0.90, 3.0),
    ]
    result = CalibrationResult(
        hardware_profile="gpu_midrange",
        quality_profile="debug_smoke",
        points=points,
        recommended=points[1],
    )
    assert len(result.points) == 2
    assert result.recommended is not None
    assert result.recommended.positions_per_sec == 600.0


def test_save_calibration_result(tmp_path: Path):
    points = [
        CalibrationPoint(64, 32, 2, 500.0, 2.0, 100.0, 0.85, 3.5),
    ]
    result = CalibrationResult("gpu_midrange", "debug_smoke", points, points[0])
    out_path = tmp_path / "cal.json"
    save_calibration_result(result, out_path)
    data = json.loads(out_path.read_text())
    assert data["hardware_profile"] == "gpu_midrange"
    assert len(data["points"]) == 1
    assert data["recommended"]["concurrent_games"] == 64


def test_print_calibration_summary(capsys):
    points = [
        CalibrationPoint(64, 32, 2, 500.0, 2.0, 100.0, 0.85, 3.5),
        CalibrationPoint(96, 64, 1, 600.0, 2.5, 120.0, 0.90, 3.0),
    ]
    result = CalibrationResult("gpu_midrange", "debug_smoke", points, points[1])
    print_calibration_summary(result)
    captured = capsys.readouterr()
    assert "Calibration Results" in captured.out
    assert "concurrent_games: 96" in captured.out
