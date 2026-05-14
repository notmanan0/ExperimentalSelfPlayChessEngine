from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import json
import subprocess
import sys
import time
from typing import Any


@dataclass
class CalibrationPoint:
    concurrent_games: int
    fixed_batch: int
    flush_ms: int
    positions_per_sec: float
    games_per_sec: float
    samples_per_sec: float
    batch_fill_ratio: float
    avg_inference_latency_ms: float


@dataclass
class CalibrationResult:
    hardware_profile: str
    quality_profile: str
    points: list[CalibrationPoint]
    recommended: CalibrationPoint | None


def run_calibration_matrix(
    selfplay_exe: Path,
    hardware_profile: str,
    quality_profile: str = "debug_smoke",
    *,
    concurrent_games_list: list[int] | None = None,
    fixed_batch_list: list[int] | None = None,
    flush_ms_list: list[int] | None = None,
    games_per_point: int = 4,
    visits_override: int = 4,
    max_plies_override: int = 20,
) -> CalibrationResult:
    if concurrent_games_list is None:
        concurrent_games_list = [16, 32, 64, 96]
    if fixed_batch_list is None:
        fixed_batch_list = [16, 32, 64]
    if flush_ms_list is None:
        flush_ms_list = [1, 2, 5]

    points: list[CalibrationPoint] = []
    best: CalibrationPoint | None = None

    combos = list(itertools.product(
        concurrent_games_list, fixed_batch_list, flush_ms_list
    ))
    total = len(combos)

    for idx, (cg, fb, fm) in enumerate(combos):
        print(f"[calibrate] {idx + 1}/{total}: "
              f"concurrent={cg} batch={fb} flush_ms={fm}")
        started = time.monotonic()

        command = [
            str(selfplay_exe),
            "--hardware-profile", hardware_profile,
            "--quality", quality_profile,
            "--games", str(games_per_point),
            "--concurrent-games", str(cg),
            "--fixed-batch", str(fb),
            "--flush-ms", str(fm),
            "--visits", str(visits_override),
            "--max-plies", str(max_plies_override),
            "--allow-debug",
            "--output-dir", f"data/calibration/{cg}_{fb}_{fm}",
        ]

        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=120, check=False,
            )
            elapsed = time.monotonic() - started
            metrics = _parse_metrics(result.stdout)

            point = CalibrationPoint(
                concurrent_games=cg,
                fixed_batch=fb,
                flush_ms=fm,
                positions_per_sec=metrics.get("positions_per_sec", 0.0),
                games_per_sec=metrics.get("games_per_sec", 0.0),
                samples_per_sec=metrics.get("samples_per_sec", 0.0),
                batch_fill_ratio=metrics.get("batch_fill", 0.0),
                avg_inference_latency_ms=metrics.get(
                    "avg_inference_latency_ms", 0.0
                ),
            )
            points.append(point)

            if best is None or point.positions_per_sec > best.positions_per_sec:
                best = point

            print(f"  -> positions/sec={point.positions_per_sec:.1f} "
                  f"batch_fill={point.batch_fill_ratio:.3f}")

        except subprocess.TimeoutExpired:
            print(f"  -> timeout after 120s")
        except Exception as e:
            print(f"  -> error: {e}")

    return CalibrationResult(
        hardware_profile=hardware_profile,
        quality_profile=quality_profile,
        points=points,
        recommended=best,
    )


def _parse_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if "selfplay summary:" in line:
            for part in line.split():
                if "=" in part:
                    key, _, value = part.partition("=")
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        pass
    return metrics


def save_calibration_result(result: CalibrationResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "hardware_profile": result.hardware_profile,
        "quality_profile": result.quality_profile,
        "points": [
            {
                "concurrent_games": p.concurrent_games,
                "fixed_batch": p.fixed_batch,
                "flush_ms": p.flush_ms,
                "positions_per_sec": p.positions_per_sec,
                "games_per_sec": p.games_per_sec,
                "batch_fill_ratio": p.batch_fill_ratio,
            }
            for p in result.points
        ],
        "recommended": None
        if result.recommended is None
        else {
            "concurrent_games": result.recommended.concurrent_games,
            "fixed_batch": result.recommended.fixed_batch,
            "flush_ms": result.recommended.flush_ms,
            "positions_per_sec": result.recommended.positions_per_sec,
        },
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def print_calibration_summary(result: CalibrationResult) -> None:
    print("\n=== Calibration Results ===")
    print(f"Profile: {result.hardware_profile} / {result.quality_profile}")
    print(f"Points tested: {len(result.points)}")
    if result.recommended:
        r = result.recommended
        print(f"\nRecommended settings:")
        print(f"  concurrent_games: {r.concurrent_games}")
        print(f"  fixed_batch: {r.fixed_batch}")
        print(f"  flush_ms: {r.flush_ms}")
        print(f"  positions/sec: {r.positions_per_sec:.1f}")
        print(f"  batch_fill: {r.batch_fill_ratio:.3f}")
    if result.points:
        print(f"\nTop 5 by positions/sec:")
        sorted_points = sorted(
            result.points, key=lambda p: p.positions_per_sec, reverse=True
        )
        for p in sorted_points[:5]:
            print(f"  cg={p.concurrent_games:3d} fb={p.fixed_batch:3d} "
                  f"fm={p.flush_ms} -> {p.positions_per_sec:.1f} pos/s "
                  f"fill={p.batch_fill_ratio:.3f}")
