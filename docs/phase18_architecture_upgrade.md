# Phase 18: Architecture Upgrade

## Overview

Phase 18 is a complete architecture upgrade to chessmoe, adding hardware-aware configuration, proper separation of concerns, production-grade infrastructure, real arena evaluation, replay buffer management, search diagnostics, and comprehensive testing.

## What Changed

### C++ Architecture (8 new files, 7 modified)

**New Modules:**

- `hardware_profiles.h/cpp` — 6 built-in hardware profiles with JSON extensibility
- `quality_profiles.h/cpp` — 5 quality profiles separating hardware from generation quality
- `hardware_probe.h/cpp` — Runtime detection of GPU, VRAM, CPU, CUDA, TensorRT, build type
- `config_resolver.h/cpp` — Unified CLI/config resolution with profile merging and guards
- `game_worker.h/cpp` — Per-game worker with opening diversity and profiling diagnostics
- `generation_controller.h/cpp` — Top-level orchestrator with resume, health checks, profile output
- `replay_health.h/cpp` — Post-generation replay quality analysis

**Modified:**

- `selfplay_main.cpp` — Uses GenerationController, adds `--probe`, `--profile-run`, `--calibrate`
- `selfplay_app.h/cpp` — Added `create_batch_evaluator_from_mode()` factory
- `gpu_selfplay_pipeline.h` — Added terminal count metrics, samples/sec, batch fill ratio
- `self_play_generator.h` — Added `opening_color_swapped`, `model_version_num`
- `CMakeLists.txt` — Added 6 new source files

### Python Architecture (11 new files, 2 modified)

**New Modules:**

- `models/registry.py` — Enhanced model registry with version lineage, promotion guards
- `pipeline/__init__.py` — Pipeline package
- `pipeline/config.py` — Python-side profile loading matching C++ profiles
- `pipeline/runner.py` — Pipeline stage orchestration
- `pipeline/report.py` — Markdown and HTML report generation
- `analysis/neural_arena.py` — Real ONNX/PyTorch-backed MCTS arena
- `analysis/calibration.py` — Matrix benchmark across concurrent_games × batch × flush_ms
- `analysis/diagnostics.py` — Policy entropy, top-k accuracy, value calibration
- `analysis/replay_buffer.py` — Rolling window, dedup, game-based split, maintenance

**New CLI:**

- `tools/chessmoe.py` — 16 commands: probe, calibrate, bootstrap, generate, train, export, build-engine, arena, arena-neural, promote, full-cycle, status, registry, report, diagnose, replay-buffer

### Configuration (3 new files)

- `configs/profiles/hardware.json` — Hardware profile definitions
- `configs/profiles/quality.json` — Quality profile definitions
- `configs/openings/diverse_openings.fen` — 15 opening positions for diversity

### Golden Test Positions

- `data/test_positions/golden.json` — 15 positions: mate-in-1, stalemate, promotion, tactical, endgame, legality, draw rules

### Tests (7 new test files, 61 new tests)

| File | Tests | Covers |
|------|-------|--------|
| `test_hardware_profiles.py` | 13 | Profile loading, JSON extensibility |
| `test_model_registry.py` | 13 | Registry CRUD, promotion guards |
| `test_pipeline_cli.py` | 3 | Pipeline configuration |
| `test_run_tracking.py` | 3 | Report generation |
| `test_neural_arena.py` | 4 | Neural arena backend, legal moves |
| `test_replay_buffer.py` | 4 | Dedup, split, rolling buffer |
| `test_golden_positions.py` | 6 | Golden positions, legal move validation |
| `test_calibration.py` | 4 | Calibration data structures |
| `test_dashboard.py` | 5 | HTML/Markdown report generation |

**Total: 12 C++ tests, 136 Python tests (61 new)**

## Feature Coverage

| # | Feature | Status |
|---|---------|--------|
| 1 | Three-layer refactor | Done |
| 2 | Neural self-play guards | Done |
| 3 | Hardware profiles | Done |
| 4 | Hardware probing | Done |
| 5 | Calibration matrix | Done |
| 6 | Quality profiles | Done |
| 7 | Generation metrics | Done |
| 8 | Replay health checks | Done |
| 9 | Crash resume | Done |
| 10 | Model registry | Done |
| 11 | Real arena | Done |
| 12 | Experiment tracking | Done |
| 13 | Pipeline full-cycle | Done |
| 14 | Replay buffer management | Done |
| 15 | Opening diversity | Done |
| 16 | Profiling mode | Done |
| 17 | Golden regression positions | Done |
| 18 | Search quality diagnostics | Done |
| 19 | Value/policy calibration | Done |
| 20 | Debug build enforcement | Done |
| 21 | UX/CLI | Done |
| 22 | HTML dashboard | Done |
| 23 | Fake progress guards | Done |
| 24 | Existing behavior preserved | Done |

## Usage

```powershell
# Hardware probe
selfplay.exe --probe

# Calibration
selfplay.exe --calibrate --hardware-profile cpu_bootstrap_debug --allow-debug
python tools/chessmoe.py calibrate --hardware-profile cpu_bootstrap_debug

# Material bootstrap (Debug build)
selfplay.exe --config configs\selfplay\bootstrap_material.json --allow-debug
python tools/chessmoe.py bootstrap

# Neural self-play (Release build required)
selfplay.exe --hardware-profile gpu_midrange --quality balanced_generation --phase 18 --engine weights\dense_bootstrap.engine

# Profiling
selfplay.exe --hardware-profile cpu_bootstrap_debug --quality debug_smoke --profile-run --allow-debug

# Pipeline
python tools/chessmoe.py generate --phase 18 --hardware-profile gpu_midrange --quality balanced_generation
python tools/chessmoe.py full-cycle --phase 18 --hardware-profile gpu_midrange --quality balanced_generation
python tools/chessmoe.py status
python tools/chessmoe.py registry

# Reports
python tools/chessmoe.py report --run-dir runs/2026-05-14_phase18 --format html --output report.html

# Diagnostics
python tools/chessmoe.py diagnose --replay-index data/replay/replay.sqlite --metric policy
python tools/chessmoe.py diagnose --replay-index data/replay/replay.sqlite --metric value

# Replay buffer
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action stats
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action dedup
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action maintain --max-chunks 5000

# Arena (seeded backend)
python tools/chessmoe.py arena --config configs/arena/tiny_arena.json

# Arena (neural evaluator)
python tools/chessmoe.py arena-neural --config configs/arena/tiny_arena.json --candidate weights/candidate.pt --best weights/best.pt --visits 64
```
