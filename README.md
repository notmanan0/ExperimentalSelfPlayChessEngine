# chessmoe

`chessmoe` is a CUDA-first AlphaZero/Lc0-style self-play chess engine. The production pipeline is:

```text
PyTorch checkpoint
  -> ONNX export
  -> TensorRT engine
  -> C++ neural evaluator
  -> batched CPU MCTS self-play
  -> .cmrep replay
  -> validation/indexing
  -> Python training
  -> export
  -> TensorRT build
  -> real arena gating
  -> promotion/model registry
```

CPU owns legal move generation, board state, MCTS tree bookkeeping, game orchestration, and replay writing. GPU owns batched neural inference and Python training.

## Status

- Legal chess core, UCI shell, evaluator boundary, CPU MCTS, replay chunks, training, export, and self-play orchestration are in place.
- `selfplay` is the production self-play executable.
- Hardware profiles, quality profiles, and hardware probing are implemented.
- Config-driven profile resolution with CLI overrides.
- Debug build detection and refusal for serious generation.
- Material evaluator is bootstrap/debug only with explicit warnings.
- PeSTO tapered handcrafted evaluator for stronger bootstrap data generation.
- Bitboard utility layer with popcount, lsb/msb, shift helpers, and attack tables.
- TensorRT support is optional at build time. Selecting `--evaluator tensorrt` without support or without `--engine` fails clearly.
- C++ ONNX Runtime inference is not implemented. Selecting `--evaluator onnx` fails clearly.
- Replay health checks run automatically after generation.
- Model registry tracks version lineage and arena results.
- `tools/chessmoe.py` provides the user-facing CLI.
- Dense transformer is the first serious default model.

## Requirements

- Windows with CMake and an MSVC/NMake environment.
- Python 3.11+.
- PyTorch for training and export.
- python-chess for arena evaluation (`pip install python-chess`).
- Optional: CUDA and TensorRT for neural C++ inference.

Before building from a normal PowerShell session, load the Visual C++ environment:

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"'
```

Or run build commands from a Developer PowerShell.

## Build And Test

```powershell
cmake -S . -B build-nmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build-nmake --config Debug
ctest --test-dir build-nmake -C Debug --output-on-failure
python -m pytest tests/python --basetemp python-test-output/pytest-current
```

Expected: 15 C++ tests and 193 Python tests pass.

## Bootstrap Strategy

The engine supports a classical bootstrap path so neural self-play starts from useful chess knowledge:

1. **Material evaluator** (`--evaluator material`): Uniform policy, material-only value. Debug/smoke only.
2. **PeSTO evaluator** (`--evaluator pesto`): Tapered piece-square table evaluation with static move ordering for policy priors. Usable for bootstrap data generation.
3. **Neural evaluator** (`--evaluator tensorrt`): Full neural inference via TensorRT.

### PeSTO Bootstrap

Generate stronger initial training data using the handcrafted PeSTO evaluator:

```powershell
.\build-nmake\bin\Debug\selfplay.exe --evaluator pesto --hardware-profile cpu_pesto_bootstrap --games 256 --allow-debug
python tools/chessmoe.py bootstrap
```

Or use the config:

```powershell
.\build-nmake\bin\Debug\selfplay.exe --config configs/selfplay/bootstrap_pesto.json --allow-debug
```

### Teacher Target Generation

Generate soft policy/value targets using alpha-beta search with PeSTO evaluation:

```powershell
python tools/teacher/generate_teacher_targets.py \
    --fen-file data/openings/bootstrap_fens.txt \
    --output data/teacher/pesto_ab_targets.jsonl \
    --depth 4
```

Optional Stockfish distillation (requires Stockfish binary):

```powershell
python tools/teacher/label_with_stockfish.py \
    --fen-file data/openings/bootstrap_fens.txt \
    --stockfish-path C:\path\to\stockfish.exe \
    --output data/teacher/sf_targets.jsonl \
    --depth 12
```

### Teacher Bootstrap Training

Train using teacher targets:

```powershell
python tools/chessmoe.py train --config configs/training/dense_teacher_bootstrap.json
```

### Bitboard Utilities

The `chessmoe::chess` namespace includes bitboard utilities in `bitboard.h`:

- `popcount`, `lsb_index`, `msb_index`, `lsb_square`, `msb_square`
- `pop_lsb` for iterating over set bits
- `file_mask`, `rank_mask` for board masks
- `shift_n/s/e/w/ne/nw/se/sw` for piece movement
- `knight_attacks`, `king_attacks` attack tables
- `BitboardIterator` for efficient bit iteration

### Alpha-Beta Teacher Search

The `chessmoe::search::AlphaBetaSearcher` provides:

- Fail-soft negamax with alpha-beta pruning
- Iterative deepening
- Transposition table using Zobrist hash
- Move ordering: TT move, captures, killer moves, history heuristic
- Quiescence search for captures/promotions
- Principal variation tracking

## Full-Cycle Pipeline

The full training cycle includes:

1. Self-play generation (material/pesto/neural)
2. Replay indexing and validation
3. Training (from replay or teacher targets)
4. ONNX export
5. TensorRT engine build (optional)
6. Neural arena evaluation
7. Promotion or rejection

### Full-Cycle Command

```powershell
python tools/chessmoe.py full-cycle \
    --phase 1 \
    --hardware-profile gpu_midrange \
    --quality balanced_generation \
    --train-config configs/training/dense_bootstrap.json \
    --checkpoint weights/dense_bootstrap.pt \
    --onnx-output weights/dense_bootstrap.onnx \
    --engine-output weights/dense_bootstrap.engine \
    --candidate weights/candidate.pt \
    --best weights/best.pt
```

Flags:
- `--skip-engine-build`: Skip TensorRT engine build
- `--skip-promotion`: Skip automatic promotion
- `--allow-debug`: Allow Debug build for generation

### Promotion

Promotion requires arena results by default:

```powershell
python tools/chessmoe.py promote --candidate weights/candidate.pt --version 2
```

Use `--force` to promote without arena evidence.

## Quick Start

### Hardware Probe

```powershell
.\build-nmake\bin\Debug\selfplay.exe --probe
```

Detects GPU, CPU, VRAM, build type, and recommends a profile.

### Calibration

```powershell
.\build-nmake\bin\Debug\selfplay.exe --calibrate --hardware-profile cpu_bootstrap_debug --allow-debug
python tools/chessmoe.py calibrate --hardware-profile cpu_bootstrap_debug
```

Tests a matrix of settings and recommends optimal configuration.

### Material Bootstrap

```powershell
.\build-nmake\bin\Debug\selfplay.exe --config configs\selfplay\bootstrap_material.json --allow-debug
python tools/chessmoe.py bootstrap
```

Debug builds require `--allow-debug`. Material evaluator prints a bootstrap-only warning.

### PeSTO Bootstrap

```powershell
.\build-nmake\bin\Debug\selfplay.exe --config configs\selfplay\bootstrap_pesto.json --allow-debug
python tools/chessmoe.py bootstrap
```

PeSTO evaluator uses tapered piece-square tables for stronger initial training data.

### Teacher Target Bootstrap

```powershell
python tools/teacher/generate_teacher_targets.py --fen-file data/openings/bootstrap_fens.txt --output data/teacher/pesto_ab_targets.jsonl --depth 4
python tools/chessmoe.py train --config configs/training/dense_teacher_bootstrap.json
```

### Neural Self-Play

```powershell
.\build-nmake\bin\Release\selfplay.exe --hardware-profile gpu_midrange --quality balanced_generation --phase 18 --engine weights\dense_bootstrap.engine
```

Serious generation requires Release build and a TensorRT engine.

### Python CLI

```powershell
python tools/chessmoe.py probe
python tools/chessmoe.py calibrate --hardware-profile cpu_bootstrap_debug
python tools/chessmoe.py bootstrap
python tools/chessmoe.py generate --phase 18 --hardware-profile gpu_midrange --quality balanced_generation
python tools/chessmoe.py train --config configs/training/dense_bootstrap.json
python tools/chessmoe.py promote --candidate weights/candidate.pt --version 2
python tools/chessmoe.py full-cycle --phase 18 --hardware-profile gpu_midrange --quality balanced_generation
python tools/chessmoe.py status
python tools/chessmoe.py registry
python tools/chessmoe.py report --run-dir runs/latest --format html
python tools/chessmoe.py diagnose --replay-index data/replay/replay.sqlite --metric policy
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action stats
```

## Hardware Profiles

Config-driven profiles define hardware-specific settings:

| Profile | Evaluator | Concurrent | Batch | Visits | Description |
|---------|-----------|------------|-------|--------|-------------|
| `cpu_bootstrap_debug` | material | 4 | 1 | 8 | Minimal debug |
| `cpu_bootstrap_fast` | material | 16 | 1 | 32 | Fast bootstrap |
| `cpu_pesto_bootstrap` | pesto | 16 | 1 | 32 | Strong PeSTO bootstrap |
| `gpu_low_vram` | tensorrt | 32 | 32 | 64 | 4 GB VRAM |
| `gpu_midrange` | tensorrt | 96 | 64 | 128 | 6-10 GB VRAM |
| `gpu_highend` | tensorrt | 128 | 128 | 200 | 12-24 GB VRAM |
| `gpu_datacenter` | tensorrt | 256 | 256 | 400 | 40+ GB VRAM |

## Quality Profiles

Separate from hardware — controls generation quality:

| Profile | Visits | Max Plies | Games | Description |
|---------|--------|-----------|-------|-------------|
| `fast_bootstrap` | 32 | 128 | 256 | Initial training data |
| `balanced_generation` | 128 | 200 | 2048 | Steady improvement |
| `high_quality_generation` | 400 | 300 | 4096 | Strong models |
| `arena_eval` | 200 | 300 | 0 | Deterministic arena |
| `debug_smoke` | 4 | 20 | 4 | Pipeline validation |

## Replay Health Checks

After generation, automatically checks:

- Total samples count
- Average plies per game
- Draw rate
- File sizes
- Terminal reason distribution

Warnings and errors are printed and written to `runs/<run_id>/replay_health.json`.

## Model Registry

Tracks model lineage at `weights/registry.json`:

```json
{
  "best": 2,
  "entries": [
    {"model_version": 1, "promotion_status": "promoted", ...},
    {"model_version": 2, "promotion_status": "candidate", ...}
  ]
}
```

Promotion requires valid arena results unless `--force` is used.

## Arena

### Seeded Arena (fast, deterministic)

```powershell
python tools/chessmoe.py arena --config configs/arena/tiny_arena.json
```

### Neural Arena (real evaluation)

```powershell
python tools/chessmoe.py arena-neural --config configs/arena/tiny_arena.json --candidate weights/candidate.pt --best weights/best.pt --visits 64
```

Uses python-chess for legal move generation and PyTorch/ONNX models for position evaluation with MCTS search.

## Replay Buffer Management

```powershell
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action stats
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action dedup
python tools/chessmoe.py replay-buffer --replay-index data/replay/replay.sqlite --action maintain --max-chunks 5000
```

## Diagnostics

```powershell
python tools/chessmoe.py diagnose --replay-index data/replay/replay.sqlite --metric policy
python tools/chessmoe.py diagnose --replay-index data/replay/replay.sqlite --metric value
```

## Dashboard

```powershell
python tools/chessmoe.py report --run-dir runs/latest --format html --output report.html
```

## Replay Validation And Indexing

```powershell
python tools/convert/validate_replay.py data/replay/phase18
python tools/convert/index_replay_dir.py data/replay/phase18 --index data/replay/replay.sqlite
python tools/convert/summarize_replay.py data/replay/phase18
```

## Promotion Rules

Promotion preserves history before replacing the current best:

```text
weights/best.pt
weights/history/model_000001.pt
weights/history/model_000002.pt
```

Never overwrite the current best without preserving a versioned history artifact.

## CLI Flags

| Flag | Description |
|------|-------------|
| `--probe` | Detect hardware and recommend profile |
| `--hardware-profile <name>` | Select hardware profile |
| `--quality <name>` | Select quality profile |
| `--allow-debug` | Allow Debug build for serious generation |
| `--resume` | Resume interrupted generation |
| `--fresh` | Overwrite existing replay |
| `--calibrate` | Run calibration benchmark |
| `--profile-run` | Output profiling breakdown JSON |
| `--engine <path>` | TensorRT engine path |
| `--phase <n>` | Phase number for run directory |

## Tools

| Command | Description |
|---------|-------------|
| `chessmoe.py probe` | List hardware profiles |
| `chessmoe.py calibrate` | Run calibration matrix |
| `chessmoe.py bootstrap` | Material bootstrap pipeline |
| `chessmoe.py generate` | Neural self-play generation |
| `chessmoe.py train` | Train from replay data |
| `chessmoe.py export` | Export model to ONNX |
| `chessmoe.py build-engine` | Build TensorRT engine |
| `chessmoe.py arena` | Seeded arena evaluation |
| `chessmoe.py arena-neural` | Neural MCTS arena |
| `chessmoe.py promote` | Promote candidate model |
| `chessmoe.py full-cycle` | Full training cycle |
| `chessmoe.py status` | Show run status |
| `chessmoe.py registry` | Show model registry |
| `chessmoe.py report` | Generate HTML/Markdown report |
| `chessmoe.py diagnose` | Policy/value diagnostics |
| `chessmoe.py replay-buffer` | Replay buffer management |

## Documentation

- [[docs/generation_pipeline.md]]
- [[docs/replay_chunk_format.md]]
- [[docs/phase10_cuda_tensorrt_inference.md]]
- [[docs/phase13_gpu_selfplay_pipeline.md]]
- [[docs/phase18_architecture_upgrade.md]]

## CI

GitHub Actions CI runs on Windows and Linux:

- C++ build and tests (all 15 tests)
- Python tests (all 193 tests)
- CPU-only, no CUDA/TensorRT required

TensorRT-dependent tests are skipped unless explicitly enabled.
