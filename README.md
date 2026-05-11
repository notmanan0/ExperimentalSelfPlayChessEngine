# chessmoe

`chessmoe` is a CUDA-first AlphaZero/Lc0-style self-play chess engine prototype. The current generation path is:

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
  -> arena gating
  -> promotion
```

CPU owns legal move generation, board state, MCTS tree bookkeeping, game orchestration, and replay writing. GPU is used for Python training and batched neural inference only.

## Status

- Legal chess core, UCI shell, evaluator boundary, CPU MCTS, replay chunks, training, export, and self-play orchestration are in place.
- `selfplay` is the production self-play executable.
- `gpu_selfplay_benchmark` is retained only for compatibility.
- Material bootstrap is weak but useful for creating initial replay.
- TensorRT support is optional at build time. Selecting `--evaluator tensorrt` without support or without `--engine` fails clearly.
- C++ ONNX Runtime inference is not implemented yet. Selecting `--evaluator onnx` reports unavailable.
- The current arena path is placeholder-seeded and must not be used for meaningful promotion decisions.
- Dense transformer is the first serious default model. MoE remains a later experimental path.

## Requirements

- Windows with CMake and an MSVC/NMake environment.
- Python 3.11+.
- PyTorch for training and export.
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
ctest --test-dir build-nmake --output-on-failure
python -m pytest tests/python --basetemp python-test-output/pytest-current
```

Expected baseline at this point:

- C++ tests pass.
- Python tests pass.

## Material Bootstrap Self-Play

```powershell
.\build-nmake\bin\selfplay.exe `
  --config configs\selfplay\bootstrap_material.json
```

Equivalent explicit command:

```powershell
.\build-nmake\bin\selfplay.exe `
  --evaluator material `
  --games 512 `
  --concurrent-games 64 `
  --fixed-batch 64 `
  --visits 64 `
  --max-plies 128 `
  --write-replay `
  --output-dir data\replay\phase13 `
  --model-version 0 `
  --progress-interval 10
```

Very short games or low samples per game usually indicate weak or broken self-play settings.

## Replay Validation And Indexing

```powershell
python tools/convert/validate_replay.py data/replay/phase13
python tools/convert/index_replay_dir.py data/replay/phase13 --index data/replay/replay.sqlite
python tools/convert/summarize_replay.py data/replay/phase13
```

Replay tooling prints human-readable progress and summaries. Training metrics remain machine-readable JSONL.

## Train Bootstrap Model

```powershell
python -m chessmoe.training.train --config configs/training/dense_bootstrap.json
```

## Export ONNX

```powershell
python python/export/export_tiny_onnx.py `
  --checkpoint weights/dense_bootstrap.pt `
  --output weights/dense_bootstrap.onnx `
  --model-kind dense_transformer `
  --verify
```

## Build TensorRT Engine

```powershell
python python/export/build_tensorrt_engine.py `
  --onnx weights/dense_bootstrap.onnx `
  --engine weights/dense_bootstrap.engine `
  --max-shape 64,18,8,8
```

## Neural Self-Play

```powershell
.\build-nmake\bin\selfplay.exe `
  --config configs\selfplay\neural_tensorrt.json `
  --engine weights\dense_bootstrap.engine
```

GPU self-play only happens with a neural evaluator such as TensorRT. Material mode is CPU evaluation.

## Pipeline Shortcuts

```powershell
python tools/run_pipeline.py bootstrap
python tools/run_pipeline.py generation --phase 14 --engine weights/dense_bootstrap.engine
python tools/run_pipeline.py arena --candidate weights/phase14.engine --baseline weights/phase13.engine --games 200 --visits 128
python tools/run_pipeline.py promote --candidate weights/candidate.engine --version 2
```

The arena shortcut currently refuses meaningful promotion because the arena backend is still placeholder-seeded.

## Promotion Rules

Promotion preserves history before replacing the current best:

```text
weights/best.pt
weights/best.onnx
weights/best.engine
weights/history/model_000001.*
```

Never overwrite the current best without preserving a versioned history artifact.

## Documentation

- [[docs/generation_pipeline.md]]
- [[docs/replay_chunk_format.md]]
- [[docs/phase10_cuda_tensorrt_inference.md]]
- [[docs/phase13_gpu_selfplay_pipeline.md]]
- [[docs/phase15_moe_transformer.md]]
