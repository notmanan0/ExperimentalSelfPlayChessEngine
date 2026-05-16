# User-Facing Generation Pipeline

This workflow keeps CPU and GPU responsibilities separate. The CPU owns legal move generation, board state, MCTS bookkeeping, game orchestration, and `.cmrep` replay writing. GPU work is limited to Python training and batched neural inference through an explicit backend such as TensorRT.

Material bootstrap is weak but useful: it creates initial replay without needing a neural model. Very short games or low samples per game usually indicate weak or broken self-play settings and should be investigated before training the next generation.

`GenerationController` plus `GameWorker` is the production self-play path. `GpuSelfPlayPipeline` and `SelfPlayGenerator` are retained only for focused legacy tests and benchmarks; new orchestration should not add logic there.

## Windows Build

```powershell
cmake -S . -B build-nmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build-nmake --config Debug
ctest --test-dir build-nmake --output-on-failure
python -m pytest tests/python --basetemp python-test-output/pytest-current
```

## Material Bootstrap Replay

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

## Validate, Index, And Summarize Replay

```powershell
python tools/convert/validate_replay.py data/replay/phase13
python tools/convert/index_replay_dir.py data/replay/phase13 --index data/replay/replay.sqlite
python tools/convert/summarize_replay.py data/replay/phase13
```

Replay indexing prints files scanned, indexed, failed, samples, updates, elapsed time, throughput, and the current file. Replay summary reports chunk count, sample totals, min/max/mean samples per chunk, model and generator versions, total size, shortest and longest files, and short-game warnings.

## Train Dense Bootstrap Model

```powershell
python -m chessmoe.training.train --config configs/training/dense_bootstrap.json
```

Console output is human-readable progress. Metrics remain machine-readable JSONL.

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

TensorRT support is optional at C++ build time. Selecting `--evaluator tensorrt` without compiled support fails clearly. Selecting TensorRT also requires `--engine`; it never silently falls back to material evaluation.

## Neural Self-Play

```powershell
.\build-nmake\bin\selfplay.exe `
  --config configs\selfplay\neural_tensorrt.json `
  --engine weights\dense_bootstrap.engine
```

GPU self-play only happens with a neural evaluator such as TensorRT. Material self-play is CPU evaluation and is for bootstrap only.

## Arena And Promotion

`arena-neural` runs real model-vs-model games through a Python PUCT search using legal move masks, neural policy priors, and side-to-move value backup. Promotion decisions may use this path when the arena configuration has enough games for the configured gate.

The seeded `arena` backend remains a deterministic test harness. Do not treat seeded results as playing-strength evidence for promotion.

Promotion preserves history before replacing the current best:

```powershell
python tools/run_pipeline.py promote --candidate weights/candidate.engine --version 2
```

Artifacts follow:

```text
weights/best.pt
weights/best.onnx
weights/best.engine
weights/history/model_000001.*
```

MoE remains available for later experiments, but it is not the default generation path yet. Dense transformer is the first serious default model.

## Pipeline Shortcuts

```powershell
python tools/run_pipeline.py bootstrap
python tools/run_pipeline.py generation --phase 14 --engine weights/dense_bootstrap.engine
python tools/run_pipeline.py arena --candidate weights/phase14.engine --baseline weights/phase13.engine --games 200 --visits 128
python tools/run_pipeline.py promote --candidate weights/candidate.engine --version 2
```
