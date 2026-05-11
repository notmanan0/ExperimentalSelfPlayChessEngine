# Phase 13: GPU-Accelerated Self-Play Pipeline

Checked against current official documentation on 2026-05-10.

Sources:

- NVIDIA CUDA Programming Guide v13.2 is the current official CUDA programming reference and includes asynchronous execution and CUDA Graphs guidance: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- NVIDIA TensorRT 10.16.1 documentation describes TensorRT as a GPU inference SDK with mixed precision, dynamic shapes, transformer optimizations, and performance tooling such as `trtexec`: https://docs.nvidia.com/deeplearning/tensorrt/latest/
- NVIDIA TensorRT dynamic-shape guidance requires build-time optimization profiles and runtime profile/input-shape selection for dynamic dimensions, so Phase 13 keeps self-play batches fixed where possible: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html
- NVIDIA Nsight Systems 2026.2 recommends focused profiling, `cudaProfilerStart()` / `cudaProfilerStop()`, and NVTX ranges to make CPU/GPU timelines understandable: https://docs.nvidia.com/nsight-systems/UserGuide/
- NVIDIA Nsight Compute 2026.1.1 is NVIDIA's CUDA kernel profiler with UI and CLI collection, detailed metrics, API debugging, baselines, and report comparison: https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- PyTorch CUDA notes describe CUDA Graphs as replaying the same kernels with the same arguments and memory addresses, reducing CPU launch overhead while giving up dynamic flexibility: https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- PyTorch `torch.cuda` documents CUDA streams, events, NVTX helpers, and CUDA Graph APIs as beta: https://docs.pytorch.org/docs/stable/cuda.html

## Objective

Scale training self-play throughput by running many CPU-side games concurrently while routing neural-network leaf evaluations through one asynchronous fixed-batch inference worker. Per-game MCTS remains CPU-side. The GPU is used only through the existing backend-agnostic batch evaluator path, which can be backed by TensorRT for production self-play.

This phase does not move MCTS traversal, node storage, tree backup, move generation, replay reanalysis, or legal chess logic to CUDA.

## Files To Create

- `cpp/inference/include/chessmoe/inference/async_batching_evaluator.h`
- `cpp/inference/src/async_batching_evaluator.cpp`
- `cpp/selfplay/include/chessmoe/selfplay/gpu_selfplay_pipeline.h`
- `cpp/selfplay/src/gpu_selfplay_pipeline.cpp`
- `cpp/selfplay/include/chessmoe/selfplay/selfplay_app.h`
- `cpp/selfplay/src/selfplay_app.cpp`
- `cpp/selfplay/src/selfplay_main.cpp`
- `tools/benchmark/CMakeLists.txt`
- `tools/benchmark/gpu_selfplay_benchmark.cpp`
- `tools/convert/index_replay_dir.py`
- `tools/convert/summarize_replay.py`
- `tools/clean.py`
- `tools/run_pipeline.py`
- `configs/selfplay/bootstrap_material.json`
- `configs/selfplay/neural_tensorrt.json`
- `configs/selfplay/gpu_selfplay_phase13.yaml`
- `tests/cpp/selfplay/test_gpu_selfplay_pipeline.cpp`
- `tests/cpp/selfplay/test_selfplay_app.cpp`
- `tests/python/test_pipeline_tools.py`
- `docs/phase13_gpu_selfplay_pipeline.md`
- `docs/generation_pipeline.md`

Files updated:

- `CMakeLists.txt`
- `cpp/inference/CMakeLists.txt`
- `cpp/selfplay/CMakeLists.txt`
- `tests/cpp/selfplay/CMakeLists.txt`

## Core Data Structures

- `inference::AsyncBatchingEvaluatorConfig`
  - `fixed_batch_size`: preferred self-play batch size, for example 64.
  - `max_pending_requests`: bounded queue capacity for backpressure.
  - `flush_timeout`: maximum wait before a partial batch is flushed.
  - `pad_to_fixed_batch`: pads partial flushes by duplicating the last request and discarding padded outputs.

- `inference::AsyncBatchingEvaluator`
  - Implements `eval::ISinglePositionEvaluator`.
  - CPU MCTS threads call `evaluate()` synchronously.
  - Internally, each call enqueues an `EvaluationRequest`, blocks on a `std::future`, and receives the matching result after the worker batch completes.

- `inference::AsyncBatchingMetrics`
  - Tracks positions/sec inputs: real positions evaluated, batches evaluated, padded positions, max queue depth, batch size histogram, valid batch size histogram, and inference latency.

- `selfplay::GpuSelfPlayPipelineConfig`
  - Controls game count, concurrent games, fixed batch size, queue capacity, flush timeout, replay output, and base `SelfPlayConfig`.

- `selfplay::GpuSelfPlayPipeline`
  - Starts concurrent CPU game workers.
  - Shares one `AsyncBatchingEvaluator`.
  - Writes replay chunks per completed game when enabled.

- `selfplay::GpuSelfPlayMetrics`
  - Reports samples written, positions/sec, games/hour, GPU utilisation placeholder/sample, batch size distribution, average inference latency, completed games, evaluated positions, padded positions, and max queue depth.

- `selfplay::SelfPlayAppOptions`
  - User-facing options for evaluator mode, TensorRT engine path, pipeline config, and progress interval.

- `selfplay::EvaluatorMode`
  - Runtime evaluator selection: `material`, `tensorrt`, or `onnx`.
  - `material` uses the bootstrap evaluator.
  - `tensorrt` requires an engine path and never silently falls back.
  - `onnx` reports unavailable until a real C++ ONNX Runtime backend exists.

## Main Algorithms

### Threading Model

```text
main thread
  creates one GpuSelfPlayPipeline
  creates one AsyncBatchingEvaluator
  starts N CPU game worker threads

game worker thread
  owns one SelfPlayGenerator
  owns one CPU-side MCTS instance per move
  calls AsyncBatchingEvaluator::evaluate() at leaf/root evaluation points
  blocks until the shared inference worker returns that position's result
  writes replay chunk after game completion when replay writing is enabled

inference worker thread
  owns queue drain and backend batch calls
  waits for fixed_batch_size requests or flush_timeout
  pads partial batches to fixed_batch_size for self-play
  calls IBatchEvaluator::evaluate_batch()
  fulfills per-request promises for real requests only
```

### Queue Pseudocode

```text
evaluate(request):
  pending = {request, promise}
  lock queue
  wait until queue.size < max_pending_requests
  push pending
  update max_queue_depth
  notify inference worker
  unlock queue
  return pending.future.get()

worker_loop():
  while not stopped or queue not empty:
    lock queue
    wait until queue not empty or stopped
    deadline = now + flush_timeout
    while queue.size < fixed_batch_size and not stopped:
      wait until notify or deadline
      if deadline reached: break
    pop up to fixed_batch_size requests
    notify blocked producers
    unlock queue

    requests = real popped requests
    while pad_to_fixed_batch and requests.size < fixed_batch_size:
      append copy of requests.back
    started = now
    results = backend.evaluate_batch(requests)
    latency = now - started
    record metrics using real count and padded count
    set promises only for real requests
```

### Self-Play Orchestration

```text
run(config):
  create async evaluator around TensorRT-backed IBatchEvaluator
  next_game = atomic counter
  start concurrent_games workers
  each worker:
    while game_id = next_game++ is in range:
      game_config.seed = base_seed + game_id
      generate game through existing SelfPlayGenerator
      write replay chunk with game_id when enabled
      store completed game
  join workers
  merge inference metrics into throughput metrics
```

### Profiling Hooks

- Use `GpuSelfPlayMetrics` for baseline throughput before CUDA Graph experiments.
- Use Nsight Systems around `gpu_selfplay_benchmark` for CPU/GPU overlap, queue waiting, inference launch timing, and stream activity.
- Use Nsight Compute only after Nsight Systems identifies a specific slow kernel or TensorRT layer/kernel region.
- Use NVTX ranges in the TensorRT backend path before deeper CUDA optimization.
- Keep CUDA Graphs disabled until fixed-shape baseline throughput and latency are recorded. If enabled later, capture only the fixed-shape self-play inference path, because CUDA Graph replay assumes stable work, arguments, and memory addresses.

## Tests

- `gpu_selfplay_pipeline_tests`
  - Async evaluator flushes a partial self-play request group as one fixed padded batch.
  - Metrics count real positions, padded positions, and batch-size histogram entries.
  - Pipeline completes multiple concurrent games.
  - Deterministic small run gives the same selected move sequence for the same seed and evaluator.
  - Replay writer integration produces a valid `CMREPLAY` chunk.

## Benchmark Command

Configure and build:

```powershell
cmake -S . -B build-nmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build-nmake --config Debug --target gpu_selfplay_benchmark
```

Run a small orchestration benchmark:

```powershell
build-nmake\tools\benchmark\gpu_selfplay_benchmark.exe --games 32 --concurrent-games 16 --fixed-batch 64 --visits 32 --max-plies 64 --write-replay --output-dir data/replay/phase13
```

Profile the same run with Nsight Systems:

```powershell
nsys profile --trace=cuda,nvtx,osrt --output phase13_selfplay build-nmake\tools\benchmark\gpu_selfplay_benchmark.exe --games 128 --concurrent-games 64 --fixed-batch 64 --visits 64 --max-plies 128
```

Only use Nsight Compute after a specific kernel is identified:

```powershell
ncu --set full --target-processes all build-nmake\tools\benchmark\gpu_selfplay_benchmark.exe --games 16 --concurrent-games 16 --fixed-batch 64 --visits 32 --max-plies 64
```

The local benchmark target uses the available `IBatchEvaluator` path. In production, run the same pipeline with the TensorRT-backed batch evaluator so the reported inference latency and GPU utilisation reflect real GPU inference.

## Production Self-Play Command

`selfplay` is the production self-play executable. It emits a start banner, periodic progress, clear failures, and a final summary.

Bootstrap material replay:

```powershell
.\build-nmake\bin\selfplay.exe `
  --config configs\selfplay\bootstrap_material.json
```

TensorRT neural replay:

```powershell
.\build-nmake\bin\selfplay.exe `
  --config configs\selfplay\neural_tensorrt.json `
  --engine weights\dense_bootstrap.engine
```

Progress includes completed games, total games, percentage, samples written, games/sec, samples/sec, elapsed time, ETA, average plies/game, active games, output directory, evaluator mode, model version, batches evaluated, padded positions, and average inference latency.

## Completion Criteria

- Multiple concurrent self-play games run through a shared evaluator boundary.
- A production `selfplay` executable exists under `build-nmake\bin`.
- Runtime evaluator mode selection is available without C++ edits after build.
- Inference requests are queued and bounded.
- Backpressure blocks producers when `max_pending_requests` is reached.
- Self-play inference batches are fixed-size and padded on timeout flush.
- Per-game MCTS state remains CPU-side.
- The batch evaluator contract remains backend-agnostic and TensorRT-compatible.
- Replay chunks are written after completed games.
- Metrics include samples written, positions/sec, games/hour, GPU utilisation sample field, batch-size distribution, and inference latency.
- Replay indexing and summary tools provide user-readable progress and final summaries.
- Pipeline and promotion tooling preserve best-model history and stop on failure.
- Profiling hooks and commands are documented.
- CUDA Graphs remain an optional fixed-shape experiment after baseline measurement.
- No reanalysis implementation is added.

## Common Failure Modes

- Moving MCTS tree state to GPU before profiling proves CPU traversal is the bottleneck.
- Letting padded inference requests leak into replay targets or game state.
- Unbounded request queues causing memory growth when game workers outpace inference.
- Using dynamic tail batches for self-play and then expecting CUDA Graph capture to be stable.
- Enabling CUDA Graphs before tensor addresses, stream usage, and batch shape are fixed.
- Measuring only average latency and missing batch-size distribution or queue depth.
- Treating GPU utilisation as a substitute for positions/sec and replay validity.
- Writing replay from games that failed before a terminal result was assigned.
- Mixing self-play batching assumptions into match-play latency-sensitive inference.

## Next Step

Run Phase 13 with a real TensorRT-backed evaluator through `selfplay`, collect a baseline Nsight Systems trace and throughput report, then decide whether fixed-shape CUDA Graph replay is worth testing for the inference worker. Reanalysis remains out of scope.
