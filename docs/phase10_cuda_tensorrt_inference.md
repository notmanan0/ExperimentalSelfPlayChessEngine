# Phase 10: CUDA/TensorRT Batched Inference

Checked against current official documentation on 2026-05-10.

Sources:

- NVIDIA CUDA Runtime stream management documents `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamSynchronize`, and stream capture APIs: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- NVIDIA TensorRT dynamic-shape guidance requires runtime dimensions marked as `-1`, build-time optimization profiles, and runtime input-shape selection before enqueue: https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/inference-library/work-dynamic-shapes.html
- NVIDIA TensorRT `IExecutionContext::setTensorAddress` / `enqueueV3` documentation requires non-null tensor addresses before enqueue: https://docs.nvidia.com/deeplearning/tensorrt/10.10.0/_static/c-api/classnvinfer1_1_1_i_execution_context.html
- NVIDIA TensorRT Python API documents ONNX parsing and serialized engine generation through `tensorrt.OnnxParser`: https://docs.nvidia.com/deeplearning/tensorrt/10.16.0/_static/python-api/parsers/Onnx/pyOnnx.html
- NVIDIA TensorRT command-line documentation states `trtexec` is useful for benchmarking ONNX networks, generating serialized engines, and generating timing caches: https://docs.nvidia.com/deeplearning/tensorrt/10.10.0/reference/command-line-programs.html
- PyTorch ONNX documentation says `torch.onnx.export(..., dynamo=True)` is the modern exporter path and `dynamic_shapes` is preferred with `dynamo=True`: https://docs.pytorch.org/docs/stable/onnx.html
- ONNX shape inference documentation describes symbolic dimensions through `dim_param` and the `onnx.shape_inference` API: https://onnx.ai/onnx/repo-docs/ShapeInference.html
- ONNX Runtime TensorRT Execution Provider documentation covers provider options including `trt_fp16_enable`, engine/timing cache options, explicit profile shapes, and `user_compute_stream`: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

## Objective

Add production-oriented batched inference for the existing tiny PyTorch baseline before any transformer work. The phase introduces ONNX export, TensorRT engine-build tooling, runtime batch-planning contracts, and a `TensorRTEvaluator` adapter that still implements the existing backend-agnostic `IBatchEvaluator` interface.

The current implementation keeps policy legality outside TensorRT: the network returns dense policy logits, and `TensorRTEvaluator` maps only legal moves into `EvaluationResult` before calling the existing legal-mask normalizer.

## Files To Create

- `cpp/inference/include/chessmoe/inference/tensor_layout.h`
- `cpp/inference/include/chessmoe/inference/tensorrt_evaluator.h`
- `cpp/inference/include/chessmoe/inference/tensorrt_engine.h`
- `cpp/inference/src/tensor_layout.cpp`
- `cpp/inference/src/tensorrt_evaluator.cpp`
- `cpp/inference/src/tensorrt_engine.cpp`
- `cuda/batching/include/chessmoe/cuda/batching/batch_planner.h`
- `cuda/batching/include/chessmoe/cuda/batching/cuda_stream.h`
- `cuda/batching/src/batch_planner.cpp`
- `cuda/batching/src/cuda_stream.cpp`
- `python/export/export_tiny_onnx.py`
- `python/export/build_tensorrt_engine.py`
- `tools/benchmark/benchmark_inference.py`
- `configs/model/tiny_inference.yaml`
- `tests/cpp/inference/test_inference.cpp`
- `tests/python/test_phase10_inference_export.py`

## Core Data Structures

- `TensorLayout`: stable input and output tensor names, NCHW shape, and policy bucket count.
- `NetworkInputBatch`: contiguous FP32 features shaped `[N, 18, 8, 8]`.
- `RawNetworkOutput`: dense network outputs shaped `policy_logits [N, 20480]`, `wdl_logits [N, 3]`, and `moves_left [N]`.
- `IInferenceBackend`: swappable raw tensor backend used by `TensorRTEvaluator`.
- `TensorRTEvaluator`: `IBatchEvaluator` adapter that encodes positions, invokes a backend, converts WDL logits to probabilities/value, maps legal moves to dense policy buckets, and applies legal masking.
- `BatchPlanner`: match-play dynamic batching and self-play fixed padded batching policy.
- `CudaStreamView`: opaque CUDA stream handle view for future TensorRT or ONNX Runtime user-stream ownership without requiring CUDA headers in CPU-only builds.
- `TensorRTEngineConfig`: placeholder for engine path, max batch, precision, warmup, profiling, device ID, and CUDA Graphs toggle.

## Main Algorithms

1. Encode each `Position` into the same `[18, 8, 8]` NCHW layout as the Python tiny model.
2. Concatenate positions into a contiguous FP32 batch.
3. Run backend inference on the batch.
4. Validate output tensor sizes against `TensorLayout`.
5. Convert WDL logits with softmax and scalar value as `$P(win) - P(loss)$`.
6. For each legal move, compute its dense policy bucket using the Python-compatible UCI mapping.
7. Keep illegal policy logits out of the result and normalize over legal moves.
8. Use dynamic tail batches for match play and fixed padded batches for self-play.

CUDA Graphs remain disabled because TensorRT and ONNX Runtime CUDA Graph capture are fixed-shape sensitive, and the fixed-shape self-play path must be benchmarked and numerically stable first.

## Tests

- C++ tensor layout stability.
- C++ policy bucket mapping parity with Python constants.
- C++ match-play dynamic tail batching.
- C++ self-play fixed padded batching.
- C++ opaque CUDA stream handle ownership semantics.
- C++ legal policy masking using a fake raw inference backend.
- Python ONNX export stable names and dynamic batch dimension when ONNX dependencies exist.
- Python ONNX Runtime parity against PyTorch when ONNX Runtime exists.
- Python policy index constants for legal masking.

## Commands

```powershell
python -m pytest tests/python/test_phase10_inference_export.py
```

```powershell
cmake -S . -B build-nmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build-nmake --config Debug --target inference_tests
ctest --test-dir build-nmake -R inference_tests --output-on-failure
```

```powershell
python python/export/export_tiny_onnx.py --output artifacts/tiny.onnx
python python/export/build_tensorrt_engine.py --onnx artifacts/tiny.onnx --engine artifacts/tiny_fp32.engine --opt-shape 8x18x8x8 --max-shape 64x18x8x8
python tools/benchmark/benchmark_inference.py --onnx artifacts/tiny.onnx --provider tensorrt --batch-size 8
```

## Completion Criteria

- ONNX export works for the tiny baseline with stable tensor names.
- TensorRT engine-build script accepts explicit min/opt/max batch profiles.
- `TensorRTEvaluator` remains behind `IBatchEvaluator`.
- Match play and self-play have distinct batching policies.
- FP32 is the default path.
- FP16 is configuration-only until parity checks pass.
- CUDA stream ownership is documented for the TensorRT/ORT path, but the local C++ wrapper does not claim a stream until TensorRT linkage is completed.
- Warmup and benchmark hooks exist.
- CUDA Graphs are off.
- Tests cover tensor shapes, policy mapping, batching behavior, legal masking, and PyTorch/ONNX parity where dependencies are installed.

## Common Failure Modes

- Exporting static batch ONNX and then trying to use match-play dynamic batches.
- Forgetting TensorRT optimization profiles for dynamic batch inputs.
- Treating TensorRT engine files as portable across model, TensorRT, ORT, or GPU changes.
- Applying policy softmax over all 20,480 buckets before legal masking.
- Enabling FP16 without comparing logits/value against FP32 and PyTorch.
- Capturing CUDA Graphs before shapes, addresses, and stream usage are stable.
- Letting self-play fixed-batch padding leak into match-play latency.

## Next Step

Wire real TensorRT C++ runtime objects behind `TensorRTEngine` or use the ONNX Runtime TensorRT Execution Provider as the first production path, then run FP32 parity and benchmark reports before enabling FP16.
