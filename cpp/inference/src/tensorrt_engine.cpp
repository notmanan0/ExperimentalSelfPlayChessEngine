#include <chessmoe/inference/tensorrt_engine.h>

#include <stdexcept>
#include <utility>

namespace chessmoe::inference {

TensorRTEngine::TensorRTEngine(TensorRTEngineConfig config)
    : config_(std::move(config)) {
  if (config_.max_batch == 0) {
    throw std::invalid_argument("max_batch must be positive");
  }
  if (config_.enable_cuda_graphs) {
    throw std::invalid_argument(
        "CUDA Graphs are intentionally disabled until the fixed-shape path is stable");
  }
}

TensorLayout TensorRTEngine::layout() const {
  return config_.layout;
}

RawNetworkOutput TensorRTEngine::infer(const NetworkInputBatch& batch) {
  (void)batch;
#if defined(CHESSMOE_ENABLE_TENSORRT)
  throw std::runtime_error(
      "TensorRT runtime binding is not wired yet; use the ONNX Runtime TensorRT "
      "path or complete the nvinfer integration behind this wrapper");
#else
  throw std::runtime_error(
      "TensorRT support is not compiled in. Reconfigure with "
      "CHESSMOE_ENABLE_TENSORRT and provide CUDA/TensorRT libraries.");
#endif
}

std::string tensorrt_build_status() {
#if defined(CHESSMOE_ENABLE_TENSORRT)
  return "compiled-with-tensorrt-flag";
#else
  return "not-compiled-with-tensorrt";
#endif
}

}  // namespace chessmoe::inference
