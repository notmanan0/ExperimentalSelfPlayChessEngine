#include <chessmoe/inference/tensorrt_engine.h>

#include <fstream>
#include <stdexcept>
#include <utility>

#if defined(CHESSMOE_ENABLE_TENSORRT)
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <iostream>
#include <vector>
#endif

namespace chessmoe::inference {
namespace {

std::vector<char> read_binary_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open TensorRT engine: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

#if defined(CHESSMOE_ENABLE_TENSORRT)
void check_cuda(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " +
                             cudaGetErrorString(status));
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "TensorRT: " << message << '\n';
    }
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T* ptr) const noexcept {
    delete ptr;
  }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter<T>>;

struct DeviceBuffer {
  void* ptr{nullptr};
  std::size_t bytes{0};

  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t byte_count) : bytes(byte_count) {
    check_cuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      (void)cudaFree(ptr);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr(std::exchange(other.ptr, nullptr)),
        bytes(std::exchange(other.bytes, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr != nullptr) {
        (void)cudaFree(ptr);
      }
      ptr = std::exchange(other.ptr, nullptr);
      bytes = std::exchange(other.bytes, 0);
    }
    return *this;
  }
};
#endif

}  // namespace

struct TensorRTEngine::Impl {
#if defined(CHESSMOE_ENABLE_TENSORRT)
  Logger logger;
  TrtPtr<nvinfer1::IRuntime> runtime;
  TrtPtr<nvinfer1::ICudaEngine> engine;
  TrtPtr<nvinfer1::IExecutionContext> context;
  cudaStream_t stream{nullptr};
  DeviceBuffer input;
  DeviceBuffer policy;
  DeviceBuffer wdl;
  DeviceBuffer moves_left;

  explicit Impl(const TensorRTEngineConfig& config) {
    check_cuda(cudaSetDevice(config.device_id), "cudaSetDevice");
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    auto bytes = read_binary_file(config.engine_path);
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
      throw std::runtime_error("failed to create TensorRT runtime");
    }
    engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine) {
      throw std::runtime_error("failed to deserialize TensorRT engine");
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
      throw std::runtime_error("failed to create TensorRT execution context");
    }

    if (!context->setOptimizationProfileAsync(0, stream)) {
      throw std::runtime_error("failed to set TensorRT optimization profile 0");
    }

    const auto max_batch = config.max_batch;
    const auto layout = config.layout;
    input = DeviceBuffer(max_batch * layout.input_elements_per_position() *
                         sizeof(float));
    policy = DeviceBuffer(max_batch * layout.policy_buckets * sizeof(float));
    wdl = DeviceBuffer(max_batch * 3 * sizeof(float));
    moves_left = DeviceBuffer(max_batch * sizeof(float));
  }

  ~Impl() {
    if (stream != nullptr) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamDestroy(stream);
    }
  }
#endif
};

TensorRTEngine::TensorRTEngine(TensorRTEngineConfig config)
    : config_(std::move(config)) {
  if (config_.max_batch == 0) {
    throw std::invalid_argument("max_batch must be positive");
  }
  if (config_.enable_cuda_graphs) {
    throw std::invalid_argument(
        "CUDA Graphs are intentionally disabled until the fixed-shape path is stable");
  }
#if defined(CHESSMOE_ENABLE_TENSORRT)
  impl_ = std::make_unique<Impl>(config_);
#else
  if (!config_.engine_path.empty()) {
    (void)read_binary_file(config_.engine_path);
  }
#endif
}

TensorRTEngine::~TensorRTEngine() = default;
TensorRTEngine::TensorRTEngine(TensorRTEngine&&) noexcept = default;
TensorRTEngine& TensorRTEngine::operator=(TensorRTEngine&&) noexcept = default;

TensorLayout TensorRTEngine::layout() const {
  return config_.layout;
}

RawNetworkOutput TensorRTEngine::infer(const NetworkInputBatch& batch) {
  if (batch.batch_size == 0) {
    return {};
  }
  if (batch.batch_size > config_.max_batch) {
    throw std::invalid_argument("TensorRT batch exceeds configured max_batch");
  }
  const auto expected_features =
      batch.batch_size * config_.layout.input_elements_per_position();
  if (batch.features.size() != expected_features) {
    throw std::invalid_argument("TensorRT input feature size does not match layout");
  }

#if defined(CHESSMOE_ENABLE_TENSORRT)
  auto& trt = *impl_;
  const auto& layout = config_.layout;
  const auto batch_n = static_cast<int>(batch.batch_size);
  const nvinfer1::Dims4 input_shape{
      batch_n, static_cast<int>(layout.channels), static_cast<int>(layout.height),
      static_cast<int>(layout.width)};
  if (!trt.context->setInputShape(layout.input_name.c_str(), input_shape)) {
    throw std::runtime_error("failed to set TensorRT input shape");
  }

  if (!trt.context->setTensorAddress(layout.input_name.c_str(), trt.input.ptr) ||
      !trt.context->setTensorAddress(layout.policy_output_name.c_str(),
                                     trt.policy.ptr) ||
      !trt.context->setTensorAddress(layout.wdl_output_name.c_str(), trt.wdl.ptr) ||
      !trt.context->setTensorAddress(layout.moves_left_output_name.c_str(),
                                     trt.moves_left.ptr)) {
    throw std::runtime_error("failed to bind TensorRT tensor addresses");
  }

  check_cuda(cudaMemcpyAsync(trt.input.ptr, batch.features.data(),
                             batch.features.size() * sizeof(float),
                             cudaMemcpyHostToDevice, trt.stream),
             "copy TensorRT input to device");
  if (!trt.context->enqueueV3(trt.stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  RawNetworkOutput output;
  output.batch_size = batch.batch_size;
  output.policy_logits.resize(batch.batch_size * layout.policy_buckets);
  output.wdl_logits.resize(batch.batch_size * 3);
  output.moves_left.resize(batch.batch_size);

  check_cuda(cudaMemcpyAsync(output.policy_logits.data(), trt.policy.ptr,
                             output.policy_logits.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, trt.stream),
             "copy TensorRT policy output to host");
  check_cuda(cudaMemcpyAsync(output.wdl_logits.data(), trt.wdl.ptr,
                             output.wdl_logits.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, trt.stream),
             "copy TensorRT WDL output to host");
  check_cuda(cudaMemcpyAsync(output.moves_left.data(), trt.moves_left.ptr,
                             output.moves_left.size() * sizeof(float),
                             cudaMemcpyDeviceToHost, trt.stream),
             "copy TensorRT moves-left output to host");
  check_cuda(cudaStreamSynchronize(trt.stream), "synchronize TensorRT stream");
  return output;
#else
  throw std::runtime_error(
      "TensorRT support is not compiled in. Reconfigure with "
      "CHESSMOE_ENABLE_TENSORRT=ON and provide CUDA/TensorRT libraries.");
#endif
}

std::string tensorrt_build_status() {
#if defined(CHESSMOE_ENABLE_TENSORRT)
  return "compiled-with-tensorrt";
#else
  return "not-compiled-with-tensorrt";
#endif
}

}  // namespace chessmoe::inference
