#pragma once

#include <string_view>

namespace chessmoe::inference {

enum class BackendKind {
  Random,
  Material,
  Torch,
  Onnx,
  TensorRt,
  CudaNative,
};

constexpr std::string_view backend_name(BackendKind kind) {
  switch (kind) {
    case BackendKind::Random:
      return "random";
    case BackendKind::Material:
      return "material";
    case BackendKind::Torch:
      return "torch";
    case BackendKind::Onnx:
      return "onnx";
    case BackendKind::TensorRt:
      return "tensorrt";
    case BackendKind::CudaNative:
      return "cuda-native";
  }
  return "unknown";
}

}  // namespace chessmoe::inference
