#pragma once

namespace chessmoe::cuda::batching {

class CudaStreamView {
 public:
  CudaStreamView() = default;
  explicit CudaStreamView(void* handle, bool owned = false);

  [[nodiscard]] void* get() const;
  [[nodiscard]] bool owns_stream() const;
  [[nodiscard]] explicit operator bool() const;

 private:
  void* handle_{nullptr};
  bool owned_{false};
};

}  // namespace chessmoe::cuda::batching
