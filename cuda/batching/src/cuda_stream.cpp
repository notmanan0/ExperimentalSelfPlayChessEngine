#include <chessmoe/cuda/batching/cuda_stream.h>

namespace chessmoe::cuda::batching {

CudaStreamView::CudaStreamView(void* handle, bool owned)
    : handle_(handle), owned_(owned) {}

void* CudaStreamView::get() const {
  return handle_;
}

bool CudaStreamView::owns_stream() const {
  return owned_;
}

CudaStreamView::operator bool() const {
  return handle_ != nullptr;
}

}  // namespace chessmoe::cuda::batching
