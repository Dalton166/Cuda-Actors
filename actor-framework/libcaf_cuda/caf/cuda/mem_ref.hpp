// mem_ref.hpp
#pragma once
#include <cuda.h>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include "caf/cuda/global.hpp"
#include "caf/cuda/types.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/utility.hpp"

namespace caf::cuda {

// keep the old name
template <class T>
class mem_ref : public caf::ref_counted {
public:
  using value_type = T;

  // Buffer constructor (old behavior)
  mem_ref(size_t num_elements,
          CUdeviceptr memory,
          int access,
          int device_id    = 0,
          int context_id   = 0,
          CUstream stream  = nullptr)
    : num_elements_(num_elements),
      memory_(memory),
      access_(access),
      device_id(device_id),
      context_id(context_id),
      stream_(stream),
      is_scalar_(false)
  {
    if (memory_ == 0)
      std::abort(); // unchanged
  }

  // Scalar constructor (new!)
  mem_ref(const T& scalar_value,
          int access,
          int device_id    = 0,
          int context_id   = 0,
          CUstream stream  = nullptr)
    : num_elements_(1),
      memory_(0),                    // no device buffer
      access_(access),
      device_id(device_id),
      context_id(context_id),
      stream_(stream),
      is_scalar_(true),
      host_scalar_(scalar_value)
  {
    // no cuMemAlloc, nothing to do
  }

  ~mem_ref() {
    reset();
  }

  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;
  mem_ref(const mem_ref&) = delete;
  mem_ref& operator=(const mem_ref&) = delete;

  bool is_scalar() const noexcept {
    return is_scalar_;
  }

  const T* host_scalar_ptr() const noexcept {
    return &host_scalar_;
  }

  size_t size()  const noexcept { return num_elements_; }
  CUdeviceptr mem()   const noexcept { return memory_; }
  int access()  const noexcept { return access_; }
  CUstream stream() const noexcept { return stream_; }

  void reset() {
    if (!is_scalar_ && memory_) {
      CHECK_CUDA(cuMemFree(memory_));
      memory_ = 0;
    }
    num_elements_ = 0;
    access_       = -1;
    stream_       = nullptr;
  }

  std::vector<T> copy_to_host() const {
    if (is_scalar_) {
      return std::vector<T>{host_scalar_};
    }
    std::vector<T> host_data(num_elements_);
    size_t bytes = num_elements_ * sizeof(T);
    auto& mgr = manager::get();
    CUcontext ctx = getContextById(device_id, context_id);
    CHECK_CUDA(cuCtxPushCurrent(ctx));
    CUstream s = stream_ ? stream_ : nullptr;
    CHECK_CUDA(cuMemcpyDtoHAsync(host_data.data(), memory_, bytes, s));
    if (s) CHECK_CUDA(cuStreamSynchronize(s));
    else  CHECK_CUDA(cuCtxSynchronize());
    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    return host_data;
  }

private:
  size_t      num_elements_{0};
  CUdeviceptr memory_{0};
  int         access_{-1};
  int         device_id{0};
  int         context_id{0};
  CUstream    stream_{nullptr};

  bool is_scalar_{false};
  T    host_scalar_{};
};

template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

} // namespace caf::cuda

