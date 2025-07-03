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

template <class T>
class mem_ref : public caf::ref_counted {
public:
  using value_type = T;

  mem_ref() = default;

  mem_ref(size_t num_elements, CUdeviceptr memory, int access, int device_id = 0, int context_id = 0, CUstream stream = nullptr)
      : num_elements_{num_elements},
        memory_{memory},
        access_{access},
        device_id{device_id},
        context_id{context_id},
        stream_(stream) {
    if (memory_ == 0) {
      std::cerr << "Fatal error: mem_ref constructor received null device pointer.\n"
                << "  -> num_elements: " << num_elements_ << "\n"
                << "  -> access: " << access_ << "\n"
                << "  -> device_id: " << device_id << ", context_id: " << context_id << std::endl;
      std::abort();
    }
  }

  ~mem_ref() {
    reset();
  }

  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;

  mem_ref(const mem_ref&) = delete;
  mem_ref& operator=(const mem_ref&) = delete;

  size_t size() const { return num_elements_; }
  CUdeviceptr mem() const { return memory_; }
  int access() const { return access_; }
  CUstream stream() const { return stream_; }

  void reset() {
    if (memory_) {
      CHECK_CUDA(cuMemFree(memory_));
      memory_ = 0;
    }
    num_elements_ = 0;
    access_ = -1;
    stream_ = nullptr;
  }

  std::vector<T> copy_to_host() const {
    if (access_ != OUT && access_ != IN_OUT) {
      throw std::runtime_error("Attempt to read from a non-output memory region.");
    }

    std::vector<T> host_data(num_elements_);
    size_t bytes = num_elements_ * sizeof(T);

    auto& mgr = manager::get();
    CUcontext ctx = getContextById(device_id, context_id);
    if (!ctx) {
      throw std::runtime_error("Invalid context in copy_to_host");
    }

    CHECK_CUDA(cuCtxPushCurrent(ctx));

    // Async copy device->host on the mem_ref's stream if set, else default stream
    CUstream s = stream_ ? stream_ : nullptr;

    CHECK_CUDA(cuMemcpyDtoHAsync(host_data.data(), memory_, bytes, s));

    // Synchronize the stream to ensure copy completes before accessing host data
    if (s) {
      CHECK_CUDA(cuStreamSynchronize(s));
    } else {
      CHECK_CUDA(cuCtxSynchronize());
    }

    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return host_data;
  }

private:
  size_t num_elements_ = 0;
  CUdeviceptr memory_ = 0;
  int access_ = -1;
  int device_id = 0;
  int context_id = 0;
  CUstream stream_ = nullptr;
};

template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

} // namespace caf::cuda

