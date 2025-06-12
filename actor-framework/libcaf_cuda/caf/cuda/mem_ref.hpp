#pragma once

#include <cuda.h>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include <vector>
#include <stdexcept>
#include "caf/cuda/global.hpp"
#include "caf/cuda/types.hpp"
#include "caf/cuda/manager.hpp"  // for manager::get()
#include "caf/cuda/utility.hpp"  // for get_context_by_id()


namespace caf::cuda {

template <class T>
class mem_ref : public caf::ref_counted {
public:
  using value_type = T;

  mem_ref() = default;

  mem_ref(size_t num_elements, CUdeviceptr memory, int access, int device_id = 0, int context_id = 0)
    : num_elements_{num_elements},
      memory_{memory},
      access_{access},
      device_id{device_id},
      context_id{context_id} {
    // nop
  }

  ~mem_ref() {
    reset();
  }

  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;

  // Disable copy
  mem_ref(const mem_ref&) = delete;
  mem_ref& operator=(const mem_ref&) = delete;

  size_t size() const { return num_elements_; }
  CUdeviceptr mem() const { return memory_; }
  int access() const { return access_; }

  void reset() {
    if (memory_) {
      cuMemFree(memory_);
      memory_ = 0;
    }
    num_elements_ = 0;
    access_ = -1;
  }

  std::vector<T> copy_to_host() const {
    if (access_ != OUT && access_ != IN_OUT) {
      throw std::runtime_error("Attempt to read from a non-output memory region.");
    }

    std::vector<T> host_data(num_elements_);
    size_t bytes = num_elements_ * sizeof(T);

    // Use manager singleton and helper function to get context
    auto& mgr = manager::get(); // Throws if not initialized
    CUcontext ctx = getContextById(device_id, context_id);

    CHECK_CUDA(cuCtxPushCurrent(ctx));
    CHECK_CUDA(cuMemcpyDtoH(host_data.data(), memory_, bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return host_data;
  }

private:
  size_t num_elements_ = 0;
  CUdeviceptr memory_ = 0;
  int access_ = -1;
  int device_id = 0;
  int context_id = 0;
};

// Alias for intrusive pointer to mem_ref<T>
template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

} // namespace caf::cuda

