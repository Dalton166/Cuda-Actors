#pragma once

#include <cuda.h>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include "caf/cuda/global.hpp"

namespace caf::cuda {

template <class T>
class mem_ref : public caf::ref_counted {
public:
  using value_type = T;

  mem_ref() = default;

  mem_ref(size_t num_elements, CUdeviceptr memory, int access)
    : num_elements_{num_elements},
      memory_{memory},
      access_{access} {
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

private:
  size_t num_elements_ = 0;
  CUdeviceptr memory_ = 0;
  int access_ = -1;
};

// Alias for intrusive pointer to mem_ref<T>
template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

} // namespace caf::cuda

