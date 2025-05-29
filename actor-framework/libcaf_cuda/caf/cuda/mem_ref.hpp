#pragma once

#include <stdexcept>

namespace caf::cuda {

template <class T>
class mem_ref {
public:
  using value_type = T;

  mem_ref() = default;

  mem_ref(void*, void*, size_t) {
    throw std::runtime_error("CUDA support disabled: mem_ref ctor");
  }

  ~mem_ref() = default;
  mem_ref(mem_ref&&) noexcept = default;
  mem_ref& operator=(mem_ref&&) noexcept = default;

  void* queue() const { return nullptr; }
  void* mem() const { return nullptr; }
  size_t size() const { return 0; }
};

} // namespace caf::cuda
