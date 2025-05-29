#pragma once

#include <vector>
#include <stdexcept>

namespace caf::cuda {
using dim_vec = std::vector<size_t>;

class nd_range {
public:
  nd_range() = default;
  explicit nd_range(const dim_vec&, const dim_vec& = {}, const dim_vec& = {}) {
    throw std::runtime_error("CUDA support disabled: nd_range ctor");
  }
  const dim_vec& dimensions() const {
    throw std::runtime_error("CUDA support disabled: nd_range::dimensions()");
  }
  const dim_vec& offsets() const {
    throw std::runtime_error("CUDA support disabled: nd_range::offsets()");
  }
  const dim_vec& local_dimensions() const {
    throw std::runtime_error("CUDA support disabled: nd_range::local_dimensions()");
  }
};

} // namespace caf::cuda
