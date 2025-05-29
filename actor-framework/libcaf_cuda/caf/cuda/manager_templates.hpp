#pragma once

#include <stdexcept>
#include <caf/cuda/opencl_err.hpp>
#include <caf/cuda/actor_facade.hpp>
#include <caf/actor.hpp>

namespace caf::cuda {

template <class Predicate>
device_ptr manager::find_device_if(Predicate&&) const {
  throw std::runtime_error("CUDA support disabled: manager::find_device_if");
}

template <bool PassConfig, class Result, class... Ts>
caf::actor manager::spawn(const char*,
                          program_ptr,
                          Ts&&...) {
  throw std::runtime_error("CUDA support disabled: manager::spawn");
}

} // namespace caf::cuda
