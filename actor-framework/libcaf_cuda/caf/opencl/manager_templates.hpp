#pragma once

#include <stdexcept>
#include <caf/opencl/opencl_err.hpp>
#include <caf/opencl/actor_facade.hpp>
#include <caf/actor.hpp>

namespace caf::opencl {

template <class Predicate>
device_ptr manager::find_device_if(Predicate&&) const {
  throw std::runtime_error("OpenCL support disabled: manager::find_device_if");
}

template <bool PassConfig, class Result, class... Ts>
caf::actor manager::spawn(const char*,
                          program_ptr,
                          Ts&&...) {
  throw std::runtime_error("OpenCL support disabled: manager::spawn");
}

} // namespace caf::opencl
