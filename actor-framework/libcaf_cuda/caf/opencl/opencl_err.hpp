#pragma once

#include <stdexcept>

namespace caf::opencl {

inline void throwcl(const char* /*fname*/, int /*err*/) {
  throw std::runtime_error("OpenCL support disabled");
}

template <class F, class... Ts>
void v1callcl(const char* /*fname*/, F /*f*/, Ts&&...) {
  throw std::runtime_error("OpenCL support disabled");
}

template <class F, class... Ts>
void v2callcl(const char* /*fname*/, F /*f*/, Ts&&...) {
  throw std::runtime_error("OpenCL support disabled");
}

template <class F, class... Ts>
void v3callcl(const char* /*fname*/, F /*f*/, Ts&&...) {
  throw std::runtime_error("OpenCL support disabled");
}

} // namespace caf::opencl
