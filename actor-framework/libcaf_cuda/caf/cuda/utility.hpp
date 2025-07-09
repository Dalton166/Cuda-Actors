#pragma once

#include <cuda.h>
#include "caf/cuda/types.hpp"
#include "caf/cuda/manager.hpp"
#include <type_traits>




/*
 * Most of these functions are just helper functions for 
 * requesting a resource or operation from the manager
 */
namespace caf::cuda {

// This helper calls manager singleton's get_context_by_id
inline CUcontext getContextById(int device_id, int context_id) {
  auto& mgr = manager::get();
  return mgr.get_context_by_id(device_id, context_id);
}



template <typename T>
in<T> create_in_arg(const T& val) {
  return in<T>{val};
}

template <typename T>
in<T> create_in_arg(const std::vector<T>& buffer) {
  return in<T>{buffer};
}

// Create `out<T>` from scalar or vector
template <typename T>
out<T> create_out_arg(const T& val) {
  return out<T>{val};
}

template <typename T>
out<T> create_out_arg(const std::vector<T>& buffer) {
  return out<T>{buffer};
}

// Create `in_out<T>` from scalar or vector
template <typename T>
in_out<T> create_in_out_arg(const T& val) {
  return in_out<T>{val};
}

template <typename T>
in_out<T> create_in_out_arg(const std::vector<T>& buffer) {
  return in_out<T>{buffer};
}

}

