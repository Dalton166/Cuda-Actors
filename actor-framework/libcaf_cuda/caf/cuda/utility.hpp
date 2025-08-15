#pragma once

#include <cuda.h>
#include "caf/cuda/types.hpp"
#include <type_traits>




namespace caf::cuda {

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

// Create `out<T>` from scalar or vector
template <typename T>
out<T> create_out_arg_with_size(int size) {
  return out<T>{size};
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

