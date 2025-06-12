#pragma once

#include <cuda.h>
#include "caf/cuda/manager.hpp"

namespace caf::cuda {

// This helper calls manager singleton's get_context_by_id
inline CUcontext get_context_by_id(int device_id, int context_id) {
  auto& mgr = manager::get();
  return mgr.get_context_by_id(device_id, context_id);
}

} // namespace caf::cuda

