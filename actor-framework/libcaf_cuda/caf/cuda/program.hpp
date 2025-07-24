#pragma once

#include <stdexcept>
#include <vector>
#include <unordered_map>

#include <caf/ref_counted.hpp>
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/platform.hpp"

namespace caf::cuda {

class program : public caf::ref_counted {
public:
  program(std::string name, std::vector<char> ptx)
    : name_(std::move(name)), ptx_(std::move(ptx)) {
    auto plat = platform::create();
    for (const auto& dev : plat->devices()) {
      CUcontext ctx = dev->getContext();
      CHECK_CUDA(cuCtxPushCurrent(ctx));
      CUmodule module;
      CHECK_CUDA(cuModuleLoadData(&module, ptx_.data()));
      CUfunction kernel;
      CHECK_CUDA(cuModuleGetFunction(&kernel, module, name_.c_str()));
      CHECK_CUDA(cuCtxPopCurrent(nullptr));
      kernels_[dev->getId()] = kernel;
    }
  }

  CUfunction get_kernel(int device_id) {
    auto it = kernels_.find(device_id);
    if (it == kernels_.end()) {
      throw std::runtime_error("Kernel not found for device ID: " + std::to_string(device_id));
    }
    return it->second;
  }

private:
  std::string name_;
  std::vector<char> ptx_;
  std::unordered_map<int, CUfunction> kernels_;
};

using program_ptr = caf::intrusive_ptr<program>;

} // namespace caf::cuda
