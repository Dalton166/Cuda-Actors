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
program(std::string name, std::vector<char> binary, bool is_fatbin = false)
    : name_(std::move(name)), binary_(std::move(binary)) {

  auto plat = platform::create();

  for (const auto& dev : plat->devices()) {
    CUcontext ctx = dev->getContext();
    CHECK_CUDA(cuCtxPushCurrent(ctx));

    CUmodule module;
    if (is_fatbin) {
      // Load fatbinary directly
      CUresult res = cuModuleLoadFatBinary(&module, binary_.data());
      if (res != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to load fatbinary for device " + std::to_string(dev->getId()));
      }
    } else {
      // Load PTX (driver will JIT-compile for the device)
      CHECK_CUDA(cuModuleLoadData(&module, binary_.data()));
    }

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
  std::vector<char> binary_; //the binary or ptx of the program
  std::unordered_map<int, CUfunction> kernels_;
};

using program_ptr = caf::intrusive_ptr<program>;

} // namespace caf::cuda
