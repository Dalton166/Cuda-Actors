#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>
#include <cuda.h>

#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"

namespace caf {
namespace cuda {

class platform;
using platform_ptr = intrusive_ptr<platform>;

class platform : public ref_counted {
public:
  friend class program;
  template <class T, class... Ts>
  friend intrusive_ptr<T> caf::make_counted(Ts&&...);

  inline const std::string& name() const;
  inline const std::string& vendor() const;
  inline const std::string& version() const;
  static platform_ptr create();

private:
  platform() {
    int device_count = 0;
    check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    devices_.resize(device_count);
    contexts_.resize(device_count);
    for (int i = 0; i < device_count; ++i) {
      CUdevice device;
      check(cuDeviceGet(&device, i), "cuDeviceGet");
      char name[256];
      cuDeviceGetName(name, 256, device);
      std::cout << "Device #" << i << ": " << name << "\n";
      check(cuCtxCreate(&contexts_[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device), "cuCtxCreate");
    
      devices_[i] = device(device,contexts[i],name,i); 
    }
    int target_device = 0; // Default to first device, ensure valid index
    if (device_count > 0) {
      check(cuCtxSetCurrent(contexts_[target_device]), "cuCtxSetCurrent");
    }
    else {
    
	    std::cout << "No valid device found\n";
	    exit(-1);
    }
  }

  ~platform() override {
    for (auto ctx : contexts_) {
      check(cuCtxDestroy(ctx), "cuCtxDestroy");
    }
  }

  inline const std::vector<device>& devices() const {
    return devices_;
  }

  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device> devices_;
  std::vector<CUcontext> contexts_;
  int target_device = 0;
};

inline const std::vector<device_ptr>& platform::devices() const {
  return devices_;
}

inline const std::string& platform::name() const {
  return name_;
}

inline const std::string& platform::vendor() const {
  return vendor_;
}

inline const std::string& platform::version() const {
  return version_;
}

} // namespace cuda
} // namespace caf
