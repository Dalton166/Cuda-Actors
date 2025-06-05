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
  inline const std::vector<device_ptr>& devices() const;

  static platform_ptr create() {
  static platform_ptr instance;
  if (!instance) {
    instance = make_counted<platform>();
  }
  return instance;
}

device_ptr getDevice(int id) {
	return devices_[id];
}


private:
  platform() {
    int device_count = 0;
    check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    devices_.resize(device_count);
    contexts_.resize(device_count);
    for (int i = 0; i < device_count; ++i) {
      CUdevice cuda_device;
      check(cuDeviceGet(&cuda_device, i), "cuDeviceGet");
      char name[256];
      cuDeviceGetName(name, 256, cuda_device);
      std::cout << "Device #" << i << ": " << name << "\n";
      check(cuCtxCreate(&contexts_[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, cuda_device), "cuCtxCreate");
      devices_[i] = make_counted<device>(cuda_device, contexts_[i], name, i);
    }
    int target_device = 0;
    if (device_count > 0) {
      check(cuCtxSetCurrent(contexts_[target_device]), "cuCtxSetCurrent");
    }
  }

  ~platform() override {
    for (auto ctx : contexts_) {
      check(cuCtxDestroy(ctx), "cuCtxDestroy");
    }
  }

  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
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

inline void intrusive_ptr_add_ref(platform* p) {
  p->ref(); // increases the reference count
}

inline void intrusive_ptr_release(platform* p) {
  p->deref(); // decreases the reference count and deletes if 0
}



} // namespace cuda
} // namespace caf
