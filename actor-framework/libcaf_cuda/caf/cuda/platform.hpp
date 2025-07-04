#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>
//#include <cuda.h>

#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"



// Forward declaration
namespace caf::cuda {
class device;
using device_ptr = intrusive_ptr<device>;
} // namespace caf::cuda



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
  /*
  std::cout << "Requested device ID: " << id << "\n";
  std::cout << "Available devices: " << devices_.size() << "\n";

  for (size_t i = 0; i < devices_.size(); ++i) {
    std::cout << "  Device[" << i << "] = " << (devices_[i] ? "initialized" : "nullptr") << "\n";
  }
   */
  if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
    std::cerr << "getDevice: invalid device ID " << id << "\n";
    throw std::out_of_range("Invalid device ID");
  }

  return devices_[id];
}



private:
platform() {
  int device_count = 0;
  check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
  devices_.resize(device_count);
  contexts_.resize(device_count);

  // Check for existing active context
  CUcontext active_ctx = nullptr;
  CUresult ctx_status = cuCtxGetCurrent(&active_ctx);
  if (ctx_status == CUDA_SUCCESS && active_ctx != nullptr) {
    std::cerr << "[Warning] Unexpected active CUDA context detected before platform initialization.\n";
    std::cerr << "  -> This may indicate the CUDA Runtime API was used elsewhere (e.g., cudaMalloc).\n";
    std::cerr << "  -> Existing context address: " << active_ctx << "\n";

    // Pop the existing context to clear it
    CUcontext popped_ctx = nullptr;
    CUresult pop_status = cuCtxPopCurrent(&popped_ctx);
    if (pop_status == CUDA_SUCCESS && popped_ctx == active_ctx) {
      std::cerr << "  -> Popped existing CUDA context: " << popped_ctx << "\n";
    } else {
      std::cerr << "  -> Failed to pop the existing CUDA context.\n";
      // You might want to throw or handle this error
    }
  }

  for (int i = 0; i < device_count; ++i) {
    CUdevice cuda_device;
    check(cuDeviceGet(&cuda_device, i), "cuDeviceGet");

    char name[256];
    cuDeviceGetName(name, 256, cuda_device);
    //std::cout << "Device #" << i << ": " << name << "\n";

    // Explicit context creation (Driver API only)
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
	    //std::cout << "Ahoi matey\n";
	    //check(cuCtxDestroy(ctx), "cuCtxDestroy");
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
