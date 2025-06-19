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
 program(std::string name,
        device_ptr device,
        int device_id,
        int context_id,
        int stream_id,
        std::vector<char> ptx)
  : name_(std::move(name)),
    device_id(device_id),
    context_id(context_id),
    stream_id(stream_id),
    device_(device) {

  // Ensure the correct context is active
  CUcontext ctx = device_->getContext(context_id);
  CHECK_CUDA(cuCtxPushCurrent(ctx));

  CUmodule module;
  CHECK_CUDA(cuModuleLoadData(&module, ptx.data()));

  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, name_.c_str()));

  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  kernel_ = kernel;
}


  ~platform() override {
    for (auto ctx : contexts_) {
	    std::cout << "Ahoi matey\n";
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
