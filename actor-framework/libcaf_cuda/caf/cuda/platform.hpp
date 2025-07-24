#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>
#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/scheduler.hpp"

namespace caf::cuda {

class platform : public ref_counted {
public:
  friend class program;
  template <class T, class... Ts>
  friend intrusive_ptr<T> caf::make_counted(Ts&&...);

  static platform_ptr create() {
    static platform_ptr instance;
    if (!instance) {
      
	     std::cout << "[platform] constructor called\n";
	    instance = make_counted<platform>();
    }
    return instance;
  }

  inline const std::string& name() const;
  inline const std::string& vendor() const;
  inline const std::string& version() const;
  inline const std::vector<device_ptr>& devices() const;

  device_ptr getDevice(int id) {
    if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
      throw std::out_of_range("Invalid device ID");
    }
    return devices_[id];
  }

  scheduler* get_scheduler() { return scheduler_.get(); }

  device_ptr schedule(int actor_id) {
    return scheduler_->schedule(actor_id);
  }

  void release_streams_for_actor(int actor_id) {
    for (auto& dev : devices_) {
      dev->release_stream_for_actor(actor_id);
    }
  }

private:
  platform() {
    int device_count = 0;
    check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    devices_.resize(device_count);
    contexts_.resize(device_count);

    std::cout << "Device count is " << device_count << " \n";

    for (int i = 0; i < device_count; ++i) {
      CUdevice cuda_device;
      check(cuDeviceGet(&cuda_device, i), "cuDeviceGet");
      char name[256];
      cuDeviceGetName(name, 256, cuda_device);
      check(cuCtxCreate(&contexts_[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, cuda_device), "cuCtxCreate");
      devices_[i] = make_counted<device>(cuda_device, contexts_[i], name, i);
    }

    
    scheduler_ = (device_count <= 1)?
    std::unique_ptr<scheduler>{std::make_unique<single_device_scheduler>()}
  : std::unique_ptr<scheduler>{std::make_unique<multi_device_scheduler>()};

    scheduler_->set_devices(devices_);

    if (device_count > 0) {
      check(cuCtxSetCurrent(contexts_[0]), "cuCtxSetCurrent");
    }
  }

  ~platform()  {
	  std::cout << "Destroying platform\n";
    // Context destruction handled by device destructors
  }

  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
  std::vector<CUcontext> contexts_;
  std::unique_ptr<scheduler> scheduler_;
};

inline const std::vector<device_ptr>& platform::devices() const { return devices_; }
inline const std::string& platform::name() const { return name_; }
inline const std::string& platform::vendor() const { return vendor_; }
inline const std::string& platform::version() const { return version_; }
inline void intrusive_ptr_add_ref(platform* p) { p->ref(); }
inline void intrusive_ptr_release(platform* p) { p->deref(); }

} // namespace caf::cuda
