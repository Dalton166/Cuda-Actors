#include "caf/cuda/platform.hpp"
#include <iostream>

namespace caf::cuda {

platform_ptr platform::create() {
  static platform_ptr instance;
  if (!instance) {
    std::cout << "[platform] constructor called\n";
    instance = make_counted<platform>();
  }
  return instance;
}

platform::platform() {
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

scheduler_ = (device_count <= 1)
    ? std::unique_ptr<scheduler>{std::make_unique<single_device_scheduler>()}
    : std::unique_ptr<scheduler>{std::make_unique<multi_device_scheduler>()};
  
    scheduler_->set_devices(devices_);

  if (device_count > 0) {
    check(cuCtxSetCurrent(contexts_[0]), "cuCtxSetCurrent");
  }
}

platform::~platform() {
  std::cout << "Destroying platform\n";
  // Context cleanup is handled in device destructors
}

const std::string& platform::name() const {
  return name_;
}

const std::string& platform::vendor() const {
  return vendor_;
}

const std::string& platform::version() const {
  return version_;
}

const std::vector<device_ptr>& platform::devices() const {
  return devices_;
}

device_ptr platform::getDevice(int id) {
  if (id < 0 || static_cast<size_t>(id) >= devices_.size()) {
    throw std::out_of_range("Invalid device ID");
  }
  return devices_[id];
}

scheduler* platform::get_scheduler() {
  return scheduler_.get();
}

device_ptr platform::schedule(int actor_id) {
  
	//std::cout <<  "schedule called\n";
	return scheduler_->schedule(actor_id);
}

void platform::release_streams_for_actor(int actor_id) {
  for (auto& dev : devices_) {
    dev->release_stream_for_actor(actor_id);
  }
}

} // namespace caf::cuda

