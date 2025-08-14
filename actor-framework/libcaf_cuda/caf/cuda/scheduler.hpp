#pragma once

#include <vector>
#include <tuple>
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"


namespace caf::cuda {

class device; // Forward declaration
using device_ptr = caf::intrusive_ptr<device>;

class scheduler {
public:
  virtual ~scheduler() = default;

  // Returns the device an actor should run on
  virtual device_ptr schedule( [[maybe_unused]] int actor_id) = 0;

  virtual device_ptr schedule([[maybe_unused]] int actor_id, [[maybe_unused]]int device_number) = 0;

  // Assigns the context and stream for the scheduled device
  virtual void getStreamAndContext(int actor_id, CUcontext* context, CUstream* stream) = 0;

  // Checks arguments for mem_refs and returns the device of the first mem_ref found
  template <typename... Ts>
  device_ptr checkArgs(const Ts&... args) {
    return checkArgsImpl(std::make_tuple(args...));
  }

  // Converts mem_refs to the target device if necessary
  template <typename... Ts>
  void convertArgs(device_ptr target_device, Ts&... args) {
    convertArgsImpl(target_device, std::make_tuple(std::ref(args)...));
  }

  // Sets the list of devices
  virtual void set_devices(const std::vector<device_ptr>& devices) = 0;

protected:
  virtual device_ptr checkArgsImpl(const std::tuple<>&) {
    return nullptr;
  }

  template <typename T, typename... Rest>
  device_ptr checkArgsImpl(const std::tuple<T, Rest...>& args) {
    if constexpr (is_mem_ptr<T>::value) {
      if (std::get<0>(args)) {
        return find_device_by_id(std::get<0>(args)->deviceID());
      }
    }
    return checkArgsImpl(std::tuple<Rest...>(std::get<Rest>(args)...));
  }

  virtual void convertArgsImpl(device_ptr, std::tuple<>&) {}

  template <typename T, typename... Rest>
  void convertArgsImpl(device_ptr target_device, std::tuple<T, Rest...>& args) {
    if constexpr (is_mem_ptr<T>::value) {
      auto& mem = std::get<0>(args);
      if (mem && mem->deviceID() != target_device->getId()) {
        mem = transfer_mem_ref(mem, target_device);
      }
    }
    convertArgsImpl(target_device, std::tuple<Rest...>(std::get<Rest>(args)...));
  }

private:
  template <typename T>
  struct is_mem_ptr : std::false_type {};

  template <typename T>
  struct is_mem_ptr<mem_ptr<T>> : std::true_type {};

  virtual device_ptr find_device_by_id(int id) = 0;

  //TODO make this work with streams 
  template <typename T>
  mem_ptr<T> transfer_mem_ref(const mem_ptr<T>& src, device_ptr target_device) {
    if (src->is_scalar()) {
      return mem_ptr<T>(new mem_ref<T>(src->host_scalar_ptr()[0], src->access(),
                                       target_device->getId(), 0, nullptr));
    }
    auto host_data = src->copy_to_host();
    size_t bytes = src->size() * sizeof(T);
    CUdeviceptr new_mem;
    CHECK_CUDA(cuCtxPushCurrent(target_device->getContext()));
    CHECK_CUDA(cuMemAlloc(&new_mem, bytes));
    CHECK_CUDA(cuMemcpyHtoD(new_mem, host_data.data(), bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));
    return mem_ptr<T>(new mem_ref<T>(src->size(), new_mem, src->access(),
                                     target_device->getId(), 0, nullptr));
  }
};

class single_device_scheduler : public scheduler {
public:
  void set_devices(const std::vector<device_ptr>& devices) override {
    devices_ = devices;
  }

  device_ptr schedule( [[maybe_unused]] int actor_id) override {
    if (devices_.empty()) {
      throw std::runtime_error("No devices available");
    }
    return devices_[0];
  }
  
  device_ptr schedule( [[maybe_unused]] int actor_id, [[maybe_unused]] int device_number) override {
    if (devices_.empty()) {
      throw std::runtime_error("No devices available");
    }
    return devices_[0];
  }




  void getStreamAndContext(int actor_id, CUcontext* context, CUstream* stream) override {
    auto dev = schedule(actor_id);
    *context = dev->getContext();
    *stream = dev->get_stream_for_actor(actor_id);
  }

  template <typename... Ts>
  device_ptr checkArgs(const Ts&...) {
    return nullptr;
  }

  template <typename... Ts>
  void convertArgs(device_ptr, Ts&...) {}

private:
  device_ptr find_device_by_id(int id) override {
    return (devices_.empty() || devices_[0]->getId() != id) ? nullptr : devices_[0];
  }

  std::vector<device_ptr> devices_;
};

class multi_device_scheduler : public scheduler {
public:
  void set_devices(const std::vector<device_ptr>& devices) override {
    devices_ = devices;
  }

  device_ptr schedule( [[maybe_unused]] int actor_id) override {
    if (devices_.empty()) {
      throw std::runtime_error("No devices available");
    }
    size_t num_devices = devices_.size();
    size_t device_index = static_cast<size_t>(random_number()) % num_devices;
    //std::cout << "picking device with id of " << devices_[device_index] -> getId() << " \n";
    return devices_[device_index];
  }




  device_ptr schedule( [[maybe_unused]] int actor_id,int device_number) override {
    if (devices_.empty()) {
      throw std::runtime_error("No devices available");
    }
    size_t num_devices = devices_.size();
    size_t device_index = static_cast<size_t>(device_number) % num_devices;
    //std::cout << "picking device with id of " << devices_[device_index] -> getId() << " \n";
    return devices_[device_index];
  
  } 


  void getStreamAndContext(int actor_id, CUcontext* context, CUstream* stream) override {
    auto dev = schedule(actor_id);
    *context = dev->getContext();
    *stream = dev->get_stream_for_actor(actor_id);
  }

  template <typename... Ts>
  device_ptr checkArgs(const Ts&... args) {
    return scheduler::checkArgs(args...);
  }

  template <typename... Ts>
  void convertArgs(device_ptr target_device, Ts&... args) {
    scheduler::convertArgs(target_device, args...);
  }

private:
  device_ptr find_device_by_id(int id) override {
    for (const auto& dev : devices_) {
      if (dev->getId() == id) {
        return dev;
      }
    }
    return nullptr;
  }

  std::vector<device_ptr> devices_;
};

} // namespace caf::cuda
