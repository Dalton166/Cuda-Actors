#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>
#include "caf/cuda/global.hpp"


namespace caf::cuda {

template <class T>
class mem_ref; // Forward-declared; full definition in mem_ref.hpp

class device;
using device_ptr = caf::intrusive_ptr<device>;

class device : public caf::ref_counted {
public:
  device([[maybe_unused]] void* device, [[maybe_unused]] void* context, [[maybe_unused]] void* queue) {
    throw std::runtime_error("CUDA support disabled: device ctor");
  }

  device(CUdevice device,CUcontext context,char * name, int number) {

	  device_ = device;
	  context_ = context;
	  name_ = name;
	  id_ = number;
  }


  //~device() override;

  //device

  std::string vendor() const;
  int type() const;
  std::size_t global_mem_size() const;
  std::size_t max_mem_alloc_size() const;
  std::size_t max_work_group_size() const;

  template <class T>
  mem_ref<T> make_arg(T*, std::size_t, void* = nullptr) {
    throw std::runtime_error("CUDA support disabled: device::make_arg()");
  }


  CUdevice getDevice() { return device_;}


private:
  //void* id_;
  int id_;
  CUcontext context_;
  void* queue_;
  int type_;
  char * name_;
  std::size_t global_mem_size_;
  std::size_t max_mem_alloc_size_;
  std::size_t max_work_group_size_;
  CUdevice device_;
};

} // namespace caf::cuda
