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
	  streamId_ = 0;
	  contextId_ = 0;

	 CHECK_CUDA(cuStreamCreate(&stream_,0));
	  
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


  //Some simple getter methods 
  CUdevice getDevice() { return device_;}
  int getId() { return id_;}
  int getStreamId() { return streamId_;}
  int getContextId() { return contextId_;}  

private:
  //void* id_;
  int id_;
  int streamId_;
  int contextId_;
  CUcontext context_;
  void* queue_;
  int type_;
  char * name_;
  std::size_t global_mem_size_;
  std::size_t max_mem_alloc_size_;
  std::size_t max_work_group_size_;
  CUdevice device_;
  CUstream stream_;
};

} // namespace caf::cuda
