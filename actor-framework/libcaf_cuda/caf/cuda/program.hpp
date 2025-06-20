#pragma once

#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>

#include <caf/ref_counted.hpp>
#include <caf/response_promise.hpp>
#include <caf/actor_control_block.hpp>
#include <caf/message.hpp>

#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {
class program : public caf::ref_counted {
public:
  program(void*, void*, void*, std::map<std::string, void*>) {
    throw std::runtime_error("CUDA support disabled: program ctor");
  }
~program() {
    std::cout << "Destroying program, kernel=" << kernel_ << "\n";
}


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


  CUcontext ctx = device_->getContext(context_id);
	  //  CUcontext current_ctx;
//CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
//if (current_ctx != ctx) {
  //throw std::runtime_error("Context mismatch during program construction.");
//}


  // Ensure the correct context is active
  CHECK_CUDA(cuCtxPushCurrent(ctx));

  CUmodule module;
  CHECK_CUDA(cuModuleLoadData(&module, ptx.data()));

  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, name_.c_str()));

  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  kernel_ = kernel;
}


//some getter methods
int get_device_id() const { return device_id; }
int get_context_id() const { return context_id; }
int get_stream_id() const { return stream_id;}
CUfunction get_kernel() const { return kernel_;}
device_ptr get_device() const {return device_;}

private:
 std::string name_;
 CUfunction kernel_; //the compiled and loaded program on a specific device
 
 //id's required for identifying where it's supposed to execute
 int device_id;
 int context_id;
 int stream_id;
 device_ptr device_; //the device it was compiled on, this is not ideal but it is neccesary to break a circular dependency
};

using program_ptr = caf::intrusive_ptr<program>;

} // namespace caf::cuda
