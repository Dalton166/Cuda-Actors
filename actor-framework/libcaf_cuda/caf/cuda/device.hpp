#pragma once

#include <string>
#include <vector>
#include <stdexcept>

#include <cuda.h>
#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/mem_ref.hpp"

namespace caf::cuda {

class device : public caf::ref_counted {
public:
  device(CUdevice device, CUcontext context, char* name, int number)
    : device_{device},
      context_{context},
      name_{name},
      id_{number} {
    streamId_ = 0;
    contextId_ = 0;
    CHECK_CUDA(cuStreamCreate(&stream_, 0));
  }

  CUdevice getDevice() const { return device_; }
  int getId() const { return id_; }
  int getStreamId() const { return streamId_; }
  int getContextId() const { return contextId_; }

  template <typename T>
  mem_ptr<T> make_arg(in<T> arg) {
    return caf::make_counted<mem_ref<T>>(global_argument(std::move(arg)));
  }

  template <typename T>
  mem_ptr<T> make_arg(in_out<T> arg) {
    return caf::make_counted<mem_ref<T>>(global_argument(std::move(arg)));
  }

  template <typename T>
  mem_ptr<T> make_arg(out<T> arg) {
    return caf::make_counted<mem_ref<T>>(scratch_argument(std::move(arg)));
  }

private:
  template <typename T>
  mem_ref<T> global_argument(in<T> arg) {
    int size = static_cast<int>(arg.buffer.size());
    int access = IN;
    CUdeviceptr device_buffer;
    int bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuMemcpyHtoD(device_buffer, arg.buffer.data(), bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return {size, device_buffer, access};
  }

  template <typename T>
  mem_ref<T> global_argument(in_out<T> arg) {
    int size = static_cast<int>(arg.buffer.size());
    int access = IN_OUT;
    CUdeviceptr device_buffer;
    int bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuMemcpyHtoD(device_buffer, arg.buffer.data(), bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return {size, device_buffer, access};
  }

  template <typename T>
  mem_ref<T> scratch_argument(out<T> arg) {
    int size = static_cast<int>(arg.buffer.size());
    int access = OUT;
    CUdeviceptr device_buffer;
    int bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return {size, device_buffer, access};
  }

  CUdevice device_;
  CUcontext context_;
  CUstream stream_;
  char* name_;
  int id_;
  int streamId_;
  int contextId_;
};

} // namespace caf::cuda

