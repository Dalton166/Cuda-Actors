#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>

#include <cuda.h>

#include "caf/cuda/global.hpp"
//#include "caf/cuda/mem_ref.hpp" introduces a circular dependency

namespace caf::cuda {

class device : public caf::ref_counted {
public:
  using device_ptr = caf::intrusive_ptr<device>;

  device(CUdevice device, CUcontext context, const char* name, int id)
    : device_(device),
      context_(context),
      id_(id),
      streamId_(0),
      contextId_(0) {
    CHECK_CUDA(cuStreamCreate(&stream_, 0));
    name_ = name;
  }

  ~device() {
    if (stream_ != nullptr) {
      cuStreamDestroy(stream_);
    }
  }

  // Disable copy
  device(const device&) = delete;
  device& operator=(const device&) = delete;

  // Move allowed
  device(device&&) noexcept = default;
  device& operator=(device&&) noexcept = default;

  const char* name() const { return name_; }

  CUdevice getDevice() const { return device_; }
  CUcontext getContext() const { return context_; }
  int getId() const { return id_; }
  int getStreamId() const { return streamId_; }
  int getContextId() const { return contextId_; }
  CUstream getStream() const { return stream_; }

  // Example method to create a mem_ref for an input buffer
  template <typename T>
  mem_ref<T> global_argument(in<T> arg) {
    size_t size = arg.buffer.size();
    int access = IN;
    CUdeviceptr device_buffer = 0;
    size_t bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuMemcpyHtoD(device_buffer, arg.buffer.data(), bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    // Pass device and context ids (default 0 for now)
    return mem_ref<T>{size, device_buffer, access, id_, contextId_};
  }

  // For in_out buffers
  template <typename T>
  mem_ref<T> global_argument(in_out<T> arg) {
    size_t size = arg.buffer.size();
    int access = IN_OUT;
    CUdeviceptr device_buffer = 0;
    size_t bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuMemcpyHtoD(device_buffer, arg.buffer.data(), bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return mem_ref<T>{size, device_buffer, access, id_, contextId_};
  }

  // For scratch (output) buffers with no initial copy
  template <typename T>
  mem_ref<T> scratch_argument(in_out<T> arg) {
    size_t size = arg.buffer.size();
    int access = OUT;
    CUdeviceptr device_buffer = 0;
    size_t bytes = size * sizeof(T);

    CHECK_CUDA(cuCtxPushCurrent(context_));
    CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
    CHECK_CUDA(cuCtxPopCurrent(nullptr));

    return mem_ref<T>{size, device_buffer, access, id_, contextId_};
  }

private:
  int id_;
  int streamId_;
  int contextId_;
  CUcontext context_;
  const char* name_;
  CUdevice device_;
  CUstream stream_;
};

} // namespace caf::cuda

