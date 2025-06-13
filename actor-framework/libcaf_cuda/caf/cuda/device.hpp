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


template <typename T>
mem_ptr<T> make_arg(in<T> arg) {
  return caf::intrusive_ptr<mem_ref<T>>{new mem_ref<T>(global_argument(std::move(arg)))};
}

template <typename T>
mem_ptr<T> make_arg(in_out<T> arg) {
  return caf::intrusive_ptr<mem_ref<T>>{new mem_ref<T>(global_argument(std::move(arg)))};
}

template <typename T>
mem_ptr<T> make_arg(out<T> arg) {
  return caf::intrusive_ptr<mem_ref<T>>{new mem_ref<T>(scratch_argument(std::move(arg)))};
}

template <typename T>
void launch_kernel(CUfunction kernel,int gridDim, int blockDim, std::tuple<mem_ptr<T> args) {


	auto kernel_args = extract_kernel_args(args);



}





private:
  CUdevice device_;
  CUcontext context_;
  int id_;
  int streamId_;
  int contextId_;
  const char* name_;
  CUstream stream_;


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

    template <typename Tuple, std::size_t... Is>
std::vector<void*> extract_kernel_args_impl(const Tuple& t, std::index_sequence<Is...>) {
  // Store CUdeviceptrs in a temporary array so we can take their addresses
  static_assert(sizeof...(Is) > 0, "At least one kernel argument is required.");
  CUdeviceptr device_ptrs[] = { std::get<Is>(t)->mem()... };

  std::vector<void*> args(sizeof...(Is));
  for (size_t i = 0; i < sizeof...(Is); ++i)
    args[i] = &device_ptrs[i];

  return args;
}

// Public interface
template <typename... Ts>
std::vector<void*> extract_kernel_args(const std::tuple<mem_ptr<Ts>...>& args_tuple) {
  return extract_kernel_args_impl(args_tuple, std::index_sequence_for<Ts...>{});
}


};


} // namespace caf::cuda

