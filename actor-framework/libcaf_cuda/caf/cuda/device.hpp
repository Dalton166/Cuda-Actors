#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>

#include <cuda.h>

#include "caf/cuda/global.hpp"
//#include "caf/cuda/mem_ref.hpp" introduces a circular dependency
#include <mutex> //TODO delete this later 


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
  CUstream getStream(int id) { 
	  
	  //for right now this does the same as getStream
	  //however it is likely that this project will expand into
	  //using multiple streams so use the id variant for now 
	  return stream_; 
  }

  CUcontext getContext(int id) { 
	  
	  //for right now this does the same as getContext
	  //however I can't tell if this project will end up using multiple contexts per device so use this for now  
	  return context_; 
  }


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
void launch_kernel(CUfunction kernel,
                   const caf::cuda::nd_range& range,
                   std::tuple<mem_ptr<T>> args,
                   int stream_id,
                   int context_id) {
    std::lock_guard<std::mutex> lock(stream_mutex); // Automatically released at end of scope

    //std::cout << "Hello my name is carlos\n";

    CUstream stream = getStream(stream_id);
    CUcontext ctx = getContext(context_id);

    // Validate resources
    if (!ctx) throw std::runtime_error("Invalid context in launch_kernel");
    if (!stream) throw std::runtime_error("Invalid stream in launch_kernel");
    if (!kernel) throw std::runtime_error("Invalid kernel handle in launch_kernel");

    // Push context to this thread
    CHECK_CUDA(cuCtxPushCurrent(ctx));

    // Extract kernel arguments (assumed to return void** suitable for cuLaunchKernel)
    auto kernel_arg_vec = extract_kernel_args(args);
    void** kernel_args = kernel_arg_vec.data();

    // Launch the kernel using nd_range for dimensions
    CHECK_CUDA(cuLaunchKernel(
        kernel,
        range.getGridDimX(), range.getGridDimY(), range.getGridDimZ(),   // Grid dimensions
        range.getBlockDimX(), range.getBlockDimY(), range.getBlockDimZ(),// Block dimensions
        0,                                                               // Shared memory size
        0,                                                          // CUDA stream
        kernel_args,                                                     // Kernel arguments
        nullptr                                                          // Extra options (usually null)
    ));

    //synchronize stream TODO use caf promises as a way to remove this, or in general find a way to get ride of this 
    CHECK_CUDA(cuStreamSynchronize(stream));
    // Pop context
    CHECK_CUDA(cuCtxPopCurrent(nullptr));
}




private:
  CUdevice device_;
  CUcontext context_;
  int id_;
  int streamId_;
  int contextId_;
  const char* name_;
  CUstream stream_;
  std::mutex stream_mutex;

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
  mem_ref<T> scratch_argument(out<T> arg) {
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

