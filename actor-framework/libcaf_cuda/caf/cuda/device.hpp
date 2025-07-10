#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>

#include <caf/intrusive_ptr.hpp>
#include <caf/ref_counted.hpp>

#include <cuda.h>

#include "caf/cuda/global.hpp"
#include "caf/cuda/types.hpp"
//#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/StreamPool.hpp" // your new stream pool header

namespace caf::cuda {

class device : public caf::ref_counted {
public:
  using device_ptr = caf::intrusive_ptr<device>;

  device(CUdevice device, CUcontext context, const char* name, int id, size_t stream_pool_size = 32)
    : device_(device),
      context_(context),
      id_(id),
      name_(name),
      stream_table_(context,stream_pool_size) {
    // No default stream creation here, streams managed per actor
  }

  ~device() {
     //stream table needs to be deleted first 
     //since once device is destroyed, stream table cant clean up
     //delete &stream_table_; 
     check(cuCtxDestroy(context_), "cuCtxDestroy");
  }

  device(const device&) = delete;
  device& operator=(const device&) = delete;

  device(device&&) noexcept = default;
  device& operator=(device&&) noexcept = default;

  const char* name() const { return name_; }
  CUdevice getDevice() const { return device_; }
  CUcontext getContext() const { return context_; }
  int getId() const { return id_; }

  int getStreamId() const { return 0; }
  int getContextId() const { return 0; }
  CUcontext getContext(int id) {

	  //for right now this does the same as getContext
	  //however I can't tell if this project will end up using multiple contexts per device so use this for now
	  return context_;
  }



  CUstream get_stream_for_actor(int actor_id) {
    return stream_table_.get_stream(actor_id);
  }

  void release_stream_for_actor(int actor_id) {
    stream_table_.release_stream(actor_id);
  }

  template <typename T>
  mem_ptr<T> make_arg(in<T> arg, int actor_id) {
    return global_argument(arg, actor_id, IN);
  }

  template <typename T>
  mem_ptr<T> make_arg(in_out<T> arg, int actor_id) {
    return global_argument(arg, actor_id, IN_OUT);
  }

  template <typename T>
  mem_ptr<T> make_arg(out<T> arg, int actor_id) {
    return scratch_argument(arg, actor_id, OUT);
  }

  template <typename... Ts>
std::vector<output_buffer> launch_kernel(CUfunction kernel,
                                         const nd_range& range,
                                         std::tuple<Ts...> args,
                                         int actor_id) {
  CUstream stream = get_stream_for_actor(actor_id);
  CUcontext ctx = getContext();

  CHECK_CUDA(cuCtxPushCurrent(ctx));

  auto kernel_arg_vec = extract_kernel_args(args);
  void** kernel_args = kernel_arg_vec.data();

  CUresult result = cuLaunchKernel(
    kernel,
    range.getGridDimX(), range.getGridDimY(), range.getGridDimZ(),
    range.getBlockDimX(), range.getBlockDimY(), range.getBlockDimZ(),
    0, stream, kernel_args, nullptr);

  if (result != CUDA_SUCCESS) {
    const char* err_name = nullptr;
    cuGetErrorName(result, &err_name);
    throw std::runtime_error(std::string("cuLaunchKernel failed: ") + (err_name ? err_name : "unknown error"));
  }

  // Synchronize only the actor's stream to ensure kernel completion
  CHECK_CUDA(cuStreamSynchronize(stream));

  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  std::vector<output_buffer> outputs;

  // Collect outputs from args that are out or in_out
  auto collect_outputs = [&outputs](auto&& arg) {
    if (arg && (arg->access() == OUT || arg->access() == IN_OUT)) {
      using T = typename std::decay_t<decltype(*arg)>::value_type;
      if constexpr (std::is_same_v<T, char>) {
        outputs.emplace_back(output_buffer{buffer_variant{arg->copy_to_host()}});
      } else if constexpr (std::is_same_v<T, int>) {
        outputs.emplace_back(output_buffer{buffer_variant{arg->copy_to_host()}});
      } else if constexpr (std::is_same_v<T, float>) {
        outputs.emplace_back(output_buffer{buffer_variant{arg->copy_to_host()}});
      } else if constexpr (std::is_same_v<T, double>) {
        outputs.emplace_back(output_buffer{buffer_variant{arg->copy_to_host()}});
      } else {
        throw std::runtime_error("Unsupported output type: " + std::string(typeid(T).name()));
      }
    }
  };

  std::apply([&](auto&&... arg) { (collect_outputs(arg), ...); }, args);

  // Clean up kernel argument pointers allocated in extract_kernel_args
  for (void* arg_ptr : kernel_arg_vec) {
    delete static_cast<CUdeviceptr*>(arg_ptr);
  }

  return outputs;
}

template <typename... Ts>
  std::vector<void*> extract_kernel_args(const std::tuple<Ts...>& t) {
    return extract_kernel_args_impl(t, std::index_sequence_for<Ts...>{});
  }

private:
  CUdevice device_;
  CUcontext context_;
  int id_;
  const char* name_;

  DeviceStreamTable stream_table_;

  std::mutex stream_mutex_;
template <typename T>
mem_ptr<T> global_argument(const in<T>& arg, int actor_id, int access) {
  CUstream stream = get_stream_for_actor(actor_id);

  if (arg.is_scalar()) {
    // ✅ Correct: scalar constructor
    return caf::intrusive_ptr<mem_ref<T>>(
      new mem_ref<T>(arg.getscalar(), access, id_, 0, stream)
    );
  }

  // Buffer path
  size_t size = arg.size();
  CUdeviceptr device_buffer = 0;
  size_t bytes = size * sizeof(T);

  CUcontext ctx = getContext();
  CHECK_CUDA(cuCtxPushCurrent(ctx));
  CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
  CHECK_CUDA(cuMemcpyHtoDAsync(device_buffer, arg.data(), bytes, stream));
  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  return caf::intrusive_ptr<mem_ref<T>>(
    new mem_ref<T>(size, device_buffer, access, id_, 0, stream)
  );
}



template <typename T>
mem_ptr<T> global_argument(const in_out<T>& arg, int actor_id, int access) {
  CUstream stream = get_stream_for_actor(actor_id);

  if (arg.is_scalar()) {
    // ✅ Correct: scalar constructor
    return caf::intrusive_ptr<mem_ref<T>>(
      new mem_ref<T>(arg.getscalar(), access, id_, 0, stream)
    );
  }

  // Buffer path
  size_t size = arg.size();
  CUdeviceptr device_buffer = 0;
  size_t bytes = size * sizeof(T);

  CUcontext ctx = getContext();
  CHECK_CUDA(cuCtxPushCurrent(ctx));
  CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
  CHECK_CUDA(cuMemcpyHtoDAsync(device_buffer, arg.data(), bytes, stream));
  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  return caf::intrusive_ptr<mem_ref<T>>(
    new mem_ref<T>(size, device_buffer, access, id_, 0, stream)
  );
}

template <typename T>
mem_ptr<T> scratch_argument(const out<T>& arg, int actor_id, int access) {
  size_t size = arg.is_scalar() ? 1 : arg.size();
  CUdeviceptr device_buffer = 0;
  size_t bytes = size * sizeof(T);

  CUcontext ctx = getContext();
  CHECK_CUDA(cuCtxPushCurrent(ctx));

  CUstream stream = get_stream_for_actor(actor_id);

  CHECK_CUDA(cuMemAlloc(&device_buffer, bytes));
  // no copy needed for output buffers

  CHECK_CUDA(cuCtxPopCurrent(nullptr));

  return caf::intrusive_ptr<mem_ref<T>>(new mem_ref<T>(size, device_buffer, access, id_, 0, stream));
}





  // inside device.hpp
template <typename Tuple, std::size_t... Is>
std::vector<void*> extract_kernel_args_impl(const Tuple& t,
                                            std::index_sequence<Is...>) {
  std::vector<void*> args(sizeof...(Is));
  size_t i = 0;
  (([&] {
     auto ptr = std::get<Is>(t);         // mem_ptr<T>
     if (ptr->is_scalar()) {
       // pass address of the host‐side scalar
       args[i++] = const_cast<void*>(
         static_cast<const void*>(ptr->host_scalar_ptr())
       );
     } else {
       // copy device pointer into its own CUdeviceptr slot
       CUdeviceptr* slot = new CUdeviceptr(ptr->mem());
       args[i++] = slot;
     }
   }()), ...);
  return args;
}
  
};

} // namespace caf::cuda

