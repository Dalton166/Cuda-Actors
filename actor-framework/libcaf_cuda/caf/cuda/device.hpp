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
    //CHECK_CUDA(cuCtxSetCurrent(context_));
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

 template <typename... Ts>
  auto launch_kernel(CUfunction kernel, const nd_range& range, std::tuple<Ts...> args, int stream_id, int context_id)
    -> std::vector<output_buffer> {
    std::lock_guard<std::mutex> lock(stream_mutex);

    try {
      CUstream stream = nullptr;
      CUcontext ctx = getContext(context_id);

      if (!ctx) throw std::runtime_error("Invalid context in launch_kernel");
      if (!kernel) throw std::runtime_error("Invalid kernel handle in launch_kernel");

      CHECK_CUDA(cuCtxPushCurrent(ctx));
      CUcontext current_ctx;
      CHECK_CUDA(cuCtxGetCurrent(&current_ctx));
      if (current_ctx != ctx) {
        throw std::runtime_error("Context mismatch");
      }

      std::cout << "launch_kernel: context=" << ctx << ", kernel=" << kernel << "\n";

      auto kernel_arg_vec = extract_kernel_args(args);
      void** kernel_args = kernel_arg_vec.data();

      for (size_t i = 0; i < kernel_arg_vec.size(); ++i) {
        std::cout << "launch_kernel: args[" << i << "]=" << *static_cast<CUdeviceptr*>(kernel_arg_vec[i]) << "\n";
      }

      CUresult result = cuLaunchKernel(
        kernel,
        range.getGridDimX(), range.getGridDimY(), range.getGridDimZ(),
        range.getBlockDimX(), range.getBlockDimY(), range.getBlockDimZ(),
        0, stream, kernel_args, nullptr
      );
      if (result != CUDA_SUCCESS) {
        const char* err_name;
        cuGetErrorName(result, &err_name);
        throw std::runtime_error("cuLaunchKernel failed: " + std::string(err_name ? err_name : "unknown error"));
      }

      CHECK_CUDA(cuCtxSynchronize());

      std::vector<output_buffer> outputs;
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

      CHECK_CUDA(cuCtxPopCurrent(nullptr));

      for (void* arg : kernel_arg_vec) {
        delete static_cast<CUdeviceptr*>(arg);
      }

      return outputs;
    } catch (const std::exception& e) {
      std::cout << "launch_kernel failed: " << e.what() << "\n";
      throw;
    }
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
        std::vector<void*> args(sizeof...(Is));
        // Allocate CUdeviceptr objects on the heap to ensure lifetime
        CUdeviceptr* device_ptrs[] = { new CUdeviceptr(std::get<Is>(t)->mem())... };
        for (size_t i = 0; i < sizeof...(Is); ++i) {
            args[i] = device_ptrs[i];
            std::cout << "extract_kernel_args: device_ptrs[" << i << "]=" << *static_cast<CUdeviceptr*>(args[i]) << "\n";
        }
        return args;
    }

    template <typename... Ts>
    std::vector<void*> extract_kernel_args(const std::tuple<Ts...>& t) {
        return extract_kernel_args_impl(t, std::index_sequence_for<Ts...>{});
    }
};


} // namespace caf::cuda

