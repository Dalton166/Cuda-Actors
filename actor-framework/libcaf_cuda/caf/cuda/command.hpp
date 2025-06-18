#pragma once

#include <tuple>
#include <vector>
#include <functional>
#include <atomic>
#include <iostream>

#include <caf/abstract_actor.hpp>
#include <caf/actor_cast.hpp>
#include <caf/logger.hpp>
#include <caf/raise_error.hpp>
#include <caf/response_promise.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/detail/type_list.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/arguments.hpp"
#include "caf/cuda/opencl_err.hpp"
#include "caf/cuda/device.hpp"
//#include "caf/cuda/utility.hpp"

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
  // Constructor template to perfectly forward argument wrappers (in/out/in_out)
  template <typename... Us>
  command(caf::response_promise promise,
          program_ptr program,
          nd_range dims,
          Us&&... xs)
    : rp(std::move(promise)),
      program_(std::move(program)),
      dims_(dims),
      mem_refs(convert_data_to_args(std::forward<Us>(xs)...)) {
    static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
  }

  void enqueue() {
    launch_kernel(program_, dims_, mem_refs, program_->get_stream_id());

    // At this point kernel should be done, data in mem_refs ready to copy back and cleanup
    print_and_cleanup_outputs(mem_refs);
  }

  ~command() override = default;

private:
  // Convert argument wrappers to mem_ptr of raw types
  template <typename... Args>
  std::tuple<mem_ptr<raw_t<std::decay_t<Args>>>...>
  convert_data_to_args(Args&&... args) {
    int dev_id = program_->get_device_id();
    int ctx_id = program_->get_context_id();
    return std::make_tuple(makeArg(dev_id, ctx_id, std::forward<Args>(args))...);
  }

  // Helper to iterate over tuple elements
  template <typename Tuple, typename Func, size_t... Is>
  void for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(t)), ...);
  }

  template <typename... Is, typename Func>
  void for_each_tuple(std::tuple<Is...>& t, Func&& f) {
    for_each_tuple_impl(t, std::forward<Func>(f), std::index_sequence_for<Is...>{});
  }

  // Print output buffers and cleanup mem_refs
  void print_and_cleanup_outputs(std::tuple<mem_ptr<raw_t<Ts>>...>& mem_refs) {
    for_each_tuple(mem_refs, [](auto& mem) {
      if (!mem)
        return;

      const int access = mem->access();
      if (access == OUT || access == IN_OUT) {
        auto host_data = mem->copy_to_host();
        std::cout << "Output buffer (" << host_data.size() << "): ";
        for (const auto& val : host_data) {
          std::cout << val << " ";
        }
        std::cout << '\n';
      }

      mem->reset(); // Always clean up
    });
  }

  void launch_kernel(program_ptr program,
                     const caf::cuda::nd_range& range,
                     std::tuple<mem_ptr<raw_t<Ts>> ...> args,
                     int stream_id) {

    int device_id = program->get_device_id();
    int context_id = program->get_context_id();
    CUfunction kernel = program->get_kernel();
    device_ptr dev = program->get_device();
    dev->launch_kernel(kernel, range, args, stream_id, context_id);
  }

  program_ptr program_;
  caf::response_promise rp;
  nd_range dims_;
  std::tuple<mem_ptr<raw_t<Ts>>...> mem_refs;
  std::atomic<int> ref_count{0};
};

// Free functions for intrusive_ptr ref counting to enable ADL
template <class Actor, class... Ts>
inline void intrusive_ptr_add_ref(command<Actor, Ts...>* ptr) {
  ++(ptr->ref_count);
  std::cout << "intrusive_ptr_add_ref: ref_count=" << ptr->ref_count.load() << "\n";
}

template <class Actor, class... Ts>
inline void intrusive_ptr_release(command<Actor, Ts...>* ptr) {
  if (--(ptr->ref_count) == 0) {
    std::cout << "intrusive_ptr_release: deleting command\n";
    delete ptr;
  } else {
    std::cout << "intrusive_ptr_release: ref_count=" << ptr->ref_count.load() << "\n";
  }
}

// Helper makeArg overloads using program_->get_device()
  template <typename T>
  mem_ptr<T> makeArg(in<T> arg) {
    device_ptr dev = program_->get_device();
    return dev->make_arg(arg);
  }

  template <typename T>
  mem_ptr<T> makeArg(in_out<T> arg) {
    device_ptr dev = program_->get_device();
    return dev->make_arg(arg);
  }

  template <typename T>
  mem_ptr<T> makeArg(out<T> arg) {
    device_ptr dev = program_->get_device();
    return dev->make_arg(arg);
  }

  // fallback template for other types (if any)
  template <typename T>
  mem_ptr<T> makeArg(T&& arg) {
    device_ptr dev = program_->get_device();
    return dev->make_arg(std::forward<T>(arg));
  }

  // Now your convert_data_to_args becomes:
  template <typename... Args>
  auto convert_data_to_args(Args&&... args) {
    // Call the member makeArg overloads (no need for device_id/context_id params)
    return std::make_tuple(makeArg(std::forward<Args>(args))...);
  }


} // namespace caf::cuda

