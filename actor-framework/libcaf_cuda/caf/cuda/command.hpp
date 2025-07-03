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
#include <caf/send.hpp>

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
  template <typename... Us>
  command(caf::response_promise promise,
          caf::actor self,
          program_ptr program,
          nd_range dims,
          int id,
          Us&&... xs)
    : rp(std::move(promise)),
      self_(std::move(self)),
      program_(std::move(program)),
      dims_(dims),
      actor_id(id),
      mem_refs(convert_data_to_args(std::forward<Us>(xs)...)) {
    static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
  }

  ~command() = default;

  void enqueue() {
    auto outputs = launch_kernel(program_, dims_, mem_refs, actor_id);
    rp.deliver(std::move(outputs));

    for_each_tuple(mem_refs, [](auto& mem) {
      if (mem)
        mem->reset();
    });

    anon_send(self_, kernel_done_atom_v);
  }

  template <class A, class... S>
  friend void intrusive_ptr_add_ref(command<A, S...>* ptr);

  template <class A, class... S>
  friend void intrusive_ptr_release(command<A, S...>* ptr);

private:
  program_ptr program_;
  caf::response_promise rp;
  caf::actor self_;
  nd_range dims_;
  std::tuple<mem_ptr<raw_t<Ts>>...> mem_refs;
  std::atomic<int> ref_count{0};
  int actor_id;

  template <typename T>
  mem_ptr<T> makeArg(in<T> arg) {
    return program_->get_device()->make_arg(arg, actor_id);
  }

  template <typename T>
  mem_ptr<T> makeArg(out<T> arg) {
    return program_->get_device()->make_arg(arg, actor_id);
  }

  template <typename T>
  mem_ptr<T> makeArg(in_out<T> arg) {
    return program_->get_device()->make_arg(arg, actor_id);
  }

  // Fallback for raw types
  template <typename T>
  mem_ptr<T> makeArg(T&& val) {
    return program_->get_device()->make_arg(std::forward<T>(val), actor_id);
  }

  template <typename... Args>
  auto convert_data_to_args(Args&&... args) {
    return std::make_tuple(makeArg(std::forward<Args>(args))...);
  }

  void print_and_cleanup_outputs(std::tuple<mem_ptr<raw_t<Ts>>...>& mem_refs) {
    for_each_tuple(mem_refs, [](auto& mem) {
      if (!mem) return;
      if (mem->access() == OUT || mem->access() == IN_OUT) {
        auto host_data = mem->copy_to_host();
        std::cout << "Output buffer (" << host_data.size() << "): ";
        for (const auto& val : host_data)
          std::cout << val << " ";
        std::cout << '\n';
      }
      mem->reset();
    });
  }

  template <typename Tuple, typename Func, size_t... Is>
  void for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>) {
    (f(std::get<Is>(t)), ...);
  }

  template <typename... Is, typename Func>
  void for_each_tuple(std::tuple<Is...>& t, Func&& f) {
    for_each_tuple_impl(t, std::forward<Func>(f), std::index_sequence_for<Is...>{});
  }

  auto launch_kernel(program_ptr program,
                     const caf::cuda::nd_range& range,
                     std::tuple<mem_ptr<raw_t<Ts>>...> args,
                     int actor_id)
    -> std::vector<output_buffer> {
    CUfunction kernel = program->get_kernel();
    device_ptr dev = program->get_device();
    return dev->launch_kernel(kernel, range, args, actor_id);
  }
};

// intrusive_ptr reference counting
template <class Actor, class... Ts>
inline void intrusive_ptr_add_ref(command<Actor, Ts...>* ptr) {
  ++(ptr->ref_count);
}

template <class Actor, class... Ts>
inline void intrusive_ptr_release(command<Actor, Ts...>* ptr) {
  if (--(ptr->ref_count) == 0)
    delete ptr;
}

} // namespace caf::cuda

