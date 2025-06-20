
#include <iostream>

#include "caf/cuda/utility.hpp"
#include "caf/cuda/manager.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/command.hpp"

namespace caf::cuda {


template <class Actor, class... Ts>
command<Actor, Ts...>::command(caf::response_promise promise,
                              program_ptr program, nd_range dims,
                              Ts&&... xs)
  : rp(std::move(promise)),
    program_(program),
    dims_(dims),
    mem_refs(convert_data_to_args(std::forward<Ts>(xs)...)) {
}

template <class Actor, class... Ts>
void command<Actor, Ts...>::enqueue() {
  launch_kernel(program_, dims_, mem_refs, program_->get_stream_id());

  // print and cleanup outputs
  print_and_cleanup_outputs(mem_refs);
}

template <class Actor, class... Ts>
command<Actor, Ts...>::~command() = default;

template <class Actor, class... Ts>
void command<Actor, Ts...>::intrusive_ptr_add_ref(command* ptr) {
  ++(ptr->ref_count);
  std::cout << "intrusive_ptr_add_ref: ref_count=" << ptr->ref_count.load() << "\n";
}

template <class Actor, class... Ts>
void command<Actor, Ts...>::intrusive_ptr_release(command* ptr) {
  if (--(ptr->ref_count) == 0) {
    std::cout << "intrusive_ptr_release: deleting command\n";
    delete ptr;
  } else {
    std::cout << "intrusive_ptr_release: ref_count=" << ptr->ref_count.load() << "\n";
  }
}

template <class Actor, class... Ts>
template <typename... Args>
std::tuple<mem_ptr<std::decay_t<Args>>...>
command<Actor, Ts...>::convert_data_to_args(Args&&... args) {
  int dev_id = program_->get_device_id();
  int ctx_id = program_->get_context_id();
  return std::make_tuple(makeArg(dev_id, ctx_id, std::forward<Args>(args))...);
}

template <class Actor, class... Ts>
template <typename Tuple, typename Func, size_t... Is>
void command<Actor, Ts...>::for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>) {
  (f(std::get<Is>(t)), ...);
}

template <class Actor, class... Ts>
template <typename... Is, typename Func>
void command<Actor, Ts...>::for_each_tuple(std::tuple<Is...>& t, Func&& f) {
  for_each_tuple_impl(t, std::forward<Func>(f), std::index_sequence_for<Is...>{});
}

template <class Actor, class... Ts>
void command<Actor, Ts...>::print_and_cleanup_outputs(std::tuple<mem_ptr<Ts>...>& mem_refs) {
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

} // namespace caf::cuda

// Explicit template instantiations if you know Actor and Ts...

// Example for Actor=some_actor, Ts=int, float
// template class caf::cuda::command<some_actor, int, float>;
// Add your real instantiations here based on usage.

