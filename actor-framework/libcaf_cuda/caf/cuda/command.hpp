#pragma once

#include <tuple>
#include <vector>
#include <iostream>

#include <caf/abstract_actor.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/message.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/mem_ref.hpp"
#include "caf/cuda/device.hpp"

namespace caf::cuda {

// -----------------------------------------------------------------------------
// Base command: schedules kernel, returns tuple of mem_ptrs
// -----------------------------------------------------------------------------
template <class Actor, class... Ts>
class base_command : public ref_counted {
public:
  // ---------------------------------------------------------------------------
  // Constructor: variadic arguments first, then optional shared_memory/device_number
  // ---------------------------------------------------------------------------
  template <typename... Us>
  base_command(program_ptr program,
               nd_range dims,
               int actor_id,
               Us&&... xs,
               int shared_memory_ = 0,
               int device_number = -1)
    : program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id(actor_id),
      kernel_args(std::make_tuple(std::forward<Us>(xs)...)),
      shared_memory(shared_memory_)
  {
      if (device_number == -1)
          dev_ = platform::create()->schedule(actor_id);
      else
          dev_ = platform::create()->schedule(actor_id, device_number);
  }

  virtual ~base_command() = default;

  // ---------------------------------------------------------------------------
  // Launch kernel and return tuple of mem_ptrs
  // ---------------------------------------------------------------------------
  virtual std::tuple<mem_ptr<raw_t<Ts>>...> base_enqueue() {
      CUfunction kernel = program_->get_kernel(dev_->getId());

      if (using_message) {
          auto unpacked = unpack_args(std::index_sequence_for<Ts...>{});
          return dev_->launch_kernel_mem_ref(kernel, dims_, unpacked, actor_id, shared_memory);
      } else {
          return dev_->launch_kernel_mem_ref(kernel, dims_, kernel_args, actor_id, shared_memory);
      }
  }

  // ---------------------------------------------------------------------------
  // CAF intrusive_ptr helpers
  // ---------------------------------------------------------------------------
  template <class A, class... S>
  friend void intrusive_ptr_add_ref(base_command<A, S...>* ptr);

  template <class A, class... S>
  friend void intrusive_ptr_release(base_command<A, S...>* ptr);

  template <size_t... Is>
  auto unpack_args(std::index_sequence<Is...>) {
      return std::make_tuple(msg_.get_as<Ts>(Is)...);
  }

protected:
  program_ptr program_;
  nd_range dims_;
  int actor_id;
  device_ptr dev_;
  int shared_memory;
  caf::message msg_;
  std::tuple<Ts...> kernel_args;
  bool using_message = false;
};

// intrusive_ptr helpers
template <class Actor, class... Ts>
inline void intrusive_ptr_add_ref(base_command<Actor, Ts...>* ptr) { ptr->ref(); }

template <class Actor, class... Ts>
inline void intrusive_ptr_release(base_command<Actor, Ts...>* ptr) { ptr->deref(); }

// -----------------------------------------------------------------------------
// Command: returns vector of output_buffer
// -----------------------------------------------------------------------------
template <class Actor, class... Ts>
class command : public base_command<Actor, Ts...> {
public:
  using base = base_command<Actor, Ts...>;

  template <typename... Us>
  command(program_ptr program,
          nd_range dims,
          int actor_id,
          Us&&... xs,
          int shared_memory_ = 0,
          int device_number = -1)
    : base(std::move(program),
           std::move(dims),
           actor_id,
           std::forward<Us>(xs)...,
           shared_memory_,
           device_number)
  {}

  // ---------------------------------------------------------------------------
  // Enqueue returns vector of output_buffer
  // ---------------------------------------------------------------------------
  std::vector<output_buffer> enqueue() {
      auto mem_refs = base::base_enqueue();
      return base::dev_->collect_output_buffers(mem_refs);
  }

  // ---------------------------------------------------------------------------
  // Tuple iteration helpers
  // ---------------------------------------------------------------------------
  template <typename Tuple, typename Func, size_t... Is>
  void for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>) {
      (f(std::get<Is>(t)), ...);
  }

  template <typename... Is, typename Func>
  void for_each_tuple(std::tuple<Is...>& t, Func&& f) {
      for_each_tuple_impl(t, std::forward<Func>(f), std::index_sequence_for<Is...>{});
  }
};

} // namespace caf::cuda
