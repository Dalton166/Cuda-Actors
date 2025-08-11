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
#include "caf/cuda/platform.hpp"
#include "caf/cuda/arguments.hpp"
#include "caf/cuda/device.hpp"
#include <caf/send.hpp>
#include <caf/message.hpp>
#include "caf/cuda/mem_ref.hpp"

namespace caf::cuda {

//this class will always launch and schedule a kernel 
//execution and return a tuple of mem ref's
template <class Actor, class... Ts>
class base_command : public ref_counted {
public:
  template <typename... Us>
  base_command(caf::message msg,
               program_ptr program,
               nd_range dims,
               int id,
               Us&&... xs)
    : msg_(std::move(msg)),
      program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id(id) {
    dev_ = platform::create()->schedule(id);
    using_message = true; 
    static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
  }
  template <typename... Us>
  base_command(
               program_ptr program,
               nd_range dims,
               int id,
               Us&&... xs)
    : program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id(id),
      kernel_args(std::make_tuple(std::forward<Us>(xs)...))
       	{
    
	dev_ = platform::create()->schedule(id);
    static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
  }



  template <typename... Us>
  base_command(
               program_ptr program,
               nd_range dims,
               int id,
	       int device_number,
               Us&&... xs)
    : program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id(id),
      kernel_args(std::make_tuple(std::forward<Us>(xs)...))
       	{
    
	dev_ = platform::create()->schedule(id,device_number);
    static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
  }





  virtual ~base_command() = default;

  // Unpacks a caf message and calls launch_kernel_mem_ref and returns its result  
  virtual std::tuple<mem_ptr<raw_t<Ts>>...> base_enqueue() {
    
    //std::cout << "Unpacking message\n";
    // Step 1: Unpack message
     if (using_message) {
     auto unpacked = unpack_args(std::index_sequence_for<Ts...>{});
    
    //std::cout << "Launching kernel\n";
    // Step 2: Launch kernel via centralized utility
    CUfunction kernel = program_->get_kernel(dev_->getId());
    return dev_->launch_kernel_mem_ref(kernel, dims_, unpacked, actor_id);
     }
     else {
     //Launch kernel via centralized utility
    CUfunction kernel = program_->get_kernel(dev_->getId());
    return dev_->launch_kernel_mem_ref(kernel, dims_, kernel_args, actor_id);
     
     
     }


     
  }

  // intrusive_ptr ref counting friends
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
  caf::message msg_;
  std::tuple<Ts ...> kernel_args;
  bool using_message = false;

};

// intrusive_ptr reference counting
template <class Actor, class... Ts>
inline void intrusive_ptr_add_ref(base_command<Actor, Ts...>* ptr) {
  ptr->ref();
}

template <class Actor, class... Ts>
inline void intrusive_ptr_release(base_command<Actor, Ts...>* ptr) {
  ptr -> deref();
}

// This class really just returns an output buffer instead of a tuple of mem refs
// also handles mem ref cleanup for you 
template <class Actor, class... Ts>
class command : public base_command<Actor, Ts...> {
public:
  using base = base_command<Actor, Ts...>;

  template <typename... Us>
  command(caf::message msg,
          program_ptr program,
          nd_range dims,
          int id,
          Us&&... xs)
    : base(std::move(msg),
           std::move(program),
           std::move(dims),
           id,
           std::forward<Us>(xs)...) {}
 
  template <typename... Us>
  command(
          program_ptr program,
          nd_range dims,
          int id,
          Us&&... xs)
    : base(
           std::move(program),
           std::move(dims),
           id,
           std::forward<Us>(xs)...) {}


 template <typename... Us>
  command(
          program_ptr program,
          nd_range dims,
          int id,
	  int device_number,
          Us&&... xs)
    : base(
           std::move(program),
           std::move(dims),
           id,
           std::forward<Us>(xs)...) {}




  // Override enqueue to return collected output_buffers instead of mem_refs tuple
  std::vector<output_buffer> enqueue() {
    // Call base enqueue to get tuple of mem_refs
    auto mem_refs = base::base_enqueue();

    // Convert mem_refs to output_buffers via helper
    auto result = base::dev_ -> collect_output_buffers(mem_refs);
     for_each_tuple(mem_refs, [](auto& mem) {
      if (mem)
        mem->reset();
    });

     return result;
  }



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
