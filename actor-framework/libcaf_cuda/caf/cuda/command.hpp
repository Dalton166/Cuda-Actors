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

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
  template <typename... Us>
  command(caf::message msg,
        caf::actor self,
        program_ptr program,
        nd_range dims,
        int id, 
        Us&&... xs)
  : msg_(msg),
    program_(program),
    dims_(dims),
    actor_id(id)
    { 
  dev_ = platform::create()->schedule(id);
  static_assert(sizeof...(Us) == sizeof...(Ts), "Argument count mismatch");
}


 ~command() = default;


  //unpacks a caf message and calls launch_kernel_mem_ref and returns its result  
 std::tuple<mem_ref<raw_t<Ts>>...> enqueue() {
  // Step 1: Unpack message
  auto unpacked = msg_.template get_as<std::tuple<Ts...>>(0);

  // Step 2: Launch kernel via centralized utility
  CUfunction kernel = program_->get_kernel(dev_->getId());
  return dev_->launch_kernel_mem_ref(kernel, dims_, unpacked, actor_id);
}



  template <class A, class... S>
  friend void intrusive_ptr_add_ref(command<A, S...>* ptr);

  template <class A, class... S>
  friend void intrusive_ptr_release(command<A, S...>* ptr);

private:
  program_ptr program_;
  nd_range dims_;
  int actor_id;
  device_ptr dev_;
  caf::message msg_;

   auto launch_kernel(program_ptr program,
                     const nd_range& range,
                     std::tuple<mem_ptr<raw_t<Ts>>...> args,
                     int actor_id) -> std::vector<output_buffer> {
    CUfunction kernel = program->get_kernel(dev_->getId());
    return dev_->launch_kernel(kernel, range, args, actor_id);
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
