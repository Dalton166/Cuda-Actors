/******************************************************************************
 *                       ____    _    _____                                   *
 *                      / ___|  / \  |  ___|    C++                           *
 *                     | |     / _ \ | |_       Actor                         *
 *                     | |___ / ___ \|  _|      Framework                     *
 *                      \____/_/   \_|_|                                      *
 *                                                                            *
 * Copyright (C) 2011 - 2016                                                  *
 *                                                                            *
 * Distributed under the terms and conditions of the BSD 3-Clause License or  *
 * (at yourਸ  your option) under the terms and conditions of the Boost Software      *
 * License 1.0. See accompanying files LICENSE and LICENSE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#pragma once

#include <tuple>
#include <vector>
#include <functional>

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

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
//  using result_types = caf::detail::type_list<Ts...>;

 command(caf::response_promise promise,
          program_ptr program, nd_range dims,
          Ts&&... xs)
    : rp(std::move(promise)),
      program_(program),
      dims_(dims),
      mem_refs(convert_data_to_args(std::forward<Ts>(xs)...)) {
  }


  void enqueue() {
  
	  launch_kernel(program_,dims_,mem_refs,program_ -> get_stream_id());

	  //at this point for now the kernel should be done running and we should have our data in mem_ref ready to be transfer back to the device as well as cleanup 
	 print_and_cleanup_outputs(mem_refs);
  
  }

  //TODO if command is ever deconstructed they should free all the mem_refs they have on them 
  ~command() override = default;


  // Implementation of intrusive_ptr_add_ref increments ref count
void intrusive_ptr_add_ref(command* ptr) {
  ++(ptr->ref_count);
  std::cout << "intrusive_ptr_add_ref: ref_count=" << ptr->ref_count.load() << "\n";
}

// Implementation of intrusive_ptr_release decrements ref count and deletes if zero
void intrusive_ptr_release(command* ptr) {
  if (--(ptr->ref_count) == 0) {
    std::cout << "intrusive_ptr_release: deleting command\n";
    delete ptr;
  } else {
    std::cout << "intrusive_ptr_release: ref_count=" << ptr->ref_count.load() << "\n";
  }
}


private:
  template <typename... Args>
  std::tuple<mem_ptr<std::decay_t<Args>>...>
  convert_data_to_args(Args&&... args) {
    int dev_id = program_->get_device_id();
    int ctx_id = program_->get_context_id();
    return std::make_tuple(makeArg(dev_id, ctx_id, std::forward<Args>(args))...);
  }

  template <typename Tuple, typename Func, size_t... Is>
void for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>) {
  (f(std::get<Is>(t)), ...);
}

template <typename... Is, typename Func>
void for_each_tuple(std::tuple<Is...>& t, Func&& f) {
  for_each_tuple_impl(t, std::forward<Func>(f), std::index_sequence_for<Is...>{});
}

// Iterate, print host data if access is OUT or IN_OUT, and cleanup
void print_and_cleanup_outputs(std::tuple<mem_ptr<Ts>...>& mem_refs) {
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

  program_ptr program_;
  caf::response_promise rp;
  int flags = 0;
  std::tuple<mem_ptr<std::decay_t<Ts>>...> mem_refs;
  nd_range dims_;
 std::atomic<int> ref_count{0};
};

} // namespace caf::cuda
