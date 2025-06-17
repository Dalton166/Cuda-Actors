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

#include <tuple>
#include <vector>
#include <functional>
#include <atomic>

#include <caf/response_promise.hpp>
#include <caf/intrusive_ptr.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/nd_range.hpp"

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
  command(caf::response_promise promise,
          program_ptr program, nd_range dims,
          Ts&&... xs);

  void enqueue();

  ~command() override;

  // intrusive_ptr helpers
  void intrusive_ptr_add_ref(command* ptr);
  void intrusive_ptr_release(command* ptr);

private:
  template <typename... Args>
  std::tuple<mem_ptr<std::decay_t<Args>>...>
  convert_data_to_args(Args&&... args);

  template <typename Tuple, typename Func, size_t... Is>
  void for_each_tuple_impl(Tuple& t, Func&& f, std::index_sequence<Is...>);

  template <typename... Is, typename Func>
  void for_each_tuple(std::tuple<Is...>& t, Func&& f);

  void print_and_cleanup_outputs(std::tuple<mem_ptr<Ts>...>& mem_refs);

  program_ptr program_;
  caf::response_promise rp;
  int flags = 0;
  std::tuple<mem_ptr<std::decay_t<Ts>>...> mem_refs;
  nd_range dims_;
  std::atomic<int> ref_count{0};
};

} // namespace caf::cuda
