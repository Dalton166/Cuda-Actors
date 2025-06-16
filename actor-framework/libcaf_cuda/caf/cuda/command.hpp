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
#include <caf/detail/type_list.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/arguments.hpp"
#include "caf/cuda/opencl_err.hpp"

namespace caf::cuda {

template <class Actor, class... Ts>
class command : public ref_counted {
public:
  using result_types = caf::detail::type_list<Ts...>;

  command(caf::response_promise,
          caf::strong_actor_ptr,
          std::vector<void*>,
          std::vector<void*>,
          std::vector<void*>,
          std::vector<size_t>,
          caf::message,
          std::tuple<Ts...>,
          nd_range) {
    throw std::runtime_error("CUDA support disabled: command ctor");
  }

  void enqueue() {
    throw std::runtime_error("CUDA support disabled: command::enqueue()");
  }

  ~command() override = default;

 template <typename... Args>
std::tuple<mem_ptr...> convert_data_to_args(Args&&... args) {
  int dev_id = program_->get_device_id();
  int ctx_id = program_->get_context_id();
  return std::make_tuple(makeArg(dev_id, ctx_id, std::forward<Args>(args))...);
}

};

} // namespace caf::cuda
