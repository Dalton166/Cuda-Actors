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
 * (at your option) under the terms and conditions of the Boost Software      *
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

#include "caf/opencl/global.hpp"
#include "caf/opencl/nd_range.hpp"
#include "caf/opencl/arguments.hpp"
#include "caf/opencl/opencl_err.hpp"

namespace caf::opencl {

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
    throw std::runtime_error("OpenCL support disabled: command ctor");
  }

  void enqueue() {
    throw std::runtime_error("OpenCL support disabled: command::enqueue()");
  }

  ~command() override = default;
};

} // namespace caf::opencl
