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

#include <functional>

#include "caf/actor.hpp"
#include "caf/actor_cast.hpp"
#include "caf/actor_system.hpp"
#include "caf/actor_system_config.hpp"
#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/behavior.hpp"
#include <type_traits>

namespace caf {
namespace detail {

template <bool PassConfig, class... Ts>
struct cuda_spawn_helper {
  using impl = cuda::actor_facade<PassConfig, std::decay_t<Ts>...>;
 
  //this operator should spawn in a facade with a program
  actor operator()(
		  actor_system * sys,
		  actor_config&& cfg,
		   caf::cuda::program_ptr prog,
		   caf::cuda::nd_range dims,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(
			    sys,
			    std::move(cfg),              
			    prog,
			    dims,
			    std::forward<Ts>(xs)...));
  }

   //spawns in an actor facade given a behavior
   actor operator()(
		  actor_system * sys,
		  actor_config&& cfg,
		  behavior_ptr behavior,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(
			    sys,
			    std::move(cfg),              
			    behavior,
			    std::forward<Ts>(xs)...));
  }




};

} // namespace detail
} // namespace caf

