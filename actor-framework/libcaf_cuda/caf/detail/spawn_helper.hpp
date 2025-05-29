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


namespace caf {
namespace detail {

template <bool PassConfig, class... Ts>
struct cuda_spawn_helper {
  using impl = cuda::actor_facade<PassConfig, Ts...>;
 
  
 /*as for right now, input and output mapping is disabled
  * will eventually/likely have to reinstall this later on
  * as a consquence all of these operators are disabled as well
 // using map_in_fun = typename impl::input_mapping;
 // using map_out_fun = typename impl::output_mapping;

  actor operator()(actor_system& sys, const opencl::program_ptr& p,
                   const char* fn, const opencl::nd_range& range,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(sys, p, fn, range,
                                          map_in_fun{}, map_out_fun{},
                                          std::forward<Ts>(xs)...));
  }

  actor operator()(actor_system& sys, const opencl::program_ptr& p,
                   const char* fn, const opencl::nd_range& range,
                   map_in_fun map_input, Ts&&... xs) const {
    return actor_cast<actor>(impl::create(sys, p, fn, range,
                                          std::move(map_input),
                                          map_out_fun{},
                                          std::forward<Ts>(xs)...));
  }

  actor operator()(actor_system& sys, const opencl::program_ptr& p,
                   const char* fn, const opencl::nd_range& range,
                   map_in_fun map_input, map_out_fun map_output,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(sys, p, fn, range,
                                          std::move(map_input),
					  std::move(map_output),
                                          std::forward<Ts>(xs)...));
  }

  */

  /*Literally just a test to see that actor_facade can still spawn
   * takes no arguments and returns an actor facade that will just say
   * hello world
   */



  /*
   * Note that you can only move the cfg file
   * so if we need to reuse the config 
   * we must spawn in another using actor_config cfg{} later on if needed
   */
  actor operator()(
		  actor_system &sys,
		  actor_config&& cfg,
                   Ts&&... xs) const {
    return actor_cast<actor>(impl::create(
			    sys,
			    std::move(cfg),              
			    std::forward<Ts>(xs)...));
  }





};

} // namespace detail
} // namespace caf

