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
 ******************************************************************************/

#include <map>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>

#include "caf/cuda/manager.hpp"
#include "caf/cuda/program.hpp"


/*
using namespace std;

namespace caf::cuda {

program::program(detail::raw_context_ptr context,
                 detail::raw_command_queue_ptr queue,
                 detail::raw_program_ptr prog,
                 std::map<std::string, detail::raw_kernel_ptr> available_kernels)
  : context_(std::move(context)),
    program_(std::move(prog)),
    queue_(std::move(queue)),
    available_kernels_(std::move(available_kernels)) {
  // nop
}

program::~program() {
  // nop
}

} // namespace caf::opencl
*/
