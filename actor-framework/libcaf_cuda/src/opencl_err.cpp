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

#include "caf/opencl/opencl_err.hpp"

#include "caf/logger.hpp"
/*
namespace caf {
namespace opencl {

void throwcl(const char*, cl_int err) {
  std::fprintf(stderr,
               "ERROR: throwcl() not functional (OpenCL removed). Err code=%d\n",
               err);
  std::abort();

  
  Original:
  if (err != CL_SUCCESS) {
    CAF_RAISE_ERROR("throwcl: unrecoverable OpenCL error");
  }
  
}

void CL_CALLBACK pfn_notify(const char* errinfo, const void*, size_t, void*) {
  std::fprintf(stderr,
               "ERROR: pfn_notify() not functional (OpenCL removed): %s\n",
               errinfo);
  std::abort();

  
  Original:
  CAF_LOG_ERROR("\n##### Error message via pfn_notify #####\n"
                << errinfo <<
                "\n########################################");
  
}

} // namespace opencl
} // namespace caf
*/
