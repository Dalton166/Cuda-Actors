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
 * (at your option) under the Boost Software      *
 * License 1.0. See accompanying files LICENSE and LICENSE_ALTERNATIVE.       *
 *                                                                            *
 * If you did not receive a copy of the license files, see                    *
 * http://opensource.org/licenses/BSD-3-Clause and                            *
 * http://www.boost.org/LICENSE_1_0.txt.                                      *
 ******************************************************************************/

#include "caf/cuda/platform.hpp"

#include <iostream>
/*
namespace caf {
namespace cuda {

platform::platform(cl_platform_id id, unsigned& current_device_id)
    : id_(id) {
  cl_uint num_devices = 0;
  clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
  if (num_devices == 0)
    return;
  std::vector<cl_device_id> device_ids(num_devices);
  clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(),
                 nullptr);
  devices_.reserve(num_devices);
  for (auto dev_id : device_ids) {
    devices_.emplace_back(std::make_shared<device>(dev_id, context_, queue_));
    ++current_device_id;
  }
}

platform::~platform() {
  // no explicit cleanup
}

std::string platform::name() const {
  size_t len = 0;
  clGetPlatformInfo(id_, CL_PLATFORM_NAME, 0, nullptr, &len);
  std::string result(len, '\0');
  clGetPlatformInfo(id_, CL_PLATFORM_NAME, len, result.data(), nullptr);
  if (!result.empty() && result.back() == '\0')
    result.pop_back();
  return result;
}

} // namespace opencl
} // namespace caf
*/
