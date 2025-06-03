#pragma once

#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>
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


#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"


/*
 * This class may be slightly irrevelant since 
 * opencl has to deal with multiple gpu platforms 
 * but cuda only works on nvida gpus so basically it becomes a 
 * find a device and grab it for me sort of deal 
 */


namespace caf {
namespace cuda {

class platform;
using platform_ptr = intrusive_ptr<platform>;

class platform : public ref_counted {
public:
  friend class program;
  template <class T, class... Ts>
  friend intrusive_ptr<T> caf::make_counted(Ts&&...);


  inline const std::string& name() const;
  inline const std::string& vendor() const;
  inline const std::string& version() const;
  static platform_ptr create();

private:

  int device_count = 0;
  check(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
  std::vector<CUdevice> devices(device_count);
  std::vector<CUcontext> contexts(device_count); 
  
  platform() {
	  //setup all possible gpus for that we can find 
    for (int i = 0; i < device_count; ++i) {
        check(cuDeviceGet(&devices[i], i), "cuDeviceGet");
        char name[256];
        cuDeviceGetName(name, 256, devices[i]);
        std::cout << "Device #" << i << ": " << name << "\n";

        // Create a context *without* setting it as current
        check(cuCtxCreate(&contexts[i], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, devices[i]), "cuCtxCreate");
    }

    // Choose one to be "active" for a particular operation
    int target_device = 1; // example
    check(cuCtxSetCurrent(contexts[target_device]), "cuCtxSetCurrent")



  }

  ~platform() {
  
	  for (auto ctx : contexts) {
        cuCtxDestroy(ctx);
    }
  
  }

  static std::string platform_info(cl_platform_id platform_id,
                                   unsigned info_flag);
  cl_platform_id platform_id_;
  detail::raw_context_ptr context_;
  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
};

/******************************************************************************\
 *                 implementation of inline member functions                  *
\******************************************************************************/

inline const std::vector<device_ptr>& platform::devices() const {
  return devices_;
}

inline const std::string& platform::name() const {
  return name_;
}

inline const std::string& platform::vendor() const {
  return vendor_;
}

inline const std::string& platform::version() const {
  return version_;
}


} // namespace cuda
} // namespace caf
