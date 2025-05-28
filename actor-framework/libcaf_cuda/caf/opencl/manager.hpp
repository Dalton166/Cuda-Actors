#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

#include <caf/actor_system.hpp>
#include <caf/actor.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/detail/type_list.hpp>

#include "caf/detail/spawn_helper.hpp"


#include "caf/opencl/device.hpp"
#include "caf/opencl/program.hpp"
#include "caf/opencl/actor_facade.hpp"
#include "caf/opencl/opencl_err.hpp"
#include "caf/opencl/global.hpp"

namespace caf::opencl {

class device;
using device_ptr = caf::intrusive_ptr<device>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

template <bool PassConfig, class... Ts>
class actor_facade;

class CAF_OPENCL_EXPORT manager {
public:
  explicit manager(caf::actor_system& sys);

  ~manager();

  device_ptr find_device(std::size_t id) const;

  template <class Predicate>
  device_ptr find_device_if(Predicate&&) const {
    throw std::runtime_error("OpenCL support disabled: manager::find_device_if");
  }

  program_ptr create_program(const std::string& source,
                             const char* options,
                             device_ptr dev);

  program_ptr create_program_from_file(const std::string& filename,
                                       const char* options,
                                       device_ptr dev);

  template <bool PassConfig, class Result, class... Ts>
  caf::actor spawn(const char*,
                   program_ptr,
                   Ts&&...) {
    throw std::runtime_error("OpenCL support disabled: manager::spawn");
  }


  template<class T,class ... Ts>
  caf::actor spawn(T &&x, Ts&& ... xs) {

	  caf::detail::cuda_spawn_helper<false,T> f;  
	  caf::actor_config cfg;
	  return f(
    		   system_,
	           std::move(cfg),
		   std::forward<T>(x),
		    std::forward<T>(xs)...);
  
  }

  caf::actor_system& system();

private:
  caf::actor_system& system_;
};

} // namespace caf::opencl

