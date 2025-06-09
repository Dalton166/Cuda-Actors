#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>

#include <caf/actor_system.hpp>
#include <caf/actor.hpp>
#include <caf/intrusive_ptr.hpp>
#include <caf/detail/type_list.hpp>

#include "caf/detail/spawn_helper.hpp"

#include "caf/cuda/device.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/actor_facade.hpp"
#include "caf/cuda/opencl_err.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/platform.hpp"

namespace caf::cuda {

class device;
using device_ptr = caf::intrusive_ptr<device>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

template <bool PassConfig, class... Ts>
class actor_facade;

class CAF_CUDA_EXPORT manager {
public:
  explicit manager(caf::actor_system& sys);

  ~manager();

  device_ptr find_device(std::size_t id) const;

  template <class Predicate>
  device_ptr find_device_if(Predicate&&) const {
    throw std::runtime_error("CUDA support disabled: manager::find_device_if");
  }

  program_ptr create_program(const std::string& source,
                             const char* options,
                             device_ptr dev);

  program_ptr create_program(const char * kernel,
                             const std::string& name,
                             device_ptr);


  program_ptr create_program_from_file(const std::string& filename,
                                       const char* options,
                                       device_ptr dev);

  template <bool PassConfig, class Result, class... Ts>
  caf::actor spawn(const char*,
                   program_ptr,
                   Ts&&...) {
    throw std::runtime_error("CUDA support disabled: manager::spawn");
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




  //this constructor should spawn in a actor facade and compile a kernel 
  template<class ... Ts>
  caf::actor spawn(const char * kernel,
		  const std::string& name,
		  Ts&& ... xs) {
          caf::detail::cuda_spawn_helper<false,Ts ...> f;  
          caf::actor_config cfg;

	  //0 is the id for the first device
	  device_ptr device = find_device(0);

	  program_ptr prog = create_program(kernel,name,device);
          return f(
                   system_,
                   std::move(cfg),
		   std::move(prog),
                    std::forward<Ts>(xs)...);
  }




  caf::actor_system& system();

private:
  caf::actor_system& system_;
  platform_ptr platform_ = platform::create();
  bool compile_nvrtc_program(const char * source, CUdevice device, std::vector<char>&ptx_out);
  std::string get_computer_architecture_string(CUdevice device);

  device_ptr find_device(int id);
};

} // namespace caf::cuda
  //
 

/* 
 * #pragma once

#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>

namespace caf::cuda {

class device;
using device_ptr = caf::intrusive_ptr<device>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

class manager {
public:
  explicit manager(caf::actor_system&) {
    throw std::runtime_error("OpenCL support disabled: manager ctor");
  }

  device_ptr find_device(size_t) const {
    throw std::runtime_error("OpenCL support disabled: manager::find_device()");
  }

  template <class Predicate>
  device_ptr find_device_if(Predicate&&) const {
    throw std::runtime_error("OpenCL support disabled: manager::find_device_if()");
  }

  program_ptr create_program(const std::string&, const char*, device_ptr) {
    throw std::runtime_error("OpenCL support disabled: manager::create_program()");
  }

  program_ptr create_program_from_file(const std::string&, const char*, device_ptr) {
    throw std::runtime_error("OpenCL support disabled: manager::create_program_from_file()");
  }

  caf::actor_system& system() {
    throw std::runtime_error("OpenCL support disabled: manager::system()");
  }
};

} // namespace caf::opencl
 */

