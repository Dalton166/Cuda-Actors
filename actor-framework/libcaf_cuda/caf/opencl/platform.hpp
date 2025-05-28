#pragma once

#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>

namespace caf::opencl {

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
