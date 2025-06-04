#include "caf/cuda/manager.hpp"
#include <stdexcept>

namespace caf::cuda {

manager::manager(caf::actor_system& sys) : system_(sys) {
  // no-op
  //check(cuInit(0),"cuInit");
}

manager::~manager() {
  // no-op
}

device_ptr manager::find_device(std::size_t) const {
  throw std::runtime_error("OpenCL support disabled: manager::find_device");
}

program_ptr manager::create_program(const std::string&,
                                    const char*,
                                    device_ptr) {
  throw std::runtime_error("OpenCL support disabled: manager::create_program");
}

program_ptr manager::create_program_from_file(const std::string&,
                                              const char*,
                                              device_ptr) {
  throw std::runtime_error("OpenCL support disabled: manager::create_program_from_file");
}

caf::actor_system& manager::system() {
  return system_;
}

} // namespace caf::cuda
