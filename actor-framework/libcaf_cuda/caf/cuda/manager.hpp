#pragma once

#include <string>
#include <cstddef>
#include <stdexcept>
#include <mutex>

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
  /// Initializes the singleton. Must be called exactly once before get().
  static void init(caf::actor_system& sys) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (instance_) {
      throw std::runtime_error("CUDA manager already initialized");
    }
    CHECK_CUDA(cuInit(0));
    instance_ = new manager(sys);
  }

  /// Returns the singleton instance. Crashes if not yet initialized.
  static manager& get() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!instance_) {
      throw std::runtime_error("CUDA manager used before initialization");
    }
    return *instance_;
  }

  /// Deletes the singleton if needed (optional).
  static void shutdown() {
    std::lock_guard<std::mutex> guard(mutex_);
    delete instance_;
    instance_ = nullptr;
  }

  // Prevent copy/assignment
  manager(const manager&) = delete;
  manager& operator=(const manager&) = delete;

  device_ptr find_device(std::size_t id) const;

  template <class Predicate>
  device_ptr find_device_if(Predicate&&) const {
    throw std::runtime_error("CUDA support disabled: manager::find_device_if");
  }

  program_ptr create_program(const std::string& source,
                             const char* options,
                             device_ptr dev);

  program_ptr create_program(const char* kernel,
                             const std::string& name,
                             device_ptr dev);

  program_ptr create_program_from_file(const std::string& filename,
                                       const char* options,
                                       device_ptr dev);

  template <bool PassConfig, class Result, class... Ts>
  caf::actor spawn(const char*,
                   program_ptr,
                   Ts&&...) {
    throw std::runtime_error("CUDA support disabled: manager::spawn");
  }

  template <class... Ts>
  caf::actor spawn(const char* kernel,
                   const std::string& name,
                   Ts&&... xs) {
    caf::detail::cuda_spawn_helper<false, Ts...> f;
    caf::actor_config cfg;

    device_ptr device = find_device(0);
    program_ptr prog = create_program(kernel, name, device);

    return f(&system_, std::move(cfg), std::move(prog), std::forward<Ts>(xs)...);
  }

  caf::actor_system& system() { return system_; }

private:
  explicit manager(caf::actor_system& sys)
    : system_(sys), platform_(platform::create()) {
    // cuInit is done in init()
  }

  caf::actor_system& system_;
  platform_ptr platform_;

  bool compile_nvrtc_program(const char* source, CUdevice device, std::vector<char>& ptx_out);
  std::string get_computer_architecture_string(CUdevice device);

  device_ptr find_device(int id);

  static manager* instance_;
  static std::mutex mutex_;
};

} // namespace caf::cuda

