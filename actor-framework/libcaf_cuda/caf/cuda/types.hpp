#pragma once

#include <caf/intrusive_ptr.hpp>

// Export macro for shared library visibility
#if defined(_MSC_VER)
  #define CAF_CUDA_EXPORT __declspec(dllexport)
#else
  #define CAF_CUDA_EXPORT __attribute__((visibility("default")))
#endif

namespace caf::cuda {

// Forward declarations and intrusive_ptr aliases

class device;
using device_ptr = caf::intrusive_ptr<device>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

template <class T>
class mem_ref;
template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;

// Forward declare manager, command, and actor_facade

class CAF_CUDA_EXPORT manager;

template <class Actor, class... Ts>
class command;

template <bool PassConfig, class... Ts>
class actor_facade;

} // namespace caf::cuda

