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


using buffer_variant = std::variant<std::vector<char>, std::vector<int>, std::vector<float>, std::vector<double>>;

struct output_buffer {
  buffer_variant data;
};


//struct wrappers to hold store buffers to declare them as in or out 
template <typename T>
struct in {
  std::vector<T> buffer;

  in() = default;

  // Construct from a single raw value by pushing it into buffer
  in(T val) {
    buffer.push_back(val);
  }
};

template <typename T>
struct out {
  std::vector<T> buffer;

  out() = default;

  out(T val) {
    buffer.push_back(val);
  }
};



template <typename T>
struct in_out {
    using value_type = T;
    std::vector<T> buffer;

    in_out() = default;

    in_out(T val) {
      buffer.push_back(val);
    }
};


// Helper to get raw type inside wrapper
template <typename T>
struct raw_type {
  using type = T;
};

// specialization for your wrapper types
template <typename T>
struct raw_type<in<T>> {
  using type = T;
};

template <typename T>
struct raw_type<out<T>> {
  using type = T;
};

template <typename T>
struct raw_type<in_out<T>> {
  using type = T;
};

template <typename T>
using raw_t = typename raw_type<T>::type;





