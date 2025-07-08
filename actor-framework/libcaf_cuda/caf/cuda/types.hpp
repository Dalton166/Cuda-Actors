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

#include <vector>
#include <type_traits>

// helper to detect scalar types
template <typename T>
constexpr bool is_scalar_v = std::is_arithmetic_v<T> || std::is_enum_v<T>;

//struct wrappers to hold store buffers to declare them as in or out 

// in<T>
template <typename T, bool IsScalar = is_scalar_v<T>>
struct in_impl;

// scalar specialization
template <typename T>
struct in_impl<T, true> {
  using value_type = T;
  T value;

  in_impl() = default;

  // Construct from a single raw value
  in_impl(T val) : value(val) {}

  T* data() { return &value; }

  std::size_t size() const { return 1; }
};

// vector specialization
template <typename T>
struct in_impl<T, false> {
  using value_type = T;
  std::vector<T> buffer;

  in_impl() = default;

  // Construct from a single raw value by pushing it into buffer
  in_impl(T val) {
    buffer.push_back(val);
  }

  T* data() { return buffer.data(); }

  std::size_t size() const { return buffer.size(); }
};

template <typename T>
using in = in_impl<T>;


// out<T>
template <typename T, bool IsScalar = is_scalar_v<T>>
struct out_impl;

// scalar specialization
template <typename T>
struct out_impl<T, true> {
  using value_type = T;
  T value;

  out_impl() = default;

  out_impl(T val) : value(val) {}

  T* data() { return &value; }

  std::size_t size() const { return 1; }
};

// vector specialization
template <typename T>
struct out_impl<T, false> {
  using value_type = T;
  std::vector<T> buffer;

  out_impl() = default;

  out_impl(T val) {
    buffer.push_back(val);
  }

  T* data() { return buffer.data(); }

  std::size_t size() const { return buffer.size(); }
};

template <typename T>
using out = out_impl<T>;


// in_out<T>
template <typename T, bool IsScalar = is_scalar_v<T>>
struct in_out_impl;

// scalar specialization
template <typename T>
struct in_out_impl<T, true> {
  using value_type = T;
  T value;

  in_out_impl() = default;

  in_out_impl(T val) : value(val) {}

  T* data() { return &value; }

  std::size_t size() const { return 1; }
};

// vector specialization
template <typename T>
struct in_out_impl<T, false> {
  using value_type = T;
  std::vector<T> buffer;

  in_out_impl() = default;

  in_out_impl(T val) {
    buffer.push_back(val);
  }

  T* data() { return buffer.data(); }

  std::size_t size() const { return buffer.size(); }
};

template <typename T>
using in_out = in_out_impl<T>;


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





