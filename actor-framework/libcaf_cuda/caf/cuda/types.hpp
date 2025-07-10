#pragma once

#include <caf/intrusive_ptr.hpp>
#include <variant>
#include <vector>
#include <stdexcept>

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


// === buffer_variant and output_buffer outside namespace or inside as needed ===

using buffer_variant = std::variant<std::vector<char>, std::vector<int>, std::vector<float>, std::vector<double>>;

struct output_buffer {
  buffer_variant data;
};


// === Wrapper types for in/out/in_out with default ctor, union safely used ===

template <typename T>
class in_impl {
private:
  std::variant<T, std::vector<T>> data_;

public:
  using value_type = T;

  // Default constructor - scalar default initialized
  in_impl() : data_(T{}) {}

  // Scalar constructor
  explicit in_impl(const T& val) : data_(val) {}

  // Vector constructor
  explicit in_impl(const std::vector<T>& buf) : data_(buf) {}

  bool is_scalar() const {
    return std::holds_alternative<T>(data_);
  }

  const T& getscalar() const {
    if (!is_scalar())
      throw std::runtime_error("in_impl does not hold scalar");
    return std::get<T>(data_);
  }

  const std::vector<T>& get_buffer() const {
    if (is_scalar())
      throw std::runtime_error("in_impl does not hold buffer");
    return std::get<std::vector<T>>(data_);
  }

  const T* data() const {
    return is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
  }

  std::size_t size() const {
    return is_scalar() ? 1 : std::get<std::vector<T>>(data_).size();
  }
};

template <typename T>
class out_impl {
private:
  std::variant<T, std::vector<T>> data_;

public:
  using value_type = T;

  out_impl() : data_(T{}) {}

  explicit out_impl(const T& val) : data_(val) {}

  explicit out_impl(const std::vector<T>& buf) : data_(buf) {}

  bool is_scalar() const {
    return std::holds_alternative<T>(data_);
  }

  const T& getscalar() const {
    if (!is_scalar())
      throw std::runtime_error("out_impl does not hold scalar");
    return std::get<T>(data_);
  }

  const std::vector<T>& get_buffer() const {
    if (is_scalar())
      throw std::runtime_error("out_impl does not hold buffer");
    return std::get<std::vector<T>>(data_);
  }

  const T* data() const {
    return is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
  }

  std::size_t size() const {
    return is_scalar() ? 1 : std::get<std::vector<T>>(data_).size();
  }
};

template <typename T>
class in_out_impl {
private:
  std::variant<T, std::vector<T>> data_;

public:
  using value_type = T;

  in_out_impl() : data_(T{}) {}

  explicit in_out_impl(const T& val) : data_(val) {}

  explicit in_out_impl(const std::vector<T>& buf) : data_(buf) {}

  bool is_scalar() const {
    return std::holds_alternative<T>(data_);
  }

  const T& getscalar() const {
    if (!is_scalar())
      throw std::runtime_error("in_out_impl does not hold scalar");
    return std::get<T>(data_);
  }

  const std::vector<T>& get_buffer() const {
    if (is_scalar())
      throw std::runtime_error("in_out_impl does not hold buffer");
    return std::get<std::vector<T>>(data_);
  }

  const T* data() const {
    return is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
  }

  std::size_t size() const {
    return is_scalar() ? 1 : std::get<std::vector<T>>(data_).size();
  }
};

// Aliases
template <typename T>
using in = in_impl<T>;

template <typename T>
using out = out_impl<T>;

template <typename T>
using in_out = in_out_impl<T>;

// Helper to get raw type inside wrapper
template <typename T>
struct raw_type {
  using type = T;
};

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


