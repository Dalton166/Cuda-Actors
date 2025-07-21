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

#include <variant>
#include <vector>
#include <iostream>
#include <stdexcept>

template <typename T>
class in_impl {
private:
  std::variant<T, std::vector<T>> data_;

public:
  using value_type = T;

  in_impl() : data_(T{}) {
    std::cout << "[in_impl] Default constructed (scalar initialized)\n";
  }

  explicit in_impl(const T& val) : data_(val) {
    std::cout << "[in_impl] Constructed scalar with value\n";
  }

  explicit in_impl(const std::vector<T>& buf) : data_(buf) {
    std::cout << "[in_impl] Constructed buffer of size " << buf.size() << "\n";
  }

  in_impl(const in_impl& other) : data_(other.data_) {
    std::cout << "[in_impl] Copy constructed\n";
  }

  in_impl(in_impl&& other) noexcept : data_(std::move(other.data_)) {
    std::cout << "[in_impl] Move constructed\n";
  }

  in_impl& operator=(const in_impl& other) {
    std::cout << "[in_impl] Copy assigned\n";
    data_ = other.data_;
    return *this;
  }

  in_impl& operator=(in_impl&& other) noexcept {
    std::cout << "[in_impl] Move assigned\n";
    data_ = std::move(other.data_);
    return *this;
  }

  ~in_impl() {
    std::cout << "[in_impl] Destroyed\n";
  }

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
    const T* ptr = is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
    std::cout << "[in_impl] data() returns pointer " << static_cast<const void*>(ptr)
              << ", size = " << size() << "\n";
    return ptr;
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

  out_impl() : data_(T{}) {
    std::cout << "[out_impl] Default constructed (scalar initialized)\n";
  }

  explicit out_impl(const T& val) : data_(val) {
    std::cout << "[out_impl] Constructed scalar with value\n";
  }

  explicit out_impl(const std::vector<T>& buf) : data_(buf) {
    std::cout << "[out_impl] Constructed buffer of size " << buf.size() << "\n";
  }

  out_impl(const out_impl& other) : data_(other.data_) {
    std::cout << "[out_impl] Copy constructed\n";
  }

  out_impl(out_impl&& other) noexcept : data_(std::move(other.data_)) {
    std::cout << "[out_impl] Move constructed\n";
  }

  out_impl& operator=(const out_impl& other) {
    std::cout << "[out_impl] Copy assigned\n";
    data_ = other.data_;
    return *this;
  }

  out_impl& operator=(out_impl&& other) noexcept {
    std::cout << "[out_impl] Move assigned\n";
    data_ = std::move(other.data_);
    return *this;
  }

  ~out_impl() {
    std::cout << "[out_impl] Destroyed\n";
  }

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
    const T* ptr = is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
    std::cout << "[out_impl] data() returns pointer " << static_cast<const void*>(ptr)
              << ", size = " << size() << "\n";
    return ptr;
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

  in_out_impl() : data_(T{}) {
    std::cout << "[in_out_impl] Default constructed (scalar initialized)\n";
  }

  explicit in_out_impl(const T& val) : data_(val) {
    std::cout << "[in_out_impl] Constructed scalar with value\n";
  }

  explicit in_out_impl(const std::vector<T>& buf) : data_(buf) {
    std::cout << "[in_out_impl] Constructed buffer of size " << buf.size() << "\n";
  }

  in_out_impl(const in_out_impl& other) : data_(other.data_) {
    std::cout << "[in_out_impl] Copy constructed\n";
  }

  in_out_impl(in_out_impl&& other) noexcept : data_(std::move(other.data_)) {
    std::cout << "[in_out_impl] Move constructed\n";
  }

  in_out_impl& operator=(const in_out_impl& other) {
    std::cout << "[in_out_impl] Copy assigned\n";
    data_ = other.data_;
    return *this;
  }

  in_out_impl& operator=(in_out_impl&& other) noexcept {
    std::cout << "[in_out_impl] Move assigned\n";
    data_ = std::move(other.data_);
    return *this;
  }

  ~in_out_impl() {
    std::cout << "[in_out_impl] Destroyed\n";
  }

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
    const T* ptr = is_scalar() ? &std::get<T>(data_) : std::get<std::vector<T>>(data_).data();
    std::cout << "[in_out_impl] data() returns pointer " << static_cast<const void*>(ptr)
              << ", size = " << size() << "\n";
    return ptr;
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


