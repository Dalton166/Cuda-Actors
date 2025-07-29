#pragma once

#include "caf/cuda/global.hpp"
#include <caf/message.hpp>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>

namespace caf::cuda {

class behavior_base {
public:
  virtual ~behavior_base() = default;
};

template <typename... Args>
class behavior : public behavior_base {
public:
  using args_tuple = std::tuple<Args...>;
  using preprocess_fn = std::function<void(caf::message&)>;
  using postprocess_fn = std::function<void(output_buffer&)>;

  behavior(program_ptr prog,
           nd_range range,
           std::string name,
           std::optional<preprocess_fn> pre = std::nullopt,
           std::optional<postprocess_fn> post = std::nullopt)
    : prog_(std::move(prog)),
      range_(std::move(range)),
      name_(std::move(name)),
      pre_(std::move(pre)),
      post_(std::move(post)) {}

  // Getters
  const program_ptr& program() const noexcept {
    return prog_;
  }

  const nd_range& range() const noexcept {
    return range_;
  }

  const std::string& name() const noexcept {
    return name_;
  }

  const std::optional<preprocess_fn>& preprocessor() const noexcept {
    return pre_;
  }

  const std::optional<postprocess_fn>& postprocessor() const noexcept {
    return post_;
  }

  std::tuple<Args...> args() const {
    return {};
  }

  static constexpr std::size_t num_args() noexcept {
    return sizeof...(Args);
  }

private:
  program_ptr prog_;
  nd_range range_;
  std::string name_;
  std::optional<preprocess_fn> pre_;
  std::optional<postprocess_fn> post_;
};

} // namespace caf::cuda

