#pragma once

#include "caf/cuda/global.hpp"
#include <caf/message.hpp>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace caf::cuda {

class behavior_base {
public:
  virtual ~behavior_base() = default;
};

// Main template
template <typename... Args>
class behavior : public behavior_base {
public:
  using tag_types = std::tuple<Args...>;
  using raw_types = std::tuple<raw_t<Args>...>;
  using preprocess_fn = std::function<void(caf::message&)>;
  using postprocess_fn = std::function<void(output_buffer&)>;

  behavior(program_ptr prog,
           nd_range range,
           std::string name,
           std::optional<preprocess_fn> pre,
           std::optional<postprocess_fn> post,
           Args&&... args)
    : prog_(std::move(prog)),
      range_(std::move(range)),
      name_(std::move(name)),
      pre_(std::move(pre)),
      post_(std::move(post)),
      args_tuple_(std::forward<Args>(args)...) {}

  // Overload for no pre/post lambdas (optional)
  behavior(program_ptr prog,
           nd_range range,
           std::string name,
           Args&&... args)
    : prog_(std::move(prog)),
      range_(std::move(range)),
      name_(std::move(name)),
      pre_(std::nullopt),
      post_(std::nullopt),
      args_tuple_(std::forward<Args>(args)...) {}

  // === Getters ===
  const program_ptr& program() const noexcept { return prog_; }
  const nd_range& range() const noexcept { return range_; }
  const std::string& name() const noexcept { return name_; }
  const std::optional<preprocess_fn>& preprocessor() const noexcept { return pre_; }
  const std::optional<postprocess_fn>& postprocessor() const noexcept { return post_; }
  const tag_types& arg_tags() const noexcept { return args_tuple_; }

  static constexpr std::size_t num_args() noexcept { return sizeof...(Args); }

private:
  program_ptr prog_;
  nd_range range_;
  std::string name_;
  std::optional<preprocess_fn> pre_;
  std::optional<postprocess_fn> post_;
  tag_types args_tuple_;
};

} // namespace caf::cuda

