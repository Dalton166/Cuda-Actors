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

  virtual const program_ptr& program() const noexcept = 0;
  virtual const nd_range& range() const noexcept = 0;
  virtual const std::string& name() const noexcept = 0;
  virtual const std::optional<std::function<void(caf::message&)>>& preprocessor() const noexcept = 0;
  virtual const std::optional<std::function<void(output_buffer&)>>& postprocessor() const noexcept = 0;
  virtual std::size_t num_args() const noexcept = 0;
};

using behavior_base_ptr = std::shared_ptr<behavior_base>;



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

  // === Overrides from behavior_base ===
  const program_ptr& program() const noexcept override { return prog_; }
  const nd_range& range() const noexcept override { return range_; }
  const std::string& name() const noexcept override { return name_; }
  const std::optional<preprocess_fn>& preprocessor() const noexcept override { return pre_; }
  const std::optional<postprocess_fn>& postprocessor() const noexcept override { return post_; }
  std::size_t num_args() const noexcept override { return sizeof...(Args); }

  // Access to tag args tuple
  const tag_types& arg_tags() const noexcept { return args_tuple_; }

private:
  program_ptr prog_;
  nd_range range_;
  std::string name_;
  std::optional<preprocess_fn> pre_;
  std::optional<postprocess_fn> post_;
  tag_types args_tuple_;
};

} // namespace caf::cuda

