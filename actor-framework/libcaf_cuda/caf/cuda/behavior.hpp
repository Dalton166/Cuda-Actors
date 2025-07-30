#pragma once

#include "caf/cuda/global.hpp"
#include <caf/actor.hpp>
#include <caf/message.hpp>

#include <string>
#include <tuple>
#include <vector>
#include <functional>
#include <utility>

namespace caf::cuda {

template <class... Ts>
class AbstractBehavior {
public:
  using preprocess_fn = std::function<void(const caf::message&)>;

  template <class... Us>
  using postprocess_fn = std::function<void(std::tuple<mem_ref<Us>...>&)>;

  AbstractBehavior(std::string name,
                 program_ptr program,
                 nd_range dims,
                 preprocess_fn preprocess,
                 std::vector<caf::actor> targets = {},
                 Args&&... xs)
  : name_(std::move(name)),
    program_(std::move(program)),
    dims_(std::move(dims)),
    preprocess_(std::move(preprocess)),
    targets_(std::move(targets)),
    args_(std::forward<Args>(xs)...) // perfect-forward and store as tuple
{},

  virtual ~AbstractBehavior() = default;

  // Entry point with response promise
  virtual void execute(const caf::message& msg, int actor_id, caf::response_promise& rp) {
    rp_ = rp;
    preprocess(msg);
    auto results = execute_command<Ts...>(msg, actor_id);
    postprocess(results);
    reply(results, rp_);
    cleanup();
  }

  // Fire-and-forget entry point without response promise
  virtual void execute(const caf::message& msg, int actor_id) {
    preprocess(msg);
    auto results = execute_command<Ts...>(msg, actor_id);
    postprocess(results);
    reply(results);
    cleanup();
  }

  // Execute command: create and enqueue kernel command
  template <class... Us >
  std::tuple<mem_ref<Us>...> execute_command(const caf::message& msg, int actor_id) {
    // Example command creation and enqueue:
    using command_t = command<caf::actor, raw_t<Us>...>;
    auto cmd = make_counted<command_t>(
      make_response_promise(),
      caf::actor_cast<caf::actor>(this),
      program_,
      dims_,
      actor_id,
      /* Here you’d extract and forward args from msg */);
    cmd->enqueue();

    // Return kernel outputs (stub: implement actual logic)
    return std::tuple<mem_ref<Us>...>{};
  }

  // Postprocess setter
  template <class... Us>
  void set_postprocess(postprocess_fn<Us...> fn) {
    postprocess_ = [fn = std::move(fn)](std::tuple<mem_ref<Us>...>& results) {
      fn(results);
    };
  }

  template <class... Us>
  void postprocess(std::tuple<mem_ref<Us>...>& results) {
    if (postprocess_) {
      postprocess_(results);
    }
  }

  // Templated reply overloads with default no-op implementations
  template <class... Us>
  void reply(const std::tuple<mem_ref<Us>...>& /*results*/, caf::response_promise& /*rp*/) {
    // default: do nothing
  }

  template <class... Us>
  void reply(const std::tuple<mem_ref<Us>...>& /*results*/) {
    // default: do nothing
  }

  virtual void cleanup() {}

  // Accessors
  const std::string& name() const { return name_; }
  bool is_asynchronous() const { return is_asynchronous_; }
  const std::vector<caf::actor>& targets() const { return targets_; }
  const program_ptr& program() const { return program_; }
  const nd_range& dims() const { return dims_; }

protected:
  std::string name_;
  bool is_asynchronous_ = false;
  program_ptr program_;
  nd_range dims_;
  preprocess_fn preprocess_;
  std::vector<caf::actor> targets_;
  caf::response_promise rp_;

  std::function<void(std::any&)> postprocess_any_;
};

template <class... Ts>
using behavior_ptr = caf::intrusive_ptr<AbstractBehavior<Ts...>>;

} // namespace caf::cuda

