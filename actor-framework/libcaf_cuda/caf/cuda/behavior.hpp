#pragma once

#include <tuple>
#include <vector>
#include <functional>
#include <string>
#include <caf/actor.hpp>
#include <caf/message.hpp>
#include <caf/response_promise.hpp>
#include <caf/intrusive_ptr.hpp>

#include "caf/cuda/global.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/command.hpp" // your command class header

namespace caf::cuda {

template <class... Ts>
class AbstractBehavior {
public:
  using preprocess_fn = std::function<void(const caf::message&)>;

  template <class... Args>
  AbstractBehavior(std::string name,
                   program_ptr program,
                   nd_range dims,
                   int actor_id,
                   preprocess_fn preprocess,
                   std::vector<caf::actor> targets = {},
                   Args&&... xs)
    : name_(std::move(name)),
      program_(std::move(program)),
      dims_(std::move(dims)),
      actor_id_(actor_id),
      preprocess_(std::move(preprocess)),
      targets_(std::move(targets)),
      args_(std::forward<Args>(xs)...)
  {}

  virtual ~AbstractBehavior() = default;

  // Main entry point with response promise
  virtual void execute(const caf::message& msg, int actor_id, caf::response_promise& rp) {
    rp_ = std::move(rp);
    preprocess(msg);
    auto results = execute_command(msg, actor_id);
    postprocess(results);
    reply(results, rp_);
    cleanup();
  }

  // Fire-and-forget entry point (no response promise)
  virtual void execute(const caf::message& msg, int actor_id) {
    preprocess(msg);
    auto results = execute_command(msg, actor_id);
    postprocess(results);
    reply(results);
    cleanup();
  }

  // Virtual command execution, overridable by derived classes
  virtual std::tuple<mem_ptr<raw_t<Ts>>...> execute_command(const caf::message& msg, int actor_id) {
    return execute_command_impl(msg, actor_id, std::make_index_sequence<sizeof...(Ts)>{});
  }

  // Getters
  const std::string& name() const {
    return name_;
  }

  bool is_asynchronous() const {
    return is_asynchronous_;
  }

protected:
  // Helper to unpack args_ tuple and launch the command
  template <std::size_t... Is>
  std::tuple<mem_ptr<raw_t<Ts>>...> execute_command_impl(const caf::message& msg, int actor_id, std::index_sequence<Is...>) {
    auto cmd = caf::make_counted<command<caf::actor, Ts...>>(
      std::move(rp_),
      self_,
      program_,
      dims_,
      actor_id,
      std::get<Is>(args_)...);
    cmd->enqueue();

    // NOTE: For now, returning empty default-constructed tuple since actual mem_refs come asynchronously
    return std::tuple<mem_ptr<raw_t<Ts>>...>{};
  }

  // Virtual hooks — override as needed
  virtual void preprocess(const caf::message& msg) {
    if (preprocess_) preprocess_(msg);
  }

  virtual void postprocess(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/) {
    // Default no-op
  }

  virtual void reply(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/, caf::response_promise& /*rp*/) {
    // Default no-op
  }

  virtual void reply(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/) {
    // Default no-op
  }

  virtual void cleanup() {
    // Default no-op
  }

protected:
  std::string name_;
  bool is_asynchronous_ = false;
  program_ptr program_;
  nd_range dims_;
  int actor_id_;
  preprocess_fn preprocess_;
  std::vector<caf::actor> targets_;
  std::tuple<Ts...> args_;
  caf::response_promise rp_;
  caf::actor self_; // You might want to set this after construction or via a setter

};

template <class... Ts>
using behavior_ptr = caf::intrusive_ptr<AbstractBehavior<Ts...>>;

} // namespace caf::cuda

