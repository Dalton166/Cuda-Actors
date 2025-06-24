#pragma once

#include <stdexcept>
#include <functional>
#include <tuple>
#include <iostream>
#include <queue>
#include <caf/local_actor.hpp>
#include <caf/actor.hpp>
#include <caf/response_promise.hpp>
#include <caf/logger.hpp>
#include <caf/raise_error.hpp>
#include <caf/scheduler.hpp>
#include <caf/resumable.hpp>
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/command.hpp"

namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade : public caf::local_actor, public caf::resumable {
public:
  static caf::actor create(
    caf::actor_system& sys,
    caf::actor_config&& actor_conf,
    program_ptr program,
    nd_range dims,
    Ts&&... xs
  ) {
    CAF_LOG_DEBUG("Creating actor with ID: " << sys.next_actor_id());
    return caf::make_actor<actor_facade<PassConfig, std::decay_t<Ts>...>, caf::actor>(
      sys.next_actor_id(),
      sys.node(),
      &sys,
      std::move(actor_conf),
      std::move(program),
      std::move(dims),
      std::forward<Ts>(xs)...);
  }

  static caf::actor create(
    caf::actor_system* sys,
    caf::actor_config&& actor_conf,
    program_ptr program,
    nd_range dims,
    Ts&&... xs
  ) {
    CAF_LOG_DEBUG("Creating actor with ID: " << sys->next_actor_id());
    return caf::make_actor<actor_facade<PassConfig, std::decay_t<Ts>...>, caf::actor>(
      sys->next_actor_id(),
      sys->node(),
      sys,
      std::move(actor_conf),
      std::move(program),
      std::move(dims),
      std::forward<Ts>(xs)...);
  }

  actor_facade(caf::actor_config&& cfg, program_ptr prog, nd_range nd, Ts&&... xs)
    : local_actor(cfg),
      config_(std::move(cfg)),
      program_(std::move(prog)),
      dims_(std::move(nd)) {
    CAF_LOG_DEBUG("Actor constructed with config: " << &cfg);
  }

  void create_command(program_ptr program, Ts&&... xs) {
    CAF_LOG_DEBUG("Creating command for kernel launch");
    caf::response_promise rp = make_response_promise();
    using command_t = command<caf::actor, raw_t<Ts>...>;
    auto cmd = make_counted<command_t>(rp, program, dims_, std::forward<Ts>(xs)...);
    cmd->enqueue(); // launches the kernel
  }

  void run_kernel(Ts&... xs) {
    CAF_LOG_DEBUG("Running kernel");
    create_command(program_, std::forward<Ts>(xs)...);
  }

private:
  caf::actor_config config_;
  program_ptr program_;
  nd_range dims_;
  std::queue<mailbox_element_ptr> mailbox_; // Custom mailbox for CAF 1.0.2

  void print_args() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_args(T&& first, Rest&&... rest) {
    std::cout << first << "\n";
    print_args(std::forward<Rest>(rest)...);
  }

bool handle_message(const message& msg) {
  CAF_LOG_DEBUG("Handling message");
  std::cout << "Hello\n";

  // Check first argument is caf::actor
  if (!msg.types().empty() && msg.types()[0] == caf::type_id_v<caf::actor>) {
    // Extract the sender actor (first argument)
    auto sender = msg.get_as<caf::actor>(0);

    // Prepare to handle the rest of the args (from index 1 on)
    if (msg.match_elements<caf::actor, Ts...>()) {
      // Wrapped types match (with sender + wrappers)
      return unpack_and_run_wrapped(sender, msg, std::index_sequence_for<Ts...>{});
    }
    if (msg.match_elements<caf::actor, raw_t<Ts>...>()) {
      // Raw types match (with sender + raw types)
      return unpack_and_run(sender, msg, std::index_sequence_for<Ts...>{});
    }
  }

  std::cout << "Message argument types do not match expected pattern\n";
  std::cout << "Message types: " << to_string(msg.types()) << "\n";
  CAF_LOG_ERROR("Message argument types do not match expected pattern");
  return false;
}

template <std::size_t... Is>
bool unpack_and_run_wrapped(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
  CAF_LOG_DEBUG("Unpacking wrapped types with sender");
  // Extract wrapped args starting from index 1
  auto wrapped = std::make_tuple(msg.get_as<Ts>(Is + 1)...);
  print_wrapped(std::get<Is>(wrapped)...);

  // Pass sender + wrapped args to run_kernel
  run_kernel(std::get<Is>(wrapped)...);
  return true;
}

template <std::size_t... Is>
bool unpack_and_run(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
  CAF_LOG_DEBUG("Unpacking raw types with sender");
  // Extract raw args starting from index 1
  auto unpacked = std::make_tuple(msg.get_as<raw_t<Ts>>(Is + 1)...);
  // Wrap them
  auto wrapped = std::make_tuple(Ts(std::get<Is>(unpacked))...);
  print_wrapped(std::get<Is>(wrapped)...);

  // Pass sender + wrapped args to run_kernel
  run_kernel(std::get<Is>(wrapped)...);
  return true;
}

  void print_wrapped() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_wrapped(T&& first, Rest&&... rest) {
    std::cout << "Buffer contents: ";
    for (auto& v : first.buffer) {
      std::cout << v << ' ';
    }
    std::cout << '\n';
    print_wrapped(std::forward<Rest>(rest)...);
  }

  // Implement caf::resumable interface
  subtype_t subtype() const noexcept override {
    return subtype_t(0); // Placeholder
  }

  resumable::resume_result resume(scheduler* sched, [[maybe_unused]] size_t max) override {
    CAF_LOG_DEBUG("Resuming actor, mailbox size: " << mailbox_.size());
    while (!mailbox_.empty()) {
      auto msg = std::move(mailbox_.front());
      mailbox_.pop();
      if (msg && msg->content()) {
        current_mailbox_element(msg.get()); // Set for CAF message processing
        handle_message(msg->content());
        current_mailbox_element(nullptr); // Reset after processing
      }
    }
    if (sched) {
      CAF_LOG_DEBUG("Scheduling actor after resume");
      sched->schedule(this);
    }
    return resumable::done;
  }

  void ref_resumable() const noexcept override {
    CAF_LOG_DEBUG("Referencing resumable");
  }

  void deref_resumable() const noexcept override {
    CAF_LOG_DEBUG("Dereferencing resumable");
  }

  // Implement caf::local_actor virtual methods
  bool enqueue(mailbox_element_ptr what, scheduler* sched) override {
    if (!what) {
      CAF_LOG_DEBUG("Enqueue failed: null message");
      return false;
    }
    CAF_LOG_DEBUG("Enqueuing message, mailbox size: " << mailbox_.size());
    bool was_empty = mailbox_.empty();
    mailbox_.push(std::move(what));
    if (was_empty && sched) {
      CAF_LOG_DEBUG("Scheduling actor");
      sched->schedule(this);
    }
    return true;
  }

  void launch(scheduler* sched, bool lazy, [[maybe_unused]] bool interruptible) override {
    if (!lazy && sched) {
      CAF_LOG_DEBUG("Launching actor");
      sched->schedule(this);
    }
  }

  void do_unstash(mailbox_element_ptr what) override {
    if (what) {
      CAF_LOG_DEBUG("Unstashing message");
      mailbox_.push(std::move(what));
    }
  }

  void force_close_mailbox() override {
    CAF_LOG_DEBUG("Closing mailbox, clearing " << mailbox_.size() << " messages");
    while (!mailbox_.empty()) {
      mailbox_.pop();
    }
  }
};

} // namespace caf::cuda
