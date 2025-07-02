#pragma once

#include <iostream>      // <-- for std::cout, std::endl
#include <stdexcept>     // for std::runtime_error, etc.
#include <functional>    // for std::function
#include <tuple>         // for std::tuple, std::make_tuple
#include <queue>         // for std::queue
#include <utility>       // for std::move, std::forward
#include <type_traits>   // for std::decay_t, etc.

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
//using kernel_done_atom = decltype(caf::atom("kernelDone"));
//constexpr auto kernel_done_atom_v = caf::atom("kernelDone");


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
    std::cout << "[LOG] static create (ref) called\n";
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
    std::cout << "[LOG] static create (ptr) called\n";
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
    std::cout << "[LOG] actor_facade constructor called\n";
    CAF_LOG_DEBUG("Actor constructed with config: " << &cfg);
  }

  ~actor_facade() {
    std::cout << "[LOG] ~actor_facade destructor called\n";
  }

  void create_command(program_ptr program, Ts&&... xs) {
  CAF_LOG_DEBUG("[LOG] create_command called");
  pending_promises_++;

  using command_t = command<caf::actor, raw_t<Ts>...>;
  auto cmd = make_counted<command_t>(
  make_response_promise(), caf::actor_cast<caf::actor>(this), program, dims_, std::forward<Ts>(xs)...);
  cmd->enqueue();
}  
  void run_kernel(Ts&... xs) {
    std::cout << "[LOG] run_kernel called\n";
    CAF_LOG_DEBUG("Running kernel");
    create_command(program_, std::forward<Ts>(xs)...);
  }

private:
  caf::actor_config config_;
  program_ptr program_;
  nd_range dims_;
  std::queue<mailbox_element_ptr> mailbox_; // Custom mailbox for CAF 1.0.2
  std::atomic<int> pending_promises_ = 0;
  std::atomic<bool> shutdown_requested_ = false;


  void print_args() {
    std::cout << "[LOG] print_args (no args)\n";
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_args(T&& first, Rest&&... rest) {
    std::cout << "[LOG] print_args (variadic)\n";
    std::cout << first << "\n";
    print_args(std::forward<Rest>(rest)...);
  }

  bool handle_message(const message& msg) {
    std::cout << "[LOG] handle_message called\n";
    CAF_LOG_DEBUG("Handling message");

    if (!msg.types().empty() && msg.types()[0] == caf::type_id_v<caf::actor>) {
      auto sender = msg.get_as<caf::actor>(0);

      if (msg.match_elements<caf::actor, Ts...>()) {
        return unpack_and_run_wrapped(sender, msg, std::index_sequence_for<Ts...>{});
      }
      if (msg.match_elements<caf::actor, raw_t<Ts>...>()) {
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
    std::cout << "[LOG] unpack_and_run_wrapped called\n";
    CAF_LOG_DEBUG("Unpacking wrapped types with sender");
    auto wrapped = std::make_tuple(msg.get_as<Ts>(Is + 1)...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }

  template <std::size_t... Is>
  bool unpack_and_run(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
    std::cout << "[LOG] unpack_and_run called\n";
    CAF_LOG_DEBUG("Unpacking raw types with sender");
    auto unpacked = std::make_tuple(msg.get_as<raw_t<Ts>>(Is + 1)...);
    auto wrapped = std::make_tuple(Ts(std::get<Is>(unpacked))...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }

  void print_wrapped() {
    std::cout << "[LOG] print_wrapped (no args)\n";
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_wrapped(T&& first, Rest&&... rest) {
    std::cout << "[LOG] print_wrapped (variadic)\n";
    std::cout << "Buffer contents: ";
    for (auto& v : first.buffer) {
      std::cout << v << ' ';
    }
    std::cout << '\n';
    print_wrapped(std::forward<Rest>(rest)...);
  }

  // caf::resumable interface
  subtype_t subtype() const noexcept override {
    std::cout << "[LOG] subtype called\n";
    return subtype_t(0);
  }

  resumable::resume_result resume(scheduler* sched, size_t) override {
  CAF_LOG_DEBUG("[LOG] resume called");
  while (!mailbox_.empty()) {
    auto msg = std::move(mailbox_.front());
    mailbox_.pop();

    if (!msg || !msg->content())
      continue;

    current_mailbox_element(msg.get());

    // Handle kernel completion message
    if (msg->content().match_elements<kernel_done_atom>()) {
      CAF_LOG_DEBUG("[LOG] received kernel_done_atom");
      if (--pending_promises_ == 0 && shutdown_requested_) {
        CAF_LOG_DEBUG("[LOG] no more pending work, quitting");
        quit(exit_reason::user_shutdown);
        return resumable::done;
      }
      current_mailbox_element(nullptr);
      continue;
    }

    // Handle exit message
    if (msg->content().match_elements<exit_msg>()) {
      auto exit = msg->content().get_as<exit_msg>(0);
      CAF_LOG_DEBUG("[LOG] exit message received");
      shutdown_requested_ = true;
      if (pending_promises_ == 0) {
        CAF_LOG_DEBUG("[LOG] immediate quit");
        quit(static_cast<exit_reason>(exit.reason.code()));
        return resumable::done;
      } else {
        CAF_LOG_DEBUG("[LOG] delaying quit due to pending work");
        current_mailbox_element(nullptr);
        return resumable::resume_later;
      }
    }

    // Default message handling
    handle_message(msg->content());

    current_mailbox_element(nullptr);
  }

  return shutdown_requested_ ? resumable::resume_later : resumable::done;
}

  void ref_resumable() const noexcept override {
    std::cout << "[LOG] ref_resumable called\n";
    CAF_LOG_DEBUG("Referencing resumable");
  }

  void deref_resumable() const noexcept override {
    std::cout << "[LOG] deref_resumable called\n";
    CAF_LOG_DEBUG("Dereferencing resumable");
  }

  // caf::local_actor interface
  bool enqueue(mailbox_element_ptr what, scheduler* sched) override {
    std::cout << "[LOG] enqueue called\n";
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
    std::cout << "[LOG] launch called\n";
    if (!lazy && sched) {
      CAF_LOG_DEBUG("Launching actor");
      sched->schedule(this);
    }
  }

  void do_unstash(mailbox_element_ptr what) override {
    std::cout << "[LOG] do_unstash called\n";
    if (what) {
      CAF_LOG_DEBUG("Unstashing message");
      mailbox_.push(std::move(what));
    }
  }

  void force_close_mailbox() override {
    std::cout << "[LOG] force_close_mailbox called\n";
    CAF_LOG_DEBUG("Closing mailbox, clearing " << mailbox_.size() << " messages");
    while (!mailbox_.empty()) {
      mailbox_.pop();
    }
  }

  void quit(exit_reason reason) {
    std::cout << "[LOG] quit called\n";
    CAF_LOG_DEBUG("Quitting actor with reason: " << to_string(reason));
    force_close_mailbox();
    current_mailbox_element(nullptr);
  }
};

} // namespace caf::cuda

