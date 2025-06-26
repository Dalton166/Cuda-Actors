#pragma once

#include <caf/local_actor.hpp>
#include <caf/actor.hpp>
#include <caf/response_promise.hpp>
#include <caf/logger.hpp>
#include <caf/scheduler.hpp>
#include <caf/resumable.hpp>
#include <caf/ref_counted.hpp>

#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/command.hpp"

#include <iostream>
#include <queue>

namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade : public local_actor,
                     public resumable,
                     public ref_counted {
public:
  static caf::actor create(
    actor_system& sys,
    actor_config&& actor_conf,
    program_ptr program,
    nd_range dims,
    Ts&&... xs
  ) {
    CAF_LOG_DEBUG("Creating actor with ID: " << sys.next_actor_id());
    return caf::make_actor<actor_facade<PassConfig, std::decay_t<Ts>...>, actor>(
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
  CAF_LOG_DEBUG("Creating actor from pointer to system");
  // Forward to the reference version
  return create(*sys, std::move(actor_conf), std::move(program), std::move(dims), std::forward<Ts>(xs)...);
}



  actor_facade(actor_config&& cfg, program_ptr prog, nd_range nd, Ts&&...)
    : local_actor(cfg),
      config_(std::move(cfg)),
      program_(std::move(prog)),
      dims_(std::move(nd)) {
    CAF_LOG_DEBUG("Actor constructed");
  }

  ~actor_facade() override {
    std::cout << "Destroying actor facade\n";
    CAF_LOG_DEBUG("Destroying actor_facade");
  }

  // Required for ref_counted support used by CAF scheduler
  void ref_resumable() const noexcept override {
    intrusive_ptr_add_ref(const_cast<actor_facade*>(this));
  }

  void deref_resumable() const noexcept override {
    intrusive_ptr_release(const_cast<actor_facade*>(this));
  }

  subtype_t subtype() const noexcept override {
    return subtype_t(0); // Customize if you subclass
  }

  resumable::resume_result resume(scheduler* sched, size_t /*max*/) override {
    CAF_LOG_DEBUG("Resuming actor, mailbox size: " << mailbox_.size());
    while (!mailbox_.empty()) {
      auto msg = std::move(mailbox_.front());
      mailbox_.pop();
      if (msg && msg->content()) {
        current_mailbox_element(msg.get());
        handle_message(msg->content());
        current_mailbox_element(nullptr);
      }
    }
    if (sched)
      sched->schedule(this);
    return resumable::done;
  }

  bool enqueue(mailbox_element_ptr what, scheduler* sched) override {
    if (!what) return false;
    bool was_empty = mailbox_.empty();
    mailbox_.push(std::move(what));
    if (was_empty && sched)
      sched->schedule(this);
    return true;
  }

  void launch(scheduler* sched, bool lazy, bool /*interruptible*/) override {
    if (!lazy && sched)
      sched->schedule(this);
  }

  void do_unstash(mailbox_element_ptr what) override {
    if (what)
      mailbox_.push(std::move(what));
  }

  void force_close_mailbox() override {
    while (!mailbox_.empty())
      mailbox_.pop();
  }

  void run_kernel(Ts&... xs) {
    create_command(program_, std::forward<Ts>(xs)...);
  }




private:
  actor_config config_;
  program_ptr program_;
  nd_range dims_;
  std::queue<mailbox_element_ptr> mailbox_;

  void create_command(program_ptr prog, Ts&&... xs) {
    response_promise rp = make_response_promise();
    using command_t = command<actor, raw_t<Ts>...>;
    auto cmd = make_counted<command_t>(rp, prog, dims_, std::forward<Ts>(xs)...);
    cmd->enqueue();
  }

  bool handle_message(const message& msg) {
    if (!msg.types().empty() && msg.types()[0] == type_id_v<actor>) {
      auto sender = msg.get_as<actor>(0);
      if (msg.match_elements<actor, Ts...>()) {
        return unpack_and_run_wrapped(sender, msg, std::index_sequence_for<Ts...>{});
      }
      if (msg.match_elements<actor, raw_t<Ts>...>()) {
        return unpack_and_run(sender, msg, std::index_sequence_for<Ts...>{});
      }
    }
    CAF_LOG_ERROR("Message argument types do not match expected pattern");
    return false;
  }

  template <size_t... Is>
  bool unpack_and_run_wrapped(actor, const message& msg, std::index_sequence<Is...>) {
    auto tup = std::make_tuple(msg.get_as<Ts>(Is + 1)...);
    run_kernel(std::get<Is>(tup)...);
    return true;
  }

  template <size_t... Is>
  bool unpack_and_run(actor, const message& msg, std::index_sequence<Is...>) {
    auto raw = std::make_tuple(msg.get_as<raw_t<Ts>>(Is + 1)...);
    auto wrapped = std::make_tuple(Ts(std::get<Is>(raw))...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }
};

} // namespace caf::cuda

