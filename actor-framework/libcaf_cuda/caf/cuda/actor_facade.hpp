#pragma once

#include <stdexcept>
#include <functional>
#include <tuple>
#include <queue>
#include <utility>
#include <type_traits>

#include <caf/local_actor.hpp>
#include <caf/actor.hpp>
#include <caf/response_promise.hpp>
#include <caf/scheduler.hpp>
#include <caf/resumable.hpp>

#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/command.hpp"
#include "caf/cuda/utility.hpp"
#include <random>
#include <climits>

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
      dims_(nd) {
  }

  ~actor_facade() {
    program_->get_device()->release_stream_for_actor(actor_id);
  }

  void create_command(program_ptr program, Ts&&... xs) {
    using command_t = command<caf::actor, raw_t<Ts>...>;
    auto cmd = make_counted<command_t>(
      make_response_promise(),
      caf::actor_cast<caf::actor>(this),
      program,
      dims_,
      actor_id,
      std::forward<Ts>(xs)...);
    cmd->enqueue();
  }

  void run_kernel(Ts&... xs) {
    create_command(program_, std::forward<Ts>(xs)...);
  }

  //again this relies on the knowledge that launch kernel is synchronous
  //it should not be, this should be deleted eventually but response promise is 
  //causing too much bugs
  //
  void run_kernel_synchronous(caf::actor sender, Ts&... xs) {
    using command_t = command<caf::actor, raw_t<Ts>...>;
    auto cmd = make_counted<command_t>(
      sender,
      caf::actor_cast<caf::actor>(this),
      program_,
      dims_,
      actor_id,
      std::forward<Ts>(xs)...);
    cmd->enqueue();
  }

private:
  caf::actor_config config_;
  program_ptr program_;
  nd_range dims_;
  std::queue<mailbox_element_ptr> mailbox_;
  std::atomic<int> pending_promises_ = 0;
  std::atomic<bool> shutdown_requested_ = false;
  int actor_id = generate_id();

  int generate_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> distrib(INT_MIN, INT_MAX);
    return distrib(gen);
  }

  bool handle_message(const message& msg) {
    if (!msg.types().empty() && msg.types()[0] == caf::type_id_v<caf::actor>) {
      auto sender = msg.get_as<caf::actor>(0);
      if (msg.match_elements<caf::actor, Ts...>()) {
        return unpack_and_run_wrapped(sender, msg, std::index_sequence_for<Ts...>{});
      }
      if (msg.match_elements<caf::actor, raw_t<Ts>...>()) {
        return unpack_and_run(sender, msg, std::index_sequence_for<Ts...>{});
      }
    }
    std::cout << "[WARNING], message format not recognized by actor facade, dropping message\n";
    
     return false;
  }

  template <std::size_t... Is>
  bool unpack_and_run_wrapped(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
    auto wrapped = std::make_tuple(msg.get_as<Ts>(Is + 1)...);
    run_kernel_synchronous(sender, std::get<Is>(wrapped)...);
    return true;
  }

  template <std::size_t... Is>
  bool unpack_and_run(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
    auto unpacked = std::make_tuple(msg.get_as<raw_t<Ts>>(Is + 1)...);
    auto wrapped = std::make_tuple(Ts(std::get<Is>(unpacked))...);
    run_kernel_synchronous(sender, std::get<Is>(wrapped)...);
    return true;
  }

  subtype_t subtype() const noexcept override {
    return subtype_t(0);
  }

  resumable::resume_result resume(scheduler* sched, size_t) override {
    while (!mailbox_.empty()) {
      
      std::cout << "Actor with id " << actor_id << " is begginning to process a message.\n";
      auto msg = std::move(mailbox_.front());
      mailbox_.pop();


      if (shutdown_requested_) {
      
	      std::cout << "Warning actor with id " << actor_id << " Is processing message after shutdown has been requested\n";
      
      }

      if (!msg || !msg->content().ptr()) { 
	      std::cout << "Warning gpu actor with id " << actor_id << "Received a message with no content, dropping message\n";
	      continue;
      }


      pending_promises_++;
      current_mailbox_element(msg.get());
      if (msg->content().match_elements<kernel_done_atom>()) {
        if (--pending_promises_ == 0 && shutdown_requested_) {
          quit(exit_reason::user_shutdown);
          return resumable::done;
        }
        current_mailbox_element(nullptr);
        continue;
      }

      if (msg->content().match_elements<exit_msg>()) {
        auto exit = msg->content().get_as<exit_msg>(0);
        shutdown_requested_ = true;
        if (--pending_promises_ == 0) {
          quit(static_cast<exit_reason>(exit.reason.code()));
          return resumable::done;
        } else {
          current_mailbox_element(nullptr);
          return resumable::resume_later;
        }
      }

      handle_message(msg->content());
      pending_promises_--;
      current_mailbox_element(nullptr);
    }

    return shutdown_requested_ ? resumable::resume_later : resumable::done;
  }

  void ref_resumable() const noexcept override {}

  void deref_resumable() const noexcept override {}

  bool enqueue(mailbox_element_ptr what, scheduler* sched) override {
    if (!what)
      return false;

    bool was_empty = mailbox_.empty();
    mailbox_.push(std::move(what));
    if (was_empty && sched) {
      sched->schedule(this);
    }
    return true;
  }

  void launch(scheduler* sched, bool lazy, [[maybe_unused]] bool interruptible) override {
    if (!lazy && sched) {
      sched->schedule(this);
    }
  }

  void do_unstash(mailbox_element_ptr what) override {
    if (what) {
      mailbox_.push(std::move(what));
    }
  }

  void force_close_mailbox() override {
    std::cout << "Actor with id : " << actor_id  << " force mailbox called\n";
    while (!mailbox_.empty()) {
      mailbox_.pop();
    }
  }

  void quit(exit_reason reason) {
    force_close_mailbox();
    current_mailbox_element(nullptr);
  }
};

} // namespace caf::cuda
