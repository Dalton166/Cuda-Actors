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

namespace caf::cuda {

template <bool PassConfig, class... Ts>
class actor_facade : public caf::local_actor, public caf::resumable {
public:
  static caf::actor create(...) {
    throw std::runtime_error("CUDA support disabled: actor_facade::create()");
  }

  static caf::actor create(
    caf::actor_system& sys,
    caf::actor_config&& actor_conf,
    Ts&&... xs
  ) {
    std::cout << "Actor has successfully spawned and was created\n";
    return make_actor<actor_facade, actor>(
      sys.next_actor_id(),
      sys.node(),
      &sys,
      std::move(actor_conf),
      std::forward<Ts>(xs)...);
  }

  actor_facade(caf::actor_config& cfg) : caf::local_actor(cfg), caf::resumable() {
    throw std::runtime_error("CUDA support disabled: actor_facade ctor");
  }

  actor_facade(caf::actor_config&& cfg, Ts&&... xs)
    : local_actor(cfg),
      resumable(),
      config_(std::move(cfg)) {
    std::cout << "Actor has successfully spawned and was created\n";
    print_args(std::forward<Ts>(xs)...);
  }

  // Implement caf::resumable interface
  subtype_t subtype() const noexcept override {
    return subtype_t(0); // Placeholder: minimal type identifier
  }

  resumable::resume_result resume(scheduler*, size_t) override {
    return resumable::done; // Minimal: actor terminates immediately
  }

  void ref_resumable() const noexcept override {
    // Minimal: do nothing or call ref_counted::ref()
  }

  void deref_resumable() const noexcept override {
    // Minimal: do nothing or call ref_counted::deref()
  }

  // Implement caf::local_actor virtual methods
  bool enqueue(mailbox_element_ptr what, scheduler*) override {
    if (what) {
      mailbox_.push(std::move(what));
      return true; // Message enqueued
    }
    return false;
  }

  void launch(scheduler* where, bool lazy, bool interruptible) override {
    if (!lazy && where) {
      where->schedule(this); // Schedule the actor (works since we inherit from resumable)
    }
  }

  void do_unstash(mailbox_element_ptr what) override {
    if (what) {
      enqueue(std::move(what), nullptr); // Re-enqueue stashed message
    }
  }

  void force_close_mailbox() override {
    while (!mailbox_.empty()) {
      mailbox_.pop(); // Clear mailbox
    }
  }

private:
  caf::actor_config config_;
  std::queue<mailbox_element_ptr> mailbox_; // Mailbox for messages

  void print_args() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_args(T&& first, Rest&&... rest) {
    std::cout << first << "\n";
    print_args(std::forward<Rest>(rest)...);
  }
};

} // namespace caf::cuda
