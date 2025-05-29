#pragma once

#include <stdexcept>
#include <functional>
#include <tuple>
#include <iostream>

#include <caf/local_actor.hpp>
#include <caf/actor.hpp>
#include <caf/response_promise.hpp>
#include <caf/logger.hpp>
#include <caf/raise_error.hpp>
#include "caf/opencl/nd_range.hpp"
#include "caf/opencl/global.hpp"

namespace caf::opencl {

template <bool PassConfig, class... Ts>
class actor_facade : public caf::local_actor {
public:
  static caf::actor create(...) {
    throw std::runtime_error("OpenCL support disabled: actor_facade::create()");
  }

  static caf::actor create(
    caf::actor_system& sys,
    caf::actor_config&& actor_conf,
    Ts&&... xs
  ) {
    std::cout << "If you are seeing this that means that an actor has successfully spawned and was created\n";
    return make_actor<actor_facade, actor>(
      sys.next_actor_id(),
      sys.node(),
      &sys,
      std::move(actor_conf),
      std::forward<Ts>(xs)...);
  }

  actor_facade(caf::actor_config& cfg) : caf::local_actor(cfg) {
    throw std::runtime_error("OpenCL support disabled: actor_facade ctor");
  }
 actor_facade(caf::actor_config&& cfg, Ts&&... xs)
    : local_actor(config_)           // pass member variable by lvalue ref
    , config_(std::move(cfg))        // move from rvalue into member variable
    //, other_members_(std::forward<Ts>(xs))...
  {
    // constructor body
    std::cout << "If you are seeing this that means that an actor has successfully spawned and was created\n";
    print_args(std::forward<Ts>(xs)...);
  }

  // === Required pure virtual function implementations ===

  bool enqueue(mailbox_element_ptr, scheduler*) override {
    throw std::runtime_error("enqueue() not implemented yet");
  }

  void launch(scheduler*, bool, bool) override {
    throw std::runtime_error("launch() not implemented yet");
  }

  void do_unstash(mailbox_element_ptr) override {
    throw std::runtime_error("do_unstash() not implemented yet");
  }

  void force_close_mailbox() override {
  throw std::runtime_error("force_close_mailbox() not implemented yet");
}

private:
  
   caf::actor_config config_;

  void print_args() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_args(T&& first, Rest&&... rest) {
    std::cout << first << "\n";
    print_args(std::forward<Rest>(rest)...);
  }
  

protected:
  actor_facade() = default; // Protected default constructor for inheritance
};


} // namespace caf::opencl

