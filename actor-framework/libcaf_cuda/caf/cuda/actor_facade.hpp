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
    std::cout << "Actor has successfully spawned and was created\n";
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
  std::cout << "Actor has successfully spawned and was created\n";
  return caf::make_actor<actor_facade<PassConfig, std::decay_t<Ts>...>, caf::actor>(
    sys->next_actor_id(),
    sys->node(),
    sys,
    std::move(actor_conf),
    std::move(program),
    std::move(dims),
    std::forward<Ts>(xs)...
  );
}
  /*
  actor_facade(caf::actor_config& cfg) : caf::local_actor(cfg), caf::resumable() {
    throw std::runtime_error("CUDA support disabled: actor_facade ctor");
  }
	*/
actor_facade(caf::actor_config&& cfg, program_ptr prog, nd_range nd, Ts&&... xs)
  : 
    local_actor(cfg),
    config_(std::move(cfg))
  , program_(std::move(prog))
  , dims_(std::move(nd)) // <-- initialize your nd_range member here
{
    std::cout << "Actor has successfully spawned and was created\n";
}


//template <class... Ts>
void create_command(program_ptr program, Ts&&... xs) {
    caf::response_promise rp = make_response_promise();

    using command_t = command<caf::actor, std::decay_t<Ts>...>;
    auto cmd = make_counted<command_t>(rp, program,dims_, std::forward<Ts>(xs)...);
    cmd -> enqueue(); //launches the kernel
}


//temperary workaround for the fact that actor_facade cannot communicate with other actors so far be sure to use in, in_out or out types otherwise this will crash 
//template <class... Ts>
void run_kernel(Ts&... xs) {
    create_command(program_, std::forward<Ts>(xs)...);
}

private:
  caf::actor_config config_;
  std::queue<mailbox_element_ptr> mailbox_; // Mailbox for messages
  program_ptr program_;
  nd_range dims_;

  void print_args() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_args(T&& first, Rest&&... rest) {
    std::cout << first << "\n";
    print_args(std::forward<Rest>(rest)...);
  }

//-------------------------Basic implementation, will need to implement fully later
  bool handle_message(const message& msg) {
    if (!msg.match_elements<raw_t<Ts>...>()) {
      std::cerr << "Message argument types do not match wrapper types!\n";
      return false;
    }

    // Extract raw values from message
    // Need index sequence
    return unpack_and_run(msg, std::index_sequence_for<Ts...>{});
  }

template <std::size_t... Is>
bool unpack_and_run(const message& msg, std::index_sequence<Is...>) {
  // Extract raw values (e.g., char, int)
  auto unpacked = std::make_tuple(msg.get_as<raw_t<Ts>>(Is)...);

  std::cout << "Hi kids\n";
  // Optional: no need to check has_value() if get_as returns raw values
  // Assume msg.match_elements<raw_t<Ts>...>() already checked

  // Wrap raw values into wrapper types
  auto wrapped = std::make_tuple(
    Ts(std::get<Is>(unpacked))...  // Requires constructor Ts(raw_t<Ts>)
  );

  // For demo, print contents:
  print_wrapped(std::get<Is>(wrapped)...);

  return true;
}
  void print_wrapped() {
    std::cout << "(no args)\n";
  }

  template <typename T, typename... Rest>
  void print_wrapped(T&& first, Rest&&... rest) {
    std::cout << "Buffer contents: ";
    for (auto& v : first.buffer)
      std::cout << v << ' ';
    std::cout << '\n';
    print_wrapped(std::forward<Rest>(rest)...);
  }
// Implement caf::resumable interface
  subtype_t subtype() const noexcept override {
    return subtype_t(0); // Placeholder: minimal type identifier
  }

  resumable::resume_result resume(scheduler*, size_t) override {
    while (!mailbox_.empty()) {
      auto& msg = mailbox_.front();
      if (msg && msg->content()) {
        handle_message(msg->content());
      }
      mailbox_.pop();
    }
      std::cout << "Hello world\n";
    return resumable::done;
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



};

} // namespace caf::cuda
