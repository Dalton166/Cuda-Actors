#pragma once

#include <stdexcept>
#include <functional>
#include <tuple>
#include <queue>
#include <utility>
#include <type_traits>
#include <random>
#include <climits>
#include <thread>
#include <iostream>

#include <caf/local_actor.hpp>
#include <caf/actor.hpp>
#include <caf/response_promise.hpp>
#include <caf/scheduler.hpp>
#include <caf/resumable.hpp>
#include <caf/message.hpp>

#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/command.hpp"
#include "caf/cuda/platform.hpp"
#include "caf/cuda/utility.hpp"
#include "caf/cuda/behavior.hpp"

namespace caf::cuda {


template <typename... Ts>
using defaultBehavior = AsynchronousUnicastBehavior<Ts...>;
using behavior_table_t = std::unordered_map<std::string, behavior_ptr>;

template <bool PassConfig, class... Ts>
class actor_facade : public caf::local_actor, public caf::resumable {
public:
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

  actor_facade(caf::actor_config&& cfg,
               program_ptr prog,
               nd_range nd,
               typename AbstractBehavior<Ts...>::preprocess_fn pre,
               typename AbstractBehavior<Ts...>::postprocess_fn post,
               Ts&&... xs)
    : local_actor(cfg),
      config_(std::move(cfg)),
      program_(std::move(prog)),
      dims_(nd) {
    
    // Create the defaultBehavior and store as base class pointer
    current_behavior = std::make_shared<defaultBehavior<Ts...>>(
      "default",         
      program_,
      dims_,
      std::move(pre),
      std::move(post),
      std::forward<Ts>(xs)...);
  } 

  actor_facade(caf::actor_config&& cfg,
               program_ptr prog,
               nd_range nd,
               Ts&&... xs)
    : local_actor(cfg),
      config_(std::move(cfg)),
      program_(std::move(prog)),
      dims_(nd) {
    
    // Create the defaultBehavior and store as base class pointer
    current_behavior = std::make_shared<defaultBehavior<Ts...>>(
      "default",         
      program_,
      dims_,
      actor_id,//techinically a race condition but this never gets used, should remove at some point,
      nullptr,
      nullptr,
      std::forward<Ts>(xs)...);
  } 



  ~actor_facade() {
    auto plat = platform::create();

    if (!shutdown_requested_) {
    
	    //throw std::invalid_argument("gpu actor shutting down before it was requested");
    }

    std::cout << "Deleting\n";
    plat->release_streams_for_actor(actor_id);
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

private:
  caf::actor_config config_;
  program_ptr program_;
  nd_range dims_;
  std::queue<mailbox_element_ptr> mailbox_;
  std::atomic<int> pending_promises_ = 0;
  std::atomic<bool> shutdown_requested_ = false;
  int actor_id = generate_id();
  std::atomic_flag resuming_flag_ = ATOMIC_FLAG_INIT;
  behavior_table_t behavior_table;
  behavior_ptr current_behavior;

  int generate_id() {
  
      return random_number();	  
  }




    bool handle_message(const caf::message& msg) {

	    /*
	    std::cout << "Hello?\n";
    if (!msg.types().empty() && msg.types()[0] == caf::type_id_v<caf::actor>) {

            return true;
    }*/

    if (!msg.types().empty()) {

            //check if actor should become a new state
            if (msg.match_element<become>(0)) {

                    return true;
            }

	    /*
	     * As of right now launch behavior would be nice 
	     * but it does not work as we would have to 
	     * modify the message in order to make it work with behavior
	     * will have to come back to this later, to see what can be done 
	     * about this, as per usual it is the annoying wrapper types 
	     * that make this infinitely more complicated to code
            //are we executing a behavior
            else if (msg.match_element<launch_behavior>(0)) {

		    //todo implement a handler somehow 
                    return true;
            }
	    */

            //update a behaviors state
            else if (msg.match_element<update_behavior>(0)) {

                    return true;
            }

            else {
         
		  try {
  // right here should be the place where we execute the current behavior
  execute_current_behavior(msg);
  return true;
} catch (const std::exception& e) {
  std::cerr << "[actor_facade] Exception caught during behavior execution: "
            << e.what() << std::endl;
  return false;
} catch (...) {
  std::cerr << "[actor_facade] Unknown exception caught during behavior execution"
            << std::endl;
  return false;
}
            }

            return true;
    }
    std::cout << "[WARNING], message format not recognized by actor facade, dropping message\n";

     return false;
  }
  
  void add_behavior(const behavior_ptr& behavior) {
     const std::string& key = behavior->name();
     behavior_table.emplace(key, behavior); // stored as behavior_ptr
    }

  behavior_ptr get_behavior(const std::string& name) {
  auto it = behavior_table.find(name);
  if (it != behavior_table.end()) {
    return it->second;
  }
  throw std::runtime_error("Behavior not found: " + name);
}

 
    void execute_current_behavior(caf::message msg) {

	//if the behavior is asynchronous make a response promise 
	if (current_behavior -> is_asynchronous()) {
		caf::response_promise rp = make_response_promise();
		current_behavior -> execute(msg, actor_id, rp, caf::actor_cast<caf::actor>(this));
	}	
	else {	
		current_behavior -> execute(msg, actor_id, caf::actor_cast<caf::actor>(this));
	}
    }


  template <std::size_t... Is>
  bool unpack_and_run_wrapped(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
    auto wrapped = std::make_tuple(msg.get_as<Ts>(Is + 1)...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }

  template <std::size_t... Is>
  bool unpack_and_run(caf::actor sender, const message& msg, std::index_sequence<Is...>) {
    auto unpacked = std::make_tuple(msg.get_as<raw_t<Ts>>(Is + 1)...);
    auto wrapped = std::make_tuple(Ts(std::get<Is>(unpacked))...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }


  template <std::size_t... Is>
  bool unpack_and_run_wrapped_async(const message& msg, std::index_sequence<Is...>) {
    auto wrapped = std::make_tuple(msg.get_as<Ts>(Is)...);
    run_kernel(std::get<Is>(wrapped)...);
    return true;
  }




  subtype_t subtype() const noexcept override {
    return subtype_t(0);
  }

 //schedules an actor for execution 
 resumable::resume_result resume(::caf::scheduler* sched, size_t max_throughput) override {
  if (resuming_flag_.test_and_set(std::memory_order_acquire)) {
    return resumable::resume_later;
  }

  //ensure the lock is released on exit of this method 
  auto clear_flag = caf::detail::scope_guard([this] noexcept {
    resuming_flag_.clear(std::memory_order_release);
  });

  size_t processed = 0;

  //process messages
  while (!mailbox_.empty() && processed < max_throughput) {
    auto msg = std::move(mailbox_.front());
    mailbox_.pop();

    if (!msg || !msg->content().ptr()) {
      std::cout << "[Thread " << std::this_thread::get_id()
                << "] Dropping message with no content\n";
      continue;
    }

    pending_promises_++;
    current_mailbox_element(msg.get());

    if (msg->content().match_elements<kernel_done_atom>()) {
	    std::cout << "Received kernel done\n";
      if (--pending_promises_ == 0 && shutdown_requested_) {
        quit(exit_reason::user_shutdown);
	std::cout << "Exiting\n";
        return resumable::done;
      }
      current_mailbox_element(nullptr);
      ++processed;
      continue;
    }

    if (msg->content().match_elements<exit_msg>()) {
	    std::cout << "Received exit message\n";
      auto exit = msg->content().get_as<exit_msg>(0);
      shutdown_requested_ = true;
      if (--pending_promises_ == 0) {
        quit(static_cast<exit_reason>(exit.reason.code()));
	std::cout << "Exiting\n";
        return resumable::done;
      } else {
        current_mailbox_element(nullptr);
        return resumable::resume_later;
      }
    }

    std::cout << "Calling handle message\n";
    handle_message(msg->content());
    pending_promises_--;
    current_mailbox_element(nullptr);
    ++processed;
    std::cout << "Done handling message\n";
  }

  // If there's still more work, return resume_later
  if (!mailbox_.empty())
    return resumable::resume_later;

  std::cout << "Running it back\n";
  return shutdown_requested_ ? resumable::resume_later : resumable::done;
}

  void ref_resumable() const noexcept override  {}

  void deref_resumable() const noexcept override  {}

  bool enqueue(mailbox_element_ptr what, ::caf::scheduler* sched) override {
    if (!what)
      return false;

    bool was_empty = mailbox_.empty();
    mailbox_.push(std::move(what));
    if (was_empty && sched) {
      sched->schedule(this);
    }
    return true;
  }

  void launch(::caf::scheduler* sched, bool lazy, [[maybe_unused]] bool interruptible) override {
    if (!lazy && sched) {
      sched->schedule(this);
    }
  }

  void do_unstash(mailbox_element_ptr what) override {
    if (what) {
      mailbox_.push(std::move(what));
    }
  }

  void force_close_mailbox() override  {
    while (!mailbox_.empty()) {
      mailbox_.pop();
    }
  }


  void quit(exit_reason reason) {
    //force_close_mailbox();
    
	  std::cout << "WEEEEEEEEEEEEEEEEEEEEEEEEEE\n";
	  current_mailbox_element(nullptr);
  }
};

} // namespace caf::cuda
