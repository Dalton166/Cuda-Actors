#pragma once
#include "caf/cuda/command.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"
#include "caf/cuda/platform.hpp"


namespace caf::cuda {

template <class... Ts>
class command_runner {
public:
  using command_t = command<caf::actor, Ts...>;
  using base_command_t = base_command<caf::actor, Ts...>;

  // ----------------------------------------
  // Synchronous versions
  // ----------------------------------------

  // Run a command with provided args, returning enqueue()'s result.
  // this is a synchronous operation that this will technically block an actor 
  // but it does ensure that the data in the output buffers is actually in
  // the output buffers 
  template <class... Us>
  auto run(program_ptr program, nd_range dims, int actor_id, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        std::forward<Us>(xs)...);
    return cmd->enqueue();
  }

  // Same as above but allows the user to specify shared memory
  template <class... Us>
  auto run(program_ptr program, nd_range dims, int actor_id, int shared_mem, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        shared_mem,
        std::forward<Us>(xs)...);
    return cmd->enqueue();
  }

  // Same as above but allows the user to select shared memory and device number
  template <class... Us>
  auto run(program_ptr program, nd_range dims, int actor_id, int shared_mem, int device_number, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        shared_mem,
        device_number,
        std::forward<Us>(xs)...);
    return cmd->enqueue();
  }

  // ----------------------------------------
  // Asynchronous versions
  // ----------------------------------------

  // this is the asynchronous version of command, it returns a tuple of mem_ptr's
  // the data can then be copied back to host or sent to other actors
  template <class... Us>
  auto run_async(program_ptr program, nd_range dims, int actor_id, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<base_command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        std::forward<Us>(xs)...);
    return cmd->base_enqueue();
  }

  // Same as above but allows the user to specify shared memory
  template <class... Us>
  auto run_async(program_ptr program, nd_range dims, int actor_id, int shared_mem, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<base_command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        shared_mem,
        std::forward<Us>(xs)...);
    return cmd->base_enqueue();
  }

  // Same as above but allows the user to select shared memory and device number
  template <class... Us>
  auto run_async(program_ptr program, nd_range dims, int actor_id, int shared_mem, int device_number, Us&&... xs) {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<base_command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        shared_mem,
        device_number,
        std::forward<Us>(xs)...);
    return cmd->base_enqueue();
  }

  // ----------------------------------------
  // Stream release
  // ----------------------------------------

  // destroys a stream for an actor given its id
  void release_stream_for_actor(int actor_id) {
    auto plat = platform::create();
    plat->release_streams_for_actor(actor_id);
  }
};

} // namespace caf::cuda

