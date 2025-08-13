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

  // ---------------------------------------------------------------------------
  // Run a command with provided args, returning enqueue()'s result.
  // This is a synchronous operation that technically blocks the actor, 
  // but ensures that the data in the output buffers is actually ready.
  // ---------------------------------------------------------------------------
  template <class... Us>
  auto run(program_ptr program,
           nd_range dims,
           int actor_id,
           Us&&... xs,
           int shared_memory = 0,
           int device_number = -1) 
  {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        std::forward<Us>(xs)...,
        shared_memory,
        device_number);
    return cmd->enqueue();
  }

  // ---------------------------------------------------------------------------
  // Asynchronous version: returns a tuple of mem_ptr's.
  // The data can then be copied back to host or sent to other actors.
  // ---------------------------------------------------------------------------
  template <class... Us>
  auto run_async(program_ptr program,
                 nd_range dims,
                 int actor_id,
                 Us&&... xs,
                 int shared_memory = 0,
                 int device_number = -1)
  {
    static_assert(sizeof...(Us) == sizeof...(Ts),
                  "Number of arguments must match Ts...");
    auto cmd = caf::make_counted<base_command_t>(
        std::move(program),
        std::move(dims),
        actor_id,
        std::forward<Us>(xs)...,
        shared_memory,
        device_number);
    return cmd->base_enqueue();
  }

  // ---------------------------------------------------------------------------
  // Destroy streams for a given actor ID
  // ---------------------------------------------------------------------------
  void release_stream_for_actor(int actor_id) {
    auto plat = platform::create();
    plat->release_streams_for_actor(actor_id);
  }
};

} // namespace caf::cuda

