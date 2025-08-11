#pragma once
#include "caf/cuda/command.hpp"
#include "caf/cuda/program.hpp"
#include "caf/cuda/nd_range.hpp"

template <class... Ts>
class command_runner {
public:
  using command_t = command<caf::actor, Ts...>;

  // Run a command with provided args, returning enqueue()'s result.
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
};
