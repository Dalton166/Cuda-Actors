#pragma once

#include "global.hpp"
#include "behavior.hpp"

namespace caf::cuda {

//represents a behavior where the actor replies to only 1 sender and 
//does so using response promises
template <class... Ts>
class AsynchronousUnicastBehavior : public AbstractBehavior<Ts...> {
public:
  using super = AbstractBehavior<Ts...>;
  using preprocess_fn = typename super::preprocess_fn;

  AsynchronousUnicastBehavior(std::string name,
                              program_ptr program,
                              nd_range dims,
                              int actor_id,
                              preprocess_fn preprocess,
                              caf::actor target,
                              Ts&&... xs)
    : super(std::move(name),
            std::move(program),
            std::move(dims),
            actor_id,
            std::move(preprocess),
            std::vector<caf::actor>{std::move(target)},
            std::forward<Ts>(xs)...) {
    this->is_asynchronous_ = true;
  }

  bool is_asynchronous() const override {
    return true;
  }

protected:
  void execute(const caf::message& msg,
               int actor_id,
               caf::response_promise& rp,
               caf::actor self) override {
    super::execute(msg, actor_id, rp, self);
    anon_mail(kernel_done_atom_v).send(self); //notify actor that we are done execution 
  }

  void reply(const std::tuple<mem_ptr<raw_t<Ts>>...>& results,
             caf::response_promise& rp) override {
    auto output_buffers = collect_output_buffers_helper(results);
    rp.deliver(std::move(output_buffers));

    // Cleanup: reset device buffers
    // this is supposed to be in cleanup but for now it is in reply 
    for_each_tuple(results, [](auto& mem) {
      if (mem)
        mem->reset();
    });
  }
};

template <typename... Ts>
using defaultBehavior = AsynchronousUnicastBehavior<Ts...>;


} // namespace caf::cuda

