#include <stdexcept>
#include "caf/cuda/global.hpp"
#include "caf/cuda/command.hpp"
#include <caf/message.hpp>

namespace caf::cuda {

namespace detail {

// Traits to get raw type and wrapping function for each wrapper
template <typename Wrapper>
struct wrapper_traits;

template <typename T>
struct wrapper_traits<in<T>> {
  using raw_type = T;
  static auto wrap(const T& val) { return create_in_arg(val); }
  static auto wrap(const std::vector<T>& val) { return create_in_arg(val); }
};

template <typename T>
struct wrapper_traits<out<T>> {
  using raw_type = T;
  static auto wrap(const T& val) { return create_out_arg(val); }
  static auto wrap(const std::vector<T>& val) { return create_out_arg(val); }
};

template <typename T>
struct wrapper_traits<in_out<T>> {
  using raw_type = T;
  static auto wrap(const T& val) { return create_in_out_arg(val); }
  static auto wrap(const std::vector<T>& val) { return create_in_out_arg(val); }
};
template <typename Wrapper>
auto wrap_msg_element(const caf::message& msg, size_t index) {
  using raw_type = typename wrapper_traits<Wrapper>::raw_type;

  auto val_type = msg.type_at(index);

  // Already wrapped? Just return as-is.
  if (val_type == caf::type_id_v<Wrapper>)
    return msg.get_as<Wrapper>(index);

  // Scalar
  if (val_type == caf::type_id_v<raw_type>) {
    return wrapper_traits<Wrapper>::wrap(msg.get_as<raw_type>(index));
  }

  // Vector
  if (val_type == caf::type_id_v<std::vector<raw_type>>) {
    return wrapper_traits<Wrapper>::wrap(msg.get_as<std::vector<raw_type>>(index));
  }

  // Otherwise: unsupported type
  throw std::runtime_error("wrap_msg_element: Unexpected type at index " + std::to_string(index));
}



template <typename Tuple, size_t... Is>
caf::message tag_message_impl(const caf::message& msg, std::index_sequence<Is...>) {
  if (msg.size() != sizeof...(Is))
    throw std::runtime_error("Message size does not match argument tuple size.");

  return caf::make_message(wrap_msg_element<std::tuple_element_t<Is, Tuple>>(msg, Is)...);
}

template <typename... Args>
caf::message tag_message_with_wrappers(const caf::message& msg, const std::tuple<Args...>& args) {
  return tag_message_impl<std::tuple<Args...>>(msg, std::make_index_sequence<sizeof...(Args)>{});
}

} // namespace detail

template <class... Ts>
class AbstractBehavior {
public:
  using preprocess_fn = std::function<void(const caf::message&)>;

AbstractBehavior(std::string name,
                 program_ptr program,
                 nd_range dims,
                 int actor_id,
                 preprocess_fn preprocess,
                 std::vector<caf::actor> targets,
                 Ts&&... xs)
  : name_(std::move(name)),
    program_(std::move(program)),
    dims_(std::move(dims)),
    actor_id_(actor_id),
    preprocess_(std::move(preprocess)),
    targets_(std::move(targets)),
    args_(std::forward<Ts>(xs)...) {} 

   virtual ~AbstractBehavior() = default;

  // Main entry point with response promise
  virtual void execute(const caf::message& msg, int actor_id, caf::response_promise& rp) {
    auto tagged_msg = preprocess(msg);
    auto results = execute_command(tagged_msg, actor_id);
    postprocess(results);
    reply(results, rp);
    cleanup();
  }

  // Fire-and-forget entry point (no response promise)
  virtual void execute(const caf::message& msg, int actor_id) {
    auto tagged_msg = preprocess(msg);
    auto results = execute_command(tagged_msg, actor_id);
    postprocess(results);
    reply(results);
    cleanup();
  }

  // Virtual command execution, overridable by derived classes
  virtual std::tuple<mem_ptr<raw_t<Ts>>...> execute_command(const caf::message& msg, int actor_id) {
    return execute_command_impl(msg, actor_id, std::make_index_sequence<sizeof...(Ts)>{});
  }

  // Getters
  const std::string& name() const {
    return name_;
  }

  bool is_asynchronous() const {
    return is_asynchronous_;
  }

protected:
  // Override preprocess to return tagged message
  virtual caf::message preprocess(const caf::message& msg) {
    if (preprocess_)
      preprocess_(msg);
    return detail::tag_message_with_wrappers(msg, args_);
  }

  template <std::size_t... Is>
  std::tuple<mem_ptr<raw_t<Ts>>...> execute_command_impl(const caf::message& msg, int actor_id, std::index_sequence<Is...>) {
    auto cmd = caf::make_counted<command<caf::actor, Ts...>>(
      msg,
      program_,
      dims_,
      actor_id,
      std::get<Is>(args_)...);
   
    return  cmd->enqueue();
  }

  // Virtual hooks — override as needed
  virtual void postprocess(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/) {
    // Default no-op
  }

  virtual void reply(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/, caf::response_promise& /*rp*/) {
    // Default no-op
  }

  virtual void reply(const std::tuple<mem_ptr<raw_t<Ts>>...>& /*results*/) {
    // Default no-op
  }

  virtual void cleanup() {
    // Default no-op
  }

protected:
  std::string name_;
  bool is_asynchronous_ = false;
  program_ptr program_;
  nd_range dims_;
  int actor_id_;
  preprocess_fn preprocess_;
  std::vector<caf::actor> targets_;
  std::tuple<Ts...> args_;
  caf::response_promise rp_;
  caf::actor self_; // You might want to set this after construction or via a setter

};

template <class... Ts>
using behavior_ptr = caf::intrusive_ptr<AbstractBehavior<Ts...>>;

} // namespace caf::cuda

