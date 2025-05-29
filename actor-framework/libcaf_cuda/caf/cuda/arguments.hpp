#pragma once

#include <functional>
#include <optional>
#include <type_traits>

#include <caf/message.hpp>

namespace caf::cuda {

// Tags
struct val {};
struct mref {};
struct hidden {};

// in, in_out, out, scratch, local, priv stubs:

template <class Arg, class Tag = val>
struct in {
  using arg_type = Arg;
};

template <class Arg, class TagIn = val, class TagOut = val>
struct in_out {
  using arg_type = Arg;
};

template <class Arg, class Tag = val>
struct out {
  using arg_type = Arg;
  out() = default;
  template <class F>
  out(F) {}
};

template <class Arg>
struct scratch {
  using arg_type = Arg;
  scratch() = default;
  template <class F>
  scratch(F) {}
};

template <class Arg>
struct local {
  using arg_type = Arg;
  local(size_t = 0) {}
  template <class F>
  local(size_t, F) {}
};

template <class Arg, class Tag = hidden>
struct priv {
  using arg_type = Arg;
  priv() = default;
  explicit priv(Arg) {}
  template <class F>
  priv(Arg, F) {}
};

} // namespace caf::cuda
