#pragma once

#include <stdexcept>
#include <vector>
#include <map>
#include <tuple>

#include <caf/ref_counted.hpp>
#include <caf/response_promise.hpp>
#include <caf/actor_control_block.hpp>
#include <caf/message.hpp>

#include "caf/opencl/nd_range.hpp"

namespace caf::opencl {

template <class... Ts>
class command : public caf::ref_counted {
public:
  command(caf::response_promise,
          caf::strong_actor_ptr,
          std::vector<void*>,
          std::vector<void*>,
          std::vector<void*>,
          std::vector<void*>,
          std::vector<size_t>,
          caf::message,
          std::tuple<Ts...>,
          nd_range) {
    throw std::runtime_error("OpenCL support disabled: command ctor");
  }

  void enqueue() {
    throw std::runtime_error("OpenCL support disabled: command::enqueue()");
  }
};

class program : public caf::ref_counted {
public:
  program(void*, void*, void*, std::map<std::string, void*>) {
    throw std::runtime_error("OpenCL support disabled: program ctor");
  }

  ~program() override = default;
};

using program_ptr = caf::intrusive_ptr<program>;

} // namespace caf::opencl
