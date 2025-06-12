// caf/cuda/types.hpp
#pragma once

#include <caf/intrusive_ptr.hpp>

namespace caf::cuda {
class device;
using device_ptr = caf::intrusive_ptr<device>;

class platform;
using platform_ptr = caf::intrusive_ptr<platform>;

class program;
using program_ptr = caf::intrusive_ptr<program>;

template <class T>
class mem_ref;
template <class T>
using mem_ptr = caf::intrusive_ptr<mem_ref<T>>;
} // namespace caf::cuda
