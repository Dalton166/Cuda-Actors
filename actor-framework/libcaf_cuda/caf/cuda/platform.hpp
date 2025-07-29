#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include <caf/intrusive_ptr.hpp>
#include <caf/actor_system.hpp>

#include "caf/ref_counted.hpp"
#include "caf/cuda/global.hpp"
#include "caf/cuda/device.hpp"
#include "caf/cuda/scheduler.hpp"

namespace caf::cuda {

class CAF_CUDA_EXPORT platform : public ref_counted {
public:
  friend class program;
  template <class T, class... Ts>
  friend intrusive_ptr<T> caf::make_counted(Ts&&...);

  static platform_ptr create();

  const std::string& name() const;
  const std::string& vendor() const;
  const std::string& version() const;
  const std::vector<device_ptr>& devices() const;

  device_ptr getDevice(int id);
  scheduler* get_scheduler();
  device_ptr schedule(int actor_id);
  void release_streams_for_actor(int actor_id);

private:
  platform();
  ~platform();

  std::string name_;
  std::string vendor_;
  std::string version_;
  std::vector<device_ptr> devices_;
  std::vector<CUcontext> contexts_;
  std::unique_ptr<scheduler> scheduler_;
};

// Intrusive pointer hooks
inline void intrusive_ptr_add_ref(platform* p) { p->ref(); }
inline void intrusive_ptr_release(platform* p) { p->deref(); }

} // namespace caf::cuda

