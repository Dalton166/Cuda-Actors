/* stream_pool.hpp */
#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <cuda.h>
#include <stdexcept>

namespace caf::cuda {

/// Thread-safe pool of CUDA streams.
/// Allocation table is owned externally by device.
class StreamPool {
public:
  /// Construct pool with a given number of pre-created streams.
  explicit StreamPool(size_t initial_size = 32) {
    for (size_t i = 0; i < initial_size; ++i) {
      available_streams_.push(create_stream());
    }
  }

  ~StreamPool() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    while (!available_streams_.empty()) {
      auto s = available_streams_.front();
      available_streams_.pop();
      cuStreamDestroy(s);
    }
  }

  /// Acquire a new stream from the pool. Expands if necessary.
  CUstream acquire() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (available_streams_.empty()) {
      return create_stream();
    }
    CUstream s = available_streams_.front();
    available_streams_.pop();
    return s;
  }

  /// Return a stream to the pool.
  void release(CUstream s) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_streams_.push(s);
  }

private:
  CUstream create_stream() {
    CUstream s;
    if (auto err = cuStreamCreate(&s, CU_STREAM_DEFAULT)) {
      throw std::runtime_error("cuStreamCreate failed: " + std::to_string(err));
    }
    return s;
  }

  std::queue<CUstream> available_streams_;
  std::mutex pool_mutex_;
};

/// Per-device stream manager. Tracks assigned streams for actor IDs.
class DeviceStreamTable {
public:
  explicit DeviceStreamTable(size_t pool_size = 32)
      : pool_(pool_size) {}

  CUstream get_stream(int actor_id) {
    {
      std::shared_lock lock(table_mutex_);
      auto it = table_.find(actor_id);
      if (it != table_.end()) {
        return it->second;
      }
    }

    std::unique_lock lock(table_mutex_);
    CUstream s = pool_.acquire();
    table_[actor_id] = s;
    return s;
  }

  void release_stream(int actor_id) {
    std::unique_lock lock(table_mutex_);
    auto it = table_.find(actor_id);
    if (it != table_.end()) {
      pool_.release(it->second);
      table_.erase(it);
    }
  }

private:
  StreamPool pool_;
  std::unordered_map<int, CUstream> table_;
  std::shared_mutex table_mutex_;
};

} // namespace caf::cuda

