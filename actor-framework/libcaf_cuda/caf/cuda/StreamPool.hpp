#pragma once

#include <vector>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <cuda.h>
#include <stdexcept>
#include <iostream>

namespace caf::cuda {

/// Thread-safe pool of CUDA streams.
/// Allocation table is owned externally by device.
class StreamPool {
public:
  /// Construct pool with a given CUDA context and number of pre-created streams.
  explicit StreamPool(CUcontext ctx, size_t initial_size = 32)
    : ctx_(ctx) {
    for (size_t i = 0; i < initial_size; ++i) {
      available_streams_.push(create_stream());
    }
  }

  ~StreamPool() {
    // No need to destroy streams explicitly since destroying context cleans up
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

  //creates a new stream to add into the stream pool
  CUstream create_stream() {
    // Push context to the current thread
    CUresult err = cuCtxPushCurrent(ctx_);
    if (err != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(err, &err_str);
      throw std::runtime_error(std::string("cuCtxPushCurrent failed: ") + (err_str ? err_str : "unknown error"));
    }

    CUstream s;
    err = cuStreamCreate(&s, CU_STREAM_DEFAULT);
    
    // Pop context right after creating stream
    CUcontext popped_ctx;
    CUresult pop_err = cuCtxPopCurrent(&popped_ctx);
    if (pop_err != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(pop_err, &err_str);
      throw std::runtime_error(std::string("cuCtxPopCurrent failed: ") + (err_str ? err_str : "unknown error"));
    }

    if (err != CUDA_SUCCESS) {
      const char* err_str = nullptr;
      cuGetErrorString(err, &err_str);
      throw std::runtime_error(std::string("cuStreamCreate failed: ") + (err_str ? err_str : "unknown error"));
    }
    return s;
  }

  CUcontext ctx_;
  std::queue<CUstream> available_streams_;
  std::mutex pool_mutex_;
};

/// Per-device stream manager. Tracks assigned streams for actor IDs.
class DeviceStreamTable {
public:
  explicit DeviceStreamTable(CUcontext ctx, size_t pool_size = 32)
      : pool_(ctx, pool_size) {}

  //gets a cuStream given an actor id 
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

  //releases a stream back into the pool
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

