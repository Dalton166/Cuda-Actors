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
/// Keeps track of the total streams and limits growth to max_size.
class StreamPool {
public:
  /// Construct pool with a given CUDA context and initial number of streams.
  /// @param ctx CUDA context
  /// @param initial_size Number of pre-created streams
  /// @param max_size Maximum allowed streams
  explicit StreamPool(CUcontext ctx, size_t initial_size = 32, size_t max_size = 500);

  ~StreamPool();

  /// Acquire a new stream from the pool.
  /// Creates new stream if under max_size, otherwise reuses.
  CUstream acquire();

  /// Return a stream to the pool.
  void release(CUstream s);

private:
  CUstream create_stream();

  CUcontext ctx_;
  std::queue<CUstream> available_streams_;
  size_t max_size_;
  size_t total_streams_ = 0;
  size_t in_use_ = 0;
  std::mutex pool_mutex_;
};

/// Per-device stream manager. Assign

