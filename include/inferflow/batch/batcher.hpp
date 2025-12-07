#ifndef INFERFLOW_BATCH_BATCHER_HPP_
#define INFERFLOW_BATCH_BATCHER_HPP_

#include <pybind11/pybind11.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "inferflow/batch/queue.hpp"

namespace py = pybind11;

namespace inferflow {
namespace batch {

/// @brief Batch processing metrics (C++ side with atomic counters).
struct BatchMetrics {
  std::atomic<uint64_t> total_requests{0};
  std::atomic<uint64_t> total_batches{0};
  std::atomic<uint64_t> rejected_requests{0};
  std::atomic<size_t> current_queue_size{0};
  std::atomic<size_t> current_batch_size{0};

  double total_latency_ms = 0.0;
  std::vector<double> recent_processing_times;
  std::vector<size_t> recent_batch_sizes;
};

/// @brief Batch metrics snapshot (copyable, for Python).
struct BatchMetricsSnapshot {
  uint64_t total_requests = 0;
  uint64_t total_batches = 0;
  uint64_t rejected_requests = 0;
  size_t current_queue_size = 0;
  size_t current_batch_size = 0;
  double total_latency_ms = 0.0;

  // Statistics
  double avg_batch_size = 0.0;
  double avg_latency_ms = 0.0;
};

/// @brief Request item with associated promise.
struct BatchRequest {
  py::object data;
  std::shared_ptr<std::promise<py::object>> promise;

  BatchRequest() = default;
  BatchRequest(py::object d, std::shared_ptr<std::promise<py::object>> p)
      : data(std::move(d)), promise(std::move(p)) {}
};

/// @brief High-performance C++ batcher with dynamic batching.
class DynamicBatcher {
 public:
  /// @brief Configuration for dynamic batching.
  struct Config {
    size_t min_batch_size = 1;
    size_t max_batch_size = 32;
    double max_wait_ms = 50.0;
    size_t queue_capacity = 1000;
    bool block_on_full = true;
  };

  explicit DynamicBatcher(const Config& config);
  ~DynamicBatcher();

  // Disable copy/move
  DynamicBatcher(const DynamicBatcher&) = delete;
  DynamicBatcher& operator=(const DynamicBatcher&) = delete;

  /// @brief Start the batcher with a Python runtime.
  void Start(py::object runtime);

  /// @brief Stop the batcher and cleanup resources.
  void Stop();

  /// @brief Submit an item for batched processing.
  std::shared_future<py::object> Submit(py::object item);

  /// @brief Check if batcher is running.
  bool IsRunning() const { return running_.load(std::memory_order_acquire); }

  /// @brief Get current metrics snapshot (thread-safe copy).
  BatchMetricsSnapshot GetMetrics() const;

  /// @brief Get current queue size.
  size_t QueueSize() const { return queue_.Size(); }

 private:
  void WorkerLoop();
  void ProcessBatch(std::vector<BatchRequest>& batch);

  Config config_;
  py::object runtime_;

  LockFreeQueue<BatchRequest> queue_;
  std::atomic<bool> running_{false};
  std::unique_ptr<std::thread> worker_thread_;

  BatchMetrics metrics_;
};

}  // namespace batch
}  // namespace inferflow

#endif  // INFERFLOW_BATCH_BATCHER_HPP_
