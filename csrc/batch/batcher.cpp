#include "inferflow/batch/batcher.hpp"

#include <pybind11/stl.h>

#include <chrono>
#include <iostream>

namespace inferflow {
namespace batch {

DynamicBatcher::DynamicBatcher(const Config& config)
    : config_(config), queue_(config.queue_capacity) {}

DynamicBatcher::~DynamicBatcher() { Stop(); }

void DynamicBatcher::Start(py::object runtime) {
  if (running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Batcher already running");
  }

  runtime_ = runtime;
  running_.store(true, std::memory_order_release);

  worker_thread_ =
      std::make_unique<std::thread>(&DynamicBatcher::WorkerLoop, this);

  std::cout << "[C++ Batcher] Started with config: "
            << "batch_size=[" << config_.min_batch_size << ", "
            << config_.max_batch_size << "], "
            << "max_wait=" << config_.max_wait_ms << "ms, "
            << "queue_capacity=" << config_.queue_capacity << std::endl;
}

void DynamicBatcher::Stop() {
  if (!running_.load(std::memory_order_acquire)) {
    return;
  }

  running_.store(false, std::memory_order_release);

  if (worker_thread_ && worker_thread_->joinable()) {
    worker_thread_->join();
  }

  std::cout << "[C++ Batcher] Stopped. Processed "
            << metrics_.total_requests.load() << " requests in "
            << metrics_.total_batches.load() << " batches." << std::endl;
}

std::shared_future<py::object> DynamicBatcher::Submit(py::object item) {
  if (!running_.load(std::memory_order_acquire)) {
    throw std::runtime_error("Batcher not running.Call Start() first.");
  }

  auto promise = std::make_shared<std::promise<py::object>>();
  auto future = promise->get_future().share();

  BatchRequest request(std::move(item), std::move(promise));

  // Try to push to queue
  if (!queue_.TryPush(std::move(request))) {
    if (config_.block_on_full) {
      // Busy-wait with exponential backoff
      using namespace std::chrono;
      auto backoff = microseconds(10);
      const auto max_backoff = microseconds(1000);

      while (!queue_.TryPush(std::move(request))) {
        std::this_thread::sleep_for(backoff);
        backoff = std::min(backoff * 2, max_backoff);
      }
    } else {
      metrics_.rejected_requests.fetch_add(1, std::memory_order_relaxed);
      throw std::runtime_error("Queue is full");
    }
  }

  metrics_.current_queue_size.store(queue_.Size(), std::memory_order_relaxed);
  return future;
}

BatchMetricsSnapshot DynamicBatcher::GetMetrics() const {
  BatchMetricsSnapshot snapshot;

  // Atomically load current values
  snapshot.total_requests =
      metrics_.total_requests.load(std::memory_order_relaxed);
  snapshot.total_batches =
      metrics_.total_batches.load(std::memory_order_relaxed);
  snapshot.rejected_requests =
      metrics_.rejected_requests.load(std::memory_order_relaxed);
  snapshot.current_queue_size =
      metrics_.current_queue_size.load(std::memory_order_relaxed);
  snapshot.current_batch_size =
      metrics_.current_batch_size.load(std::memory_order_relaxed);

  snapshot.total_latency_ms = metrics_.total_latency_ms;

  // Calculate statistics
  if (!metrics_.recent_batch_sizes.empty()) {
    uint64_t sum = 0;
    for (size_t size : metrics_.recent_batch_sizes) {
      sum += size;
    }
    snapshot.avg_batch_size =
        static_cast<double>(sum) / metrics_.recent_batch_sizes.size();
  }

  if (snapshot.total_requests > 0) {
    snapshot.avg_latency_ms =
        snapshot.total_latency_ms / snapshot.total_requests;
  }

  return snapshot;
}

void DynamicBatcher::WorkerLoop() {
  py::gil_scoped_acquire acquire;  // Acquire GIL for this thread

  while (running_.load(std::memory_order_acquire)) {
    std::vector<BatchRequest> batch;
    batch.reserve(config_.max_batch_size);

    // Wait for first item
    std::optional<BatchRequest> first_item;

    while (running_.load(std::memory_order_acquire)) {
      first_item = queue_.TryPop();
      if (first_item) {
        batch.push_back(std::move(*first_item));
        break;
      }

      // Sleep briefly to avoid busy-waiting
      py::gil_scoped_release release;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (batch.empty()) {
      continue;  // No items, check running flag and retry
    }

    // Collect more items up to max_batch_size or max_wait
    auto batch_start = std::chrono::steady_clock::now();
    const auto max_wait =
        std::chrono::duration<double, std::milli>(config_.max_wait_ms);

    while (batch.size() < config_.max_batch_size) {
      auto elapsed = std::chrono::steady_clock::now() - batch_start;
      if (elapsed >= max_wait) {
        break;
      }

      auto item = queue_.TryPop();
      if (item) {
        batch.push_back(std::move(*item));
      } else {
        // Brief sleep if queue is empty
        py::gil_scoped_release release;
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }

    // Process the batch
    if (!batch.empty()) {
      ProcessBatch(batch);
    }
  }

  py::gil_scoped_release release;
}

void DynamicBatcher::ProcessBatch(std::vector<BatchRequest>& batch) {
  const size_t batch_size = batch.size();

  auto start_time = std::chrono::steady_clock::now();

  try {
    // Collect batch items
    py::list batch_items;
    for (const auto& req : batch) {
      batch_items.append(req.data);
    }

    // Call runtime.infer_batch(items)
    py::object infer_batch_method = runtime_.attr("infer_batch");
    py::object results_obj = infer_batch_method(batch_items);

    // Convert to list
    py::list results = results_obj.cast<py::list>();

    // Distribute results to promises
    for (size_t i = 0; i < batch_size; ++i) {
      batch[i].promise->set_value(results[i]);
    }

    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto processing_time =
        std::chrono::duration<double>(end_time - start_time).count();

    metrics_.total_requests.fetch_add(batch_size, std::memory_order_relaxed);
    metrics_.total_batches.fetch_add(1, std::memory_order_relaxed);
    metrics_.current_batch_size.store(batch_size, std::memory_order_relaxed);
    metrics_.current_queue_size.store(queue_.Size(), std::memory_order_relaxed);

  } catch (const std::exception& e) {
    // Set exception on all promises
    for (auto& req : batch) {
      try {
        req.promise->set_exception(std::current_exception());
      } catch (...) {
        // Promise already satisfied, ignore
      }
    }

    std::cerr << "[C++ Batcher] Batch processing failed: " << e.what()
              << std::endl;
  }
}

}  // namespace batch
}  // namespace inferflow
