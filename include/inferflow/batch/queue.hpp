// Copyright 2025 BoChenSHEN
// Licensed under the MIT License

#ifndef INFERFLOW_BATCH_LOCK_FREE_QUEUE_HPP_
#define INFERFLOW_BATCH_LOCK_FREE_QUEUE_HPP_

#include <atomic>
#include <memory>
#include <optional>
#include <vector>

namespace inferflow {
namespace batch {

/// @brief Lock-free SPSC (Single Producer Single Consumer) queue.
///
/// This queue is optimized for high-throughput batching scenarios.
/// It uses atomic operations to avoid mutex contention.
template <typename T>
class LockFreeQueue {
 public:
  explicit LockFreeQueue(size_t capacity)
      : capacity_(capacity), buffer_(capacity), head_(0), tail_(0) {}

  /// @brief Try to push an item into the queue.
  ///
  /// @param item Item to push
  /// @return true if successful, false if queue is full
  bool TryPush(T item) {
    const size_t tail = tail_.load(std::memory_order_relaxed);
    const size_t next_tail = (tail + 1) % capacity_;

    if (next_tail == head_.load(std::memory_order_acquire)) {
      return false;  // Queue is full
    }

    buffer_[tail] = std::move(item);
    tail_.store(next_tail, std::memory_order_release);
    return true;
  }

  /// @brief Try to pop an item from the queue.
  ///
  /// @return Item if available, std::nullopt if queue is empty
  std::optional<T> TryPop() {
    const size_t head = head_.load(std::memory_order_relaxed);

    if (head == tail_.load(std::memory_order_acquire)) {
      return std::nullopt;  // Queue is empty
    }

    T item = std::move(buffer_[head]);
    head_.store((head + 1) % capacity_, std::memory_order_release);
    return item;
  }

  /// @brief Get current queue size (approximate).
  size_t Size() const {
    const size_t head = head_.load(std::memory_order_acquire);
    const size_t tail = tail_.load(std::memory_order_acquire);
    return (tail + capacity_ - head) % capacity_;
  }

  /// @brief Check if queue is empty.
  bool IsEmpty() const {
    return head_.load(std::memory_order_acquire) ==
           tail_.load(std::memory_order_acquire);
  }

 private:
  const size_t capacity_;
  std::vector<T> buffer_;
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
};

}  // namespace batch
}  // namespace inferflow

#endif  // INFERFLOW_BATCH_LOCK_FREE_QUEUE_HPP_
