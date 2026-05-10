#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <chessmoe/eval/evaluator.h>

namespace chessmoe::inference {

struct AsyncBatchingEvaluatorConfig {
  std::size_t fixed_batch_size{64};
  std::size_t max_pending_requests{4096};
  std::chrono::milliseconds flush_timeout{2};
  bool pad_to_fixed_batch{true};
};

struct AsyncBatchingMetrics {
  std::uint64_t positions_evaluated{0};
  std::uint64_t batches_evaluated{0};
  std::uint64_t padded_positions{0};
  std::uint64_t max_queue_depth{0};
  double total_inference_latency_ms{0.0};
  std::vector<std::uint64_t> batch_size_histogram;
  std::vector<std::uint64_t> valid_batch_size_histogram;

  [[nodiscard]] double average_inference_latency_ms() const;
};

class AsyncBatchingEvaluator final : public eval::ISinglePositionEvaluator {
 public:
  AsyncBatchingEvaluator(eval::IBatchEvaluator& backend,
                         AsyncBatchingEvaluatorConfig config);
  ~AsyncBatchingEvaluator() override;

  AsyncBatchingEvaluator(const AsyncBatchingEvaluator&) = delete;
  AsyncBatchingEvaluator& operator=(const AsyncBatchingEvaluator&) = delete;

  eval::EvaluationResult evaluate(const eval::EvaluationRequest& request) override;

  [[nodiscard]] AsyncBatchingMetrics metrics_snapshot() const;

 private:
  struct PendingRequest {
    eval::EvaluationRequest request;
    std::promise<eval::EvaluationResult> promise;
  };

  void worker_loop();
  void process_batch(std::vector<std::shared_ptr<PendingRequest>> pending);
  void record_batch_metrics(std::size_t valid_count,
                            std::size_t padded_count,
                            double latency_ms);
  void stop();

  eval::IBatchEvaluator& backend_;
  AsyncBatchingEvaluatorConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable not_empty_;
  std::condition_variable not_full_;
  std::deque<std::shared_ptr<PendingRequest>> queue_;
  bool stopping_{false};
  std::thread worker_;

  mutable std::mutex metrics_mutex_;
  AsyncBatchingMetrics metrics_;
};

}  // namespace chessmoe::inference
