#include <chessmoe/inference/async_batching_evaluator.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace chessmoe::inference {

namespace {

void validate_config(const AsyncBatchingEvaluatorConfig& config) {
  if (config.fixed_batch_size == 0) {
    throw std::invalid_argument("fixed batch size must be positive");
  }
  if (config.max_pending_requests == 0) {
    throw std::invalid_argument("max pending requests must be positive");
  }
  if (config.flush_timeout.count() < 0) {
    throw std::invalid_argument("flush timeout must not be negative");
  }
}

void ensure_histogram_size(std::vector<std::uint64_t>& histogram,
                           std::size_t batch_size) {
  if (histogram.size() <= batch_size) {
    histogram.resize(batch_size + 1, 0);
  }
}

}  // namespace

double AsyncBatchingMetrics::average_inference_latency_ms() const {
  return batches_evaluated == 0
             ? 0.0
             : total_inference_latency_ms /
                   static_cast<double>(batches_evaluated);
}

AsyncBatchingEvaluator::AsyncBatchingEvaluator(
    eval::IBatchEvaluator& backend, AsyncBatchingEvaluatorConfig config)
    : backend_(backend), config_(config) {
  validate_config(config_);
  metrics_.batch_size_histogram.resize(config_.fixed_batch_size + 1, 0);
  metrics_.valid_batch_size_histogram.resize(config_.fixed_batch_size + 1, 0);
  worker_ = std::thread([this] { worker_loop(); });
}

AsyncBatchingEvaluator::~AsyncBatchingEvaluator() {
  stop();
}

eval::EvaluationResult AsyncBatchingEvaluator::evaluate(
    const eval::EvaluationRequest& request) {
  auto pending = std::make_shared<PendingRequest>();
  pending->request = request;
  auto future = pending->promise.get_future();

  {
    std::unique_lock lock(mutex_);
    not_full_.wait(lock, [&] {
      return stopping_ || queue_.size() < config_.max_pending_requests;
    });
    if (stopping_) {
      throw std::runtime_error("async evaluator is stopped");
    }

    queue_.push_back(pending);
    {
      std::lock_guard metrics_lock(metrics_mutex_);
      metrics_.max_queue_depth = std::max<std::uint64_t>(
          metrics_.max_queue_depth, static_cast<std::uint64_t>(queue_.size()));
    }
  }
  not_empty_.notify_one();

  return future.get();
}

AsyncBatchingMetrics AsyncBatchingEvaluator::metrics_snapshot() const {
  std::lock_guard lock(metrics_mutex_);
  return metrics_;
}

void AsyncBatchingEvaluator::worker_loop() {
  for (;;) {
    std::vector<std::shared_ptr<PendingRequest>> pending;

    {
      std::unique_lock lock(mutex_);
      not_empty_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
      if (queue_.empty() && stopping_) {
        return;
      }

      const auto deadline = std::chrono::steady_clock::now() + config_.flush_timeout;
      while (!stopping_ && queue_.size() < config_.fixed_batch_size) {
        if (not_empty_.wait_until(lock, deadline) == std::cv_status::timeout) {
          break;
        }
      }

      const auto take_count =
          std::min(queue_.size(), config_.fixed_batch_size);
      pending.reserve(take_count);
      for (std::size_t i = 0; i < take_count; ++i) {
        pending.push_back(std::move(queue_.front()));
        queue_.pop_front();
      }
    }

    not_full_.notify_all();
    if (!pending.empty()) {
      process_batch(std::move(pending));
    }
  }
}

void AsyncBatchingEvaluator::process_batch(
    std::vector<std::shared_ptr<PendingRequest>> pending) {
  std::vector<eval::EvaluationRequest> requests;
  requests.reserve(config_.fixed_batch_size);
  for (const auto& item : pending) {
    requests.push_back(item->request);
  }

  const auto valid_count = requests.size();
  if (config_.pad_to_fixed_batch && !requests.empty()) {
    while (requests.size() < config_.fixed_batch_size) {
      requests.push_back(requests.back());
    }
  }
  const auto padded_count = requests.size();

  const auto started = std::chrono::steady_clock::now();
  try {
    auto results = backend_.evaluate_batch(requests);
    const auto stopped = std::chrono::steady_clock::now();
    if (results.size() != padded_count) {
      throw std::runtime_error("batch evaluator returned wrong result count");
    }

    const auto latency_ms =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    record_batch_metrics(valid_count, padded_count, latency_ms);

    for (std::size_t i = 0; i < valid_count; ++i) {
      pending[i]->promise.set_value(std::move(results[i]));
    }
  } catch (...) {
    const auto error = std::current_exception();
    for (auto& item : pending) {
      item->promise.set_exception(error);
    }
  }
}

void AsyncBatchingEvaluator::record_batch_metrics(std::size_t valid_count,
                                                  std::size_t padded_count,
                                                  double latency_ms) {
  std::lock_guard lock(metrics_mutex_);
  metrics_.positions_evaluated += static_cast<std::uint64_t>(valid_count);
  metrics_.batches_evaluated += 1;
  metrics_.padded_positions +=
      static_cast<std::uint64_t>(padded_count - valid_count);
  metrics_.total_inference_latency_ms += latency_ms;
  ensure_histogram_size(metrics_.batch_size_histogram, padded_count);
  ensure_histogram_size(metrics_.valid_batch_size_histogram, valid_count);
  metrics_.batch_size_histogram[padded_count] += 1;
  metrics_.valid_batch_size_histogram[valid_count] += 1;
}

void AsyncBatchingEvaluator::stop() {
  {
    std::lock_guard lock(mutex_);
    if (stopping_) {
      return;
    }
    stopping_ = true;
  }
  not_empty_.notify_all();
  not_full_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

}  // namespace chessmoe::inference
