#include <chessmoe/cuda/batching/batch_planner.h>

#include <algorithm>
#include <stdexcept>

namespace chessmoe::cuda::batching {

BatchPlanner::BatchPlanner(BatchPlannerConfig config) : config_(config) {
  if (config_.max_batch == 0 || config_.preferred_batch == 0 ||
      config_.min_batch == 0) {
    throw std::invalid_argument("batch sizes must be positive");
  }
  if (config_.min_batch > config_.preferred_batch ||
      config_.preferred_batch > config_.max_batch) {
    throw std::invalid_argument(
        "batch sizes must satisfy min <= preferred <= max");
  }
}

const BatchPlannerConfig& BatchPlanner::config() const {
  return config_;
}

std::vector<BatchSlice> BatchPlanner::plan(std::size_t request_count) const {
  std::vector<BatchSlice> batches;
  std::size_t start = 0;

  while (start < request_count) {
    const std::size_t remaining = request_count - start;
    const std::size_t target =
        config_.mode == InferenceMode::SelfPlay ? config_.preferred_batch
                                                : config_.max_batch;
    const std::size_t valid_count = std::min(remaining, target);
    const bool should_pad =
        config_.mode == InferenceMode::SelfPlay && config_.pad_to_preferred_batch;
    const std::size_t padded_count =
        should_pad ? config_.preferred_batch : valid_count;

    batches.push_back(BatchSlice{start, valid_count, padded_count});
    start += valid_count;
  }

  return batches;
}

}  // namespace chessmoe::cuda::batching
