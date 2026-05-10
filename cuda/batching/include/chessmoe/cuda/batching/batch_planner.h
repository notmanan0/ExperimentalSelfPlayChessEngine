#pragma once

#include <cstddef>
#include <vector>

namespace chessmoe::cuda::batching {

enum class InferenceMode {
  MatchPlay,
  SelfPlay,
};

struct BatchPlannerConfig {
  InferenceMode mode{InferenceMode::MatchPlay};
  std::size_t min_batch{1};
  std::size_t preferred_batch{1};
  std::size_t max_batch{1};
  bool pad_to_preferred_batch{false};
};

struct BatchSlice {
  std::size_t start{0};
  std::size_t valid_count{0};
  std::size_t padded_count{0};
};

class BatchPlanner {
 public:
  explicit BatchPlanner(BatchPlannerConfig config);

  [[nodiscard]] const BatchPlannerConfig& config() const;
  [[nodiscard]] std::vector<BatchSlice> plan(std::size_t request_count) const;

 private:
  BatchPlannerConfig config_;
};

}  // namespace chessmoe::cuda::batching
