#pragma once

#include <atomic>
#include <cstdint>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>
#include <chessmoe/eval/material_evaluator.h>

namespace chessmoe::search {

struct SearchLimits {
  int depth{0};
  int nodes{0};
  int movetime_ms{0};
};

struct SearchResult {
  bool has_best_move{false};
  chess::Move best_move{};
  int score_cp{0};
  std::uint64_t nodes{0};
  bool stopped{false};
};

class SimpleSearcher {
 public:
  explicit SimpleSearcher(eval::MaterialEvaluator evaluator = {});

  [[nodiscard]] SearchResult search(const chess::Position& root,
                                    SearchLimits limits,
                                    std::atomic_bool& stop_requested) const;

 private:
  eval::MaterialEvaluator evaluator_;
};

}  // namespace chessmoe::search
