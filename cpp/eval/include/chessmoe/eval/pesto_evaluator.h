#pragma once

#include <span>
#include <vector>

#include <chessmoe/eval/evaluator.h>
#include <chessmoe/chess/position.h>

namespace chessmoe::eval {

class PestoEvaluator final : public IBatchEvaluator {
 public:
  std::vector<EvaluationResult> evaluate_batch(
      std::span<const EvaluationRequest> requests) override;

  [[nodiscard]] int evaluate_mg(const chess::Position& position,
                                chess::Color perspective) const;

  [[nodiscard]] int evaluate_eg(const chess::Position& position,
                                chess::Color perspective) const;

  [[nodiscard]] int evaluate_tapered(const chess::Position& position,
                                     chess::Color perspective) const;

  [[nodiscard]] int game_phase(const chess::Position& position) const;
};

}  // namespace chessmoe::eval
