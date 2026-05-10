#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include <chessmoe/chess/move_generator.h>
#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>

namespace chessmoe::eval {

struct EvaluationRequest {
  chess::Position position;
  std::vector<chess::Move> legal_moves;
  std::uint64_t hash{0};
  chess::Color side_to_move{chess::Color::White};

  static EvaluationRequest from_position(const chess::Position& position);
};

struct PolicyEntry {
  chess::Move move;
  double logit{0.0};
  double probability{0.0};
};

struct WdlOutput {
  double win{0.0};
  double draw{1.0};
  double loss{0.0};
};

struct EvaluationResult {
  std::vector<PolicyEntry> policy;
  WdlOutput wdl{};
  double value{0.0};
  std::optional<double> moves_left{};
};

class IBatchEvaluator {
 public:
  virtual ~IBatchEvaluator() = default;

  virtual std::vector<EvaluationResult> evaluate_batch(
      std::span<const EvaluationRequest> requests) = 0;
};

class ISinglePositionEvaluator {
 public:
  virtual ~ISinglePositionEvaluator() = default;

  virtual EvaluationResult evaluate(const EvaluationRequest& request) = 0;
};

class SynchronousEvaluator final : public ISinglePositionEvaluator {
 public:
  explicit SynchronousEvaluator(IBatchEvaluator& batch_evaluator);

  EvaluationResult evaluate(const EvaluationRequest& request) override;

 private:
  IBatchEvaluator& batch_evaluator_;
};

EvaluationResult normalize_policy_over_legal_moves(
    EvaluationResult result, std::span<const chess::Move> legal_moves);

}  // namespace chessmoe::eval
