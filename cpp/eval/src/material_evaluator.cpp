#include <chessmoe/eval/material_evaluator.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace chessmoe::eval {

namespace {

WdlOutput value_to_wdl(double value) {
  const double decisive = std::abs(value);
  const double draw = std::max(0.0, 1.0 - decisive);
  const double win = value > 0.0 ? decisive : 0.0;
  const double loss = value < 0.0 ? decisive : 0.0;
  return {win, draw, loss};
}

}  // namespace

std::vector<EvaluationResult> MaterialEvaluator::evaluate_batch(
    std::span<const EvaluationRequest> requests) {
  std::vector<EvaluationResult> results;
  results.reserve(requests.size());

  for (const auto& request : requests) {
    EvaluationResult result;
    result.policy.reserve(request.legal_moves.size());
    const double prior =
        request.legal_moves.empty() ? 0.0 : 1.0 / request.legal_moves.size();
    const double logit = request.legal_moves.empty() ? 0.0 : std::log(prior);
    for (const auto move : request.legal_moves) {
      result.policy.push_back({move, logit, prior});
    }

    const int cp = evaluate(request.position, request.side_to_move);
    result.value = std::clamp(std::tanh(static_cast<double>(cp) / 1000.0), -1.0, 1.0);
    result.wdl = value_to_wdl(result.value);
    result.moves_left = 40.0;
    results.push_back(
        normalize_policy_over_legal_moves(std::move(result), request.legal_moves));
  }

  return results;
}

int MaterialEvaluator::evaluate(const chess::Position& position,
                                chess::Color perspective) const {
  static constexpr std::array<int, 6> values{100, 320, 330, 500, 900, 0};

  int white_minus_black = 0;
  for (int piece = 0; piece < 6; ++piece) {
    const auto type = static_cast<chess::PieceType>(piece);
    const int value = values[static_cast<std::size_t>(piece)];
    white_minus_black +=
        value * std::popcount(position.board().pieces_of(chess::Color::White, type));
    white_minus_black -=
        value * std::popcount(position.board().pieces_of(chess::Color::Black, type));
  }

  return perspective == chess::Color::White ? white_minus_black : -white_minus_black;
}

}  // namespace chessmoe::eval
