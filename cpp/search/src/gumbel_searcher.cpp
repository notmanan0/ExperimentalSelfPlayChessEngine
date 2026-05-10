#include <chessmoe/search/gumbel_searcher.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include <chessmoe/chess/move_generator.h>

namespace chessmoe::search {
namespace {

struct Candidate {
  chess::Move move{};
  double prior{0.0};
  double logit{0.0};
  double gumbel{0.0};
  int visits{0};
  double total_value{0.0};
  std::uint64_t board_hash{0};

  [[nodiscard]] double mean_value() const {
    return visits == 0 ? 0.0 : total_value / visits;
  }
};

double terminal_value_for_side_to_move(const chess::Position& position) {
  if (position.in_check(position.side_to_move())) {
    return -1.0;
  }
  return 0.0;
}

int normalized_simulations(GumbelSearchLimits limits) {
  return std::max(1, limits.simulations);
}

int normalized_max_considered(GumbelSearchLimits limits, std::size_t legal_count) {
  const int requested = limits.max_considered_actions <= 0
                            ? static_cast<int>(legal_count)
                            : limits.max_considered_actions;
  return std::max(1, std::min(requested, static_cast<int>(legal_count)));
}

double gumbel_sample(std::mt19937& rng) {
  std::uniform_real_distribution<double> uniform(1e-12, 1.0 - 1e-12);
  const double u = uniform(rng);
  return -std::log(-std::log(u));
}

double candidate_score(const Candidate& candidate, double value_scale) {
  return candidate.gumbel + candidate.logit + value_scale * candidate.mean_value();
}

double evaluate_child_from_root_perspective(
    const chess::Position& root,
    const chess::Move move,
    eval::ISinglePositionEvaluator& evaluator) {
  auto child_position = root;
  child_position.make_move(move);
  const auto legal = chess::MoveGenerator::legal_moves(child_position);
  if (legal.empty()) {
    return -terminal_value_for_side_to_move(child_position);
  }
  const auto request = eval::EvaluationRequest::from_position(child_position);
  return -std::clamp(evaluator.evaluate(request).value, -1.0, 1.0);
}

void visit_candidate(Candidate& candidate, const chess::Position& root,
                     eval::ISinglePositionEvaluator& evaluator) {
  candidate.total_value +=
      evaluate_child_from_root_perspective(root, candidate.move, evaluator);
  candidate.visits += 1;
}

std::vector<std::size_t> top_indices_by_initial_score(
    const std::vector<Candidate>& candidates, int count) {
  std::vector<std::size_t> indices(candidates.size());
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  std::stable_sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) {
    const auto left = candidates[lhs].gumbel + candidates[lhs].logit;
    const auto right = candidates[rhs].gumbel + candidates[rhs].logit;
    if (left != right) {
      return left > right;
    }
    return candidates[lhs].move.to_uci() < candidates[rhs].move.to_uci();
  });
  indices.resize(static_cast<std::size_t>(count));
  return indices;
}

void keep_best_half(std::vector<std::size_t>& remaining,
                    const std::vector<Candidate>& candidates,
                    double value_scale) {
  std::stable_sort(remaining.begin(), remaining.end(), [&](auto lhs, auto rhs) {
    const auto left = candidate_score(candidates[lhs], value_scale);
    const auto right = candidate_score(candidates[rhs], value_scale);
    if (left != right) {
      return left > right;
    }
    return candidates[lhs].move.to_uci() < candidates[rhs].move.to_uci();
  });
  const auto keep = std::max<std::size_t>(1, (remaining.size() + 1) / 2);
  remaining.resize(keep);
}

std::vector<double> softmax_scores(const std::vector<Candidate>& candidates,
                                   double value_scale) {
  std::vector<double> probabilities(candidates.size(), 0.0);
  if (candidates.empty()) {
    return probabilities;
  }
  double max_score = -std::numeric_limits<double>::infinity();
  for (const auto& candidate : candidates) {
    max_score = std::max(max_score, candidate_score(candidate, value_scale));
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    probabilities[i] = std::exp(candidate_score(candidates[i], value_scale) - max_score);
    sum += probabilities[i];
  }
  if (sum <= 0.0) {
    const double uniform = 1.0 / candidates.size();
    std::fill(probabilities.begin(), probabilities.end(), uniform);
    return probabilities;
  }
  for (auto& probability : probabilities) {
    probability /= sum;
  }
  return probabilities;
}

MctsResult make_gumbel_result(const std::vector<Candidate>& candidates,
                              double root_value,
                              int simulations,
                              double value_scale) {
  MctsResult result;
  result.root_value = root_value;
  result.root_visits = static_cast<std::uint64_t>(simulations);
  const auto probabilities = softmax_scores(candidates, value_scale);

  const Candidate* best = nullptr;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& candidate = candidates[i];
    result.root_distribution.push_back({
        candidate.move,
        candidate.prior,
        probabilities[i],
        candidate.visits,
        candidate.total_value,
        candidate.mean_value(),
        candidate.board_hash,
    });
    if (best == nullptr ||
        candidate_score(candidate, value_scale) > candidate_score(*best, value_scale) ||
        (candidate_score(candidate, value_scale) == candidate_score(*best, value_scale) &&
         candidate.move.to_uci() < best->move.to_uci())) {
      best = &candidate;
    }
  }

  if (best != nullptr) {
    result.has_best_move = true;
    result.best_move = best->move;
  }
  return result;
}

}  // namespace

GumbelSearcher::GumbelSearcher(eval::ISinglePositionEvaluator& evaluator)
    : evaluator_(evaluator) {}

MctsResult GumbelSearcher::search(const chess::Position& root,
                                  GumbelSearchLimits limits) {
  const auto request = eval::EvaluationRequest::from_position(root);
  if (request.legal_moves.empty()) {
    MctsResult result;
    result.root_value = terminal_value_for_side_to_move(root);
    result.root_visits = 1;
    result.terminal = true;
    return result;
  }

  auto evaluation = evaluator_.evaluate(request);
  const auto normalized =
      eval::normalize_policy_over_legal_moves(std::move(evaluation), request.legal_moves);
  const double root_value = std::clamp(normalized.value, -1.0, 1.0);
  const int simulations = normalized_simulations(limits);
  const int considered_count =
      std::min(normalized_max_considered(limits, request.legal_moves.size()),
               simulations);

  std::mt19937 rng(limits.seed);
  std::vector<Candidate> all_candidates;
  all_candidates.reserve(request.legal_moves.size());
  for (std::size_t i = 0; i < request.legal_moves.size(); ++i) {
    auto child_position = root;
    child_position.make_move(request.legal_moves[i]);
    const double prior = std::max(1e-12, normalized.policy[i].probability);
    all_candidates.push_back(Candidate{
        request.legal_moves[i],
        prior,
        std::log(prior),
        limits.deterministic ? 0.0 : gumbel_sample(rng),
        0,
        0.0,
        child_position.hash(),
    });
  }

  const auto selected_indices =
      top_indices_by_initial_score(all_candidates, considered_count);
  std::vector<Candidate> candidates;
  candidates.reserve(selected_indices.size());
  for (const auto index : selected_indices) {
    candidates.push_back(all_candidates[index]);
  }

  std::vector<std::size_t> remaining(candidates.size());
  std::iota(remaining.begin(), remaining.end(), std::size_t{0});

  int spent = 0;
  while (spent < simulations && !remaining.empty()) {
    for (const auto index : remaining) {
      if (spent >= simulations) {
        break;
      }
      visit_candidate(candidates[index], root, evaluator_);
      ++spent;
    }
    if (remaining.size() > 1 && spent < simulations) {
      keep_best_half(remaining, candidates, limits.value_scale);
    }
  }

  return make_gumbel_result(candidates, root_value, simulations, limits.value_scale);
}

}  // namespace chessmoe::search
