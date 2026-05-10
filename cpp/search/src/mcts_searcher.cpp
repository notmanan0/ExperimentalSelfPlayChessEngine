#include <chessmoe/search/mcts_searcher.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include <chessmoe/chess/move_generator.h>

namespace chessmoe::search {

namespace {

double terminal_value_for_side_to_move(const chess::Position& position) {
  if (position.in_check(position.side_to_move())) {
    return -1.0;
  }
  return 0.0;
}

int normalized_visits(MctsLimits limits) {
  return std::max(1, limits.visits);
}

int normalized_depth(MctsLimits limits) {
  return std::max(0, limits.max_depth);
}

double normalized_cpuct(MctsLimits limits) {
  return limits.cpuct > 0.0 ? limits.cpuct : 1.5;
}

void apply_root_dirichlet_noise(MctsNode& root, MctsLimits limits) {
  if (root.children.empty() || limits.root_dirichlet_alpha <= 0.0 ||
      limits.root_dirichlet_epsilon <= 0.0) {
    return;
  }

  const double epsilon = std::clamp(limits.root_dirichlet_epsilon, 0.0, 1.0);
  std::mt19937 rng(limits.root_noise_seed);
  std::gamma_distribution<double> gamma(limits.root_dirichlet_alpha, 1.0);
  std::vector<double> noise;
  noise.reserve(root.children.size());

  for (std::size_t i = 0; i < root.children.size(); ++i) {
    noise.push_back(gamma(rng));
  }

  double sum = std::accumulate(noise.begin(), noise.end(), 0.0);
  if (sum <= 0.0) {
    const double uniform = 1.0 / root.children.size();
    std::fill(noise.begin(), noise.end(), uniform);
  } else {
    for (auto& value : noise) {
      value /= sum;
    }
  }

  for (std::size_t i = 0; i < root.children.size(); ++i) {
    auto& child = *root.children[i];
    child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
  }
}

double expand_and_evaluate(MctsNode& node, const chess::Position& position,
                           eval::ISinglePositionEvaluator& evaluator) {
  const auto request = eval::EvaluationRequest::from_position(position);
  node.board_hash = position.hash();
  node.expanded = true;

  if (request.legal_moves.empty()) {
    node.terminal = true;
    return terminal_value_for_side_to_move(position);
  }

  auto evaluation = evaluator.evaluate(request);
  const auto normalized =
      eval::normalize_policy_over_legal_moves(std::move(evaluation), request.legal_moves);
  node.children.reserve(request.legal_moves.size());

  for (std::size_t i = 0; i < request.legal_moves.size(); ++i) {
    auto child = std::make_unique<MctsNode>();
    child->move = request.legal_moves[i];
    child->prior = normalized.policy[i].probability;
    auto child_position = position;
    child_position.make_move(request.legal_moves[i]);
    child->board_hash = child_position.hash();
    node.children.push_back(std::move(child));
  }

  return std::clamp(normalized.value, -1.0, 1.0);
}

MctsNode& select_child(MctsNode& node, double cpuct) {
  if (node.children.empty()) {
    throw std::runtime_error("cannot select child from unexpanded leaf");
  }

  const double parent_visits = std::max(1, node.visit_count);
  double best_score = -std::numeric_limits<double>::infinity();
  MctsNode* best = nullptr;

  for (auto& child_ptr : node.children) {
    auto& child = *child_ptr;
    const double q = child.mean_value();
    const double u =
        cpuct * child.prior * std::sqrt(parent_visits) / (1.0 + child.visit_count);
    const double score = q + u;
    if (score > best_score) {
      best_score = score;
      best = &child;
    }
  }

  return *best;
}

void backpropagate(std::vector<MctsNode*>& path, double leaf_value) {
  double value_for_node_side = leaf_value;

  for (std::size_t reverse_index = path.size(); reverse_index > 0; --reverse_index) {
    auto& node = *path[reverse_index - 1];

    if (reverse_index == 1) {
      node.visit_count += 1;
      node.total_value += value_for_node_side;
    } else {
      node.visit_count += 1;
      node.total_value += -value_for_node_side;
    }

    value_for_node_side = -value_for_node_side;
  }
}

void run_playout(MctsNode& root, const chess::Position& root_position,
                 eval::ISinglePositionEvaluator& evaluator, MctsLimits limits) {
  chess::Position position = root_position;
  std::vector<MctsNode*> path;
  path.push_back(&root);

  MctsNode* node = &root;
  const int max_depth = normalized_depth(limits);
  int depth = 0;

  while (node->expanded && !node->terminal && !node->children.empty() &&
         (max_depth == 0 || depth < max_depth)) {
    node = &select_child(*node, normalized_cpuct(limits));
    position.make_move(node->move);
    path.push_back(node);
    ++depth;
  }

  double leaf_value = 0.0;
  if (node->expanded && !node->terminal &&
      max_depth > 0 && depth >= max_depth) {
    const auto legal_moves = chess::MoveGenerator::legal_moves(position);
    if (legal_moves.empty()) {
      node->terminal = true;
      leaf_value = terminal_value_for_side_to_move(position);
    } else {
      auto request = eval::EvaluationRequest::from_position(position);
      leaf_value = std::clamp(evaluator.evaluate(request).value, -1.0, 1.0);
    }
  } else {
    leaf_value = expand_and_evaluate(*node, position, evaluator);
  }

  backpropagate(path, leaf_value);
}

MctsResult make_result(const MctsNode& root) {
  MctsResult result;
  result.root_value = root.mean_value();
  result.root_visits = static_cast<std::uint64_t>(root.visit_count);
  result.terminal = root.terminal;
  result.root_distribution.reserve(root.children.size());

  const MctsNode* best = nullptr;
  for (const auto& child_ptr : root.children) {
    const auto& child = *child_ptr;
    result.root_distribution.push_back({
        child.move,
        child.prior,
        child.visit_count,
        child.total_value,
        child.mean_value(),
        child.board_hash,
    });

    if (best == nullptr || child.visit_count > best->visit_count ||
        (child.visit_count == best->visit_count &&
         child.mean_value() > best->mean_value())) {
      best = &child;
    }
  }

  if (best != nullptr && best->visit_count > 0) {
    result.has_best_move = true;
    result.best_move = best->move;
  }

  return result;
}

}  // namespace

MctsSearcher::MctsSearcher(eval::ISinglePositionEvaluator& evaluator)
    : evaluator_(evaluator) {}

MctsResult MctsSearcher::search(const chess::Position& root, MctsLimits limits) {
  MctsNode root_node;
  root_node.board_hash = root.hash();

  const double root_value = expand_and_evaluate(root_node, root, evaluator_);
  if (root_node.terminal) {
    root_node.visit_count = 1;
    root_node.total_value = root_value;
    return make_result(root_node);
  }

  apply_root_dirichlet_noise(root_node, limits);

  for (int visit = 0; visit < normalized_visits(limits); ++visit) {
    run_playout(root_node, root, evaluator_, limits);
  }

  return make_result(root_node);
}

}  // namespace chessmoe::search
