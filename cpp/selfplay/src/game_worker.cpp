#include <chessmoe/selfplay/game_worker.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>

namespace chessmoe::selfplay {
namespace {

constexpr std::string_view kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

GameResult win_for(chess::Color color) {
  return color == chess::Color::White ? GameResult::WhiteWin
                                      : GameResult::BlackWin;
}

}  // namespace

std::vector<std::string> load_opening_pool(const std::filesystem::path& path) {
  std::vector<std::string> pool;
  if (!std::filesystem::exists(path)) {
    return pool;
  }
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    if (!line.empty() && line[0] != '#') {
      pool.push_back(line);
    }
  }
  return pool;
}

std::string select_opening(const std::vector<std::string>& pool, int game_id,
                           bool deterministic, std::mt19937& rng) {
  if (pool.empty()) {
    return std::string(kStartFen);
  }
  if (deterministic) {
    return pool[static_cast<std::size_t>(game_id) % pool.size()];
  }
  std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
  return pool[dist(rng)];
}

GameWorker::GameWorker(eval::ISinglePositionEvaluator& evaluator,
                       GameWorkerConfig config)
    : evaluator_(evaluator),
      config_(std::move(config)),
      rng_(config_.game.seed + config_.worker_id * 7919) {}

std::string GameWorker::select_opening_for_game(int game_id) {
  return select_opening(config_.openings.fen_pool, game_id,
                        config_.openings.deterministic_selection, rng_);
}

GameWorkerResult GameWorker::run(int game_id) {
  GameWorkerResult result;
  auto& game = result.game;
  auto& diag = result.diagnostics;

  auto game_config = config_.game;
  game_config.seed = config_.game.seed + static_cast<std::uint32_t>(game_id);
  if (!config_.openings.fen_pool.empty() || !game_config.opening_fen) {
    game_config.opening_fen = select_opening_for_game(game_id);
  }

  if (config_.openings.color_balancing && !config_.openings.fen_pool.empty()) {
    const auto opening_idx =
        static_cast<std::size_t>(game_id) % config_.openings.fen_pool.size();
    game_config.opening_color_swapped = (opening_idx % 2 == 1);
  }

  auto position = chess::Fen::parse(
      game_config.opening_fen.has_value() ? *game_config.opening_fen
                                          : std::string(kStartFen));
  std::map<std::uint64_t, int> repetitions;
  repetitions[position.hash()] = 1;
  std::mt19937 rng(game_config.seed);

  for (int ply = 0; ply <= std::max(0, game_config.max_plies); ++ply) {
    const auto legal = chess::MoveGenerator::legal_moves(position);
    diag.legal_gen_calls++;

    if (legal.empty()) {
      if (position.in_check(position.side_to_move())) {
        game.result = win_for(chess::opposite(position.side_to_move()));
        game.terminal_reason = TerminalReason::Checkmate;
      } else {
        game.result = GameResult::Draw;
        game.terminal_reason = TerminalReason::Stalemate;
      }
      break;
    }

    auto rep_it = repetitions.find(position.hash());
    if (rep_it != repetitions.end() && rep_it->second >= 3) {
      game.result = GameResult::Draw;
      game.terminal_reason = TerminalReason::Repetition;
      break;
    }
    if (position.halfmove_clock() >= 100) {
      game.result = GameResult::Draw;
      game.terminal_reason = TerminalReason::FiftyMoveRule;
      break;
    }
    if (ply >= game_config.max_plies) {
      game.result = GameResult::Draw;
      game.terminal_reason = TerminalReason::MaxPlies;
      break;
    }

    search::MctsLimits limits;
    limits.visits = std::max(1, game_config.search_visits);
    limits.max_depth = std::max(0, game_config.search_max_depth);
    limits.cpuct = game_config.cpuct;
    if (game_config.add_root_dirichlet_noise) {
      limits.root_dirichlet_alpha = game_config.root_dirichlet_alpha;
      limits.root_dirichlet_epsilon = game_config.root_dirichlet_epsilon;
      limits.root_noise_seed =
          game_config.seed + static_cast<std::uint32_t>(ply * 9973);
    }

    search::MctsResult search_result;
    if (game_config.search_mode == search::SearchMode::Gumbel) {
      search::GumbelSearcher searcher(evaluator_);
      search::GumbelSearchLimits gumbel_limits;
      gumbel_limits.simulations = limits.visits;
      gumbel_limits.max_considered_actions =
          std::max(1, game_config.gumbel_max_considered_actions);
      gumbel_limits.value_scale = game_config.gumbel_value_scale > 0.0
                                      ? game_config.gumbel_value_scale
                                      : 1.0;
      gumbel_limits.deterministic = game_config.deterministic;
      gumbel_limits.seed = game_config.seed + static_cast<std::uint32_t>(ply * 9973);
      search_result = searcher.search(position, gumbel_limits);
    } else {
      search::MctsSearcher searcher(evaluator_);
      search_result = searcher.search(position, limits);
    }
    diag.mcts_selection_calls++;
    diag.mcts_legal_move_generation_calls +=
        search_result.profile.legal_move_generation_calls;
    diag.mcts_legal_move_generation_ms +=
        search_result.profile.legal_move_generation_ms;

    if (!search_result.has_best_move) {
      game.result = GameResult::Draw;
      game.terminal_reason = TerminalReason::Stalemate;
      break;
    }

    std::vector<VisitEntry> visits;
    visits.reserve(search_result.root_distribution.size());
    const double target_total = std::accumulate(
        search_result.root_distribution.begin(),
        search_result.root_distribution.end(), 0.0,
        [](double sum, const auto& s) { return sum + s.target_probability; });
    const int visit_total = std::accumulate(
        search_result.root_distribution.begin(),
        search_result.root_distribution.end(), 0,
        [](int sum, const auto& s) { return sum + s.visit_count; });
    for (const auto& stat : search_result.root_distribution) {
      visits.push_back({
          stat.move,
          stat.visit_count,
          target_total > 0.0
              ? stat.target_probability / target_total
              : (visit_total > 0
                     ? static_cast<double>(stat.visit_count) / visit_total
                     : 0.0),
      });
    }

    const double temp = ply < game_config.temperature.cutoff_ply
                            ? game_config.temperature.initial
                            : game_config.temperature.final;
    chess::Move selected{};
    if (temp <= 0.0) {
      const auto best = std::max_element(
          visits.begin(), visits.end(), [](const auto& a, const auto& b) {
            if (a.visit_count != b.visit_count)
              return a.visit_count < b.visit_count;
            return a.move.to_uci() > b.move.to_uci();
          });
      selected = best->move;
    } else {
      std::vector<double> weights;
      weights.reserve(visits.size());
      for (const auto& e : visits) {
        weights.push_back(
            std::pow(static_cast<double>(std::max(0, e.visit_count)),
                     1.0 / temp));
      }
      if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
        std::fill(weights.begin(), weights.end(), 1.0);
      }
      std::discrete_distribution<std::size_t> dist(weights.begin(),
                                                    weights.end());
      selected = visits[dist(rng)].move;
    }

    SelfPlaySample sample;
    sample.board_fen = chess::Fen::to_string(position);
    sample.legal_moves = legal;
    sample.visit_distribution = visits;
    sample.root_value = search_result.root_value;
    sample.selected_move = selected;
    sample.model_version = game_config.model_version;
    sample.search_budget = limits.visits;
    sample.side_to_move = position.side_to_move();
    game.samples.push_back(sample);

    position.make_move(selected);
    repetitions[position.hash()] += 1;
  }

  if (game.result == GameResult::Unknown) {
    game.result = GameResult::Draw;
    game.terminal_reason = TerminalReason::MaxPlies;
  }
  game.final_fen = chess::Fen::to_string(position);

  for (auto& sample : game.samples) {
    sample.final_result = game.result;
  }

  return result;
}

}  // namespace chessmoe::selfplay
