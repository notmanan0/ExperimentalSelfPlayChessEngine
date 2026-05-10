#include <chessmoe/selfplay/self_play_generator.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>

namespace chessmoe::selfplay {

namespace {

constexpr std::string_view kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

GameResult win_for(chess::Color color) {
  return color == chess::Color::White ? GameResult::WhiteWin : GameResult::BlackWin;
}

double temperature_for_ply(const TemperatureSchedule& schedule, int ply) {
  return ply < schedule.cutoff_ply ? schedule.initial : schedule.final;
}

search::MctsLimits limits_for_config(const SelfPlayConfig& config, int ply) {
  search::MctsLimits limits;
  limits.visits = std::max(1, config.search_visits);
  limits.max_depth = std::max(0, config.search_max_depth);
  limits.cpuct = config.cpuct;
  if (config.add_root_dirichlet_noise) {
    limits.root_dirichlet_alpha = config.root_dirichlet_alpha;
    limits.root_dirichlet_epsilon = config.root_dirichlet_epsilon;
    limits.root_noise_seed = config.seed + static_cast<std::uint32_t>(ply * 9973);
  }
  return limits;
}

std::vector<VisitEntry> make_visit_distribution(
    const std::vector<search::RootMoveStats>& root_distribution) {
  std::vector<VisitEntry> visits;
  visits.reserve(root_distribution.size());
  const int total = std::accumulate(
      root_distribution.begin(), root_distribution.end(), 0,
      [](int sum, const auto& stat) { return sum + stat.visit_count; });

  for (const auto& stat : root_distribution) {
    visits.push_back({
        stat.move,
        stat.visit_count,
        total > 0 ? static_cast<double>(stat.visit_count) / total : 0.0,
    });
  }
  return visits;
}

chess::Move choose_move(const std::vector<VisitEntry>& visits, double temperature,
                        std::mt19937& rng) {
  if (visits.empty()) {
    throw std::runtime_error("cannot choose move without visit distribution");
  }

  if (temperature <= 0.0) {
    const auto best = std::max_element(
        visits.begin(), visits.end(), [](const auto& lhs, const auto& rhs) {
          if (lhs.visit_count != rhs.visit_count) {
            return lhs.visit_count < rhs.visit_count;
          }
          return lhs.move.to_uci() > rhs.move.to_uci();
        });
    return best->move;
  }

  std::vector<double> weights;
  weights.reserve(visits.size());
  for (const auto& entry : visits) {
    weights.push_back(std::pow(static_cast<double>(std::max(0, entry.visit_count)),
                               1.0 / temperature));
  }

  if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
    std::fill(weights.begin(), weights.end(), 1.0);
  }

  std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
  return visits[dist(rng)].move;
}

std::optional<std::pair<GameResult, TerminalReason>> terminal_status(
    const chess::Position& position,
    const std::map<std::uint64_t, int>& repetitions,
    int ply,
    int max_plies) {
  const auto legal = chess::MoveGenerator::legal_moves(position);
  if (legal.empty()) {
    if (position.in_check(position.side_to_move())) {
      return std::pair{win_for(chess::opposite(position.side_to_move())),
                       TerminalReason::Checkmate};
    }
    return std::pair{GameResult::Draw, TerminalReason::Stalemate};
  }

  const auto repetition = repetitions.find(position.hash());
  if (repetition != repetitions.end() && repetition->second >= 3) {
    return std::pair{GameResult::Draw, TerminalReason::Repetition};
  }

  if (position.halfmove_clock() >= 100) {
    return std::pair{GameResult::Draw, TerminalReason::FiftyMoveRule};
  }

  if (ply >= max_plies) {
    return std::pair{GameResult::Draw, TerminalReason::MaxPlies};
  }

  return std::nullopt;
}

std::string json_escape(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      escaped.push_back('\\');
    }
    escaped.push_back(c);
  }
  return escaped;
}

}  // namespace

SelfPlayGenerator::SelfPlayGenerator(eval::ISinglePositionEvaluator& evaluator)
    : evaluator_(evaluator) {}

SelfPlayGame SelfPlayGenerator::generate(const SelfPlayConfig& config) {
  SelfPlayGame game;
  auto position = chess::Fen::parse(
      config.opening_fen.has_value() ? *config.opening_fen : std::string{kStartFen});
  std::map<std::uint64_t, int> repetitions;
  repetitions[position.hash()] = 1;
  std::mt19937 rng(config.seed);

  for (int ply = 0; ply <= std::max(0, config.max_plies); ++ply) {
    if (const auto terminal =
            terminal_status(position, repetitions, ply, std::max(0, config.max_plies))) {
      game.result = terminal->first;
      game.terminal_reason = terminal->second;
      break;
    }

    search::MctsSearcher searcher(evaluator_);
    const auto limits = limits_for_config(config, ply);
    const auto search_result = searcher.search(position, limits);
    if (!search_result.has_best_move) {
      game.result = GameResult::Draw;
      game.terminal_reason = TerminalReason::Stalemate;
      break;
    }

    auto visits = make_visit_distribution(search_result.root_distribution);
    const auto selected =
        choose_move(visits, temperature_for_ply(config.temperature, ply), rng);

    SelfPlaySample sample;
    sample.board_fen = chess::Fen::to_string(position);
    sample.legal_moves = chess::MoveGenerator::legal_moves(position);
    sample.visit_distribution = visits;
    sample.root_value = search_result.root_value;
    sample.selected_move = selected;
    sample.model_version = config.model_version;
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

  return game;
}

std::string to_string(GameResult result) {
  switch (result) {
    case GameResult::Unknown:
      return "unknown";
    case GameResult::WhiteWin:
      return "white_win";
    case GameResult::Draw:
      return "draw";
    case GameResult::BlackWin:
      return "black_win";
  }
  return "unknown";
}

std::string to_string(TerminalReason reason) {
  switch (reason) {
    case TerminalReason::None:
      return "none";
    case TerminalReason::Checkmate:
      return "checkmate";
    case TerminalReason::Stalemate:
      return "stalemate";
    case TerminalReason::Repetition:
      return "repetition";
    case TerminalReason::FiftyMoveRule:
      return "fifty_move_rule";
    case TerminalReason::MaxPlies:
      return "max_plies";
  }
  return "none";
}

std::string to_debug_json(const SelfPlayGame& game) {
  std::ostringstream out;
  out << "{\"result\":\"" << to_string(game.result) << "\",";
  out << "\"terminal_reason\":\"" << to_string(game.terminal_reason) << "\",";
  out << "\"final_fen\":\"" << json_escape(game.final_fen) << "\",";
  out << "\"samples\":[";
  for (std::size_t i = 0; i < game.samples.size(); ++i) {
    const auto& sample = game.samples[i];
    if (i != 0) {
      out << ',';
    }
    out << "{\"fen\":\"" << json_escape(sample.board_fen) << "\",";
    out << "\"selected\":\"" << sample.selected_move.to_uci() << "\",";
    out << "\"result\":\"" << to_string(sample.final_result) << "\"}";
  }
  out << "]}";
  return out.str();
}

}  // namespace chessmoe::selfplay
