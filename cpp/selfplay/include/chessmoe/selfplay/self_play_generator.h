#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/search/gumbel_searcher.h>
#include <chessmoe/search/mcts_searcher.h>
#include <chessmoe/search/search_mode.h>

namespace chessmoe::selfplay {

enum class GameResult {
  Unknown,
  WhiteWin,
  Draw,
  BlackWin,
};

enum class TerminalReason {
  None,
  Checkmate,
  Stalemate,
  Repetition,
  FiftyMoveRule,
  MaxPlies,
};

struct TemperatureSchedule {
  double initial{1.0};
  double final{0.0};
  int cutoff_ply{30};
};

struct SelfPlayConfig {
  std::optional<std::string> opening_fen{};
  int max_plies{256};
  int search_visits{64};
  int search_max_depth{0};
  double cpuct{1.5};
  search::SearchMode search_mode{search::SearchMode::Puct};
  int gumbel_max_considered_actions{16};
  double gumbel_value_scale{1.0};
  TemperatureSchedule temperature{};
  bool add_root_dirichlet_noise{true};
  double root_dirichlet_alpha{0.3};
  double root_dirichlet_epsilon{0.25};
  bool resignation_enabled{false};
  bool deterministic{false};
  std::uint32_t seed{1};
  std::string model_version{"unknown"};
};

struct VisitEntry {
  chess::Move move{};
  int visit_count{0};
  double probability{0.0};
};

struct SelfPlaySample {
  std::string board_fen;
  std::vector<chess::Move> legal_moves;
  std::vector<VisitEntry> visit_distribution;
  double root_value{0.0};
  chess::Move selected_move{};
  std::string model_version;
  int search_budget{0};
  chess::Color side_to_move{chess::Color::White};
  GameResult final_result{GameResult::Unknown};
};

struct SelfPlayGame {
  std::vector<SelfPlaySample> samples;
  GameResult result{GameResult::Unknown};
  TerminalReason terminal_reason{TerminalReason::None};
  std::string final_fen;
};

class SelfPlayGenerator {
 public:
  explicit SelfPlayGenerator(eval::ISinglePositionEvaluator& evaluator);

  SelfPlayGame generate(const SelfPlayConfig& config);

 private:
  eval::ISinglePositionEvaluator& evaluator_;
};

std::string to_string(GameResult result);
std::string to_string(TerminalReason reason);
std::string to_debug_json(const SelfPlayGame& game);

}  // namespace chessmoe::selfplay
