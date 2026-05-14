#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <chessmoe/selfplay/self_play_generator.h>

namespace chessmoe::selfplay {

struct OpeningPoolConfig {
  std::vector<std::string> fen_pool;
  bool deterministic_selection{true};
  bool color_balancing{true};
  std::uint32_t opening_seed{42};
};

struct GameWorkerConfig {
  SelfPlayConfig game;
  std::uint32_t worker_id{0};
  OpeningPoolConfig openings;
};

struct GameWorkerDiagnostics {
  int legal_gen_calls{0};
  int mcts_selection_calls{0};
  int evaluator_calls{0};
  double evaluator_wait_ms{0.0};
  double mcts_total_ms{0.0};
  double terminal_check_ms{0.0};
  double legal_gen_ms{0.0};
  double mcts_selection_ms{0.0};
  double mcts_expansion_ms{0.0};
  double replay_write_ms{0.0};
};

struct GameWorkerResult {
  SelfPlayGame game;
  GameWorkerDiagnostics diagnostics;
};

std::vector<std::string> load_opening_pool(const std::filesystem::path& path);
std::string select_opening(const std::vector<std::string>& pool, int game_id,
                           bool deterministic, std::mt19937& rng);

class GameWorker {
 public:
  GameWorker(eval::ISinglePositionEvaluator& evaluator, GameWorkerConfig config);

  GameWorkerResult run(int game_id);

 private:
  eval::ISinglePositionEvaluator& evaluator_;
  GameWorkerConfig config_;
  std::mt19937 rng_;

  std::string select_opening_for_game(int game_id);
};

}  // namespace chessmoe::selfplay
