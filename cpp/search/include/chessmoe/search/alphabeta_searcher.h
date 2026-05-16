#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>
#include <chessmoe/eval/evaluator.h>

namespace chessmoe::search {

struct AlphaBetaLimits {
  int depth{4};
  int nodes{0};
  int movetime_ms{0};
  bool use_quiescence{true};
};

struct RootMoveScore {
  chess::Move move{};
  int score_cp{0};
  std::uint64_t nodes{0};
};

struct AlphaBetaResult {
  bool has_best_move{false};
  chess::Move best_move{};
  int score_cp{0};
  int depth_searched{0};
  std::uint64_t nodes{0};
  bool stopped{false};
  std::vector<RootMoveScore> root_moves;
};

enum class TTFlag : uint8_t {
  Exact = 0,
  Lower = 1,
  Upper = 2,
};

struct TTEntry {
  std::uint64_t key{0};
  int score{0};
  int depth{0};
  TTFlag flag{TTFlag::Exact};
  chess::Move best_move{};
};

class AlphaBetaSearcher {
 public:
  explicit AlphaBetaSearcher(eval::IBatchEvaluator& evaluator);

  AlphaBetaResult search(const chess::Position& root,
                         AlphaBetaLimits limits,
                         std::atomic_bool& stop_requested);

 private:
  int negamax(chess::Position& position, int depth, int ply,
              int alpha, int beta, bool allow_null, int static_eval);

  int quiescence(chess::Position& position, int ply,
                 int alpha, int beta);

  void order_moves(std::vector<chess::Move>& moves,
                   const chess::Move& tt_move, int ply);

  void record_killer(const chess::Move& move, int ply);
  void record_history(const chess::Move& move, int depth);

  [[nodiscard]] int static_eval(const chess::Position& position);
  [[nodiscard]] int reduction(int depth, int move_count, bool is_pv,
                              bool in_check, bool gives_check, bool is_capture);

  eval::IBatchEvaluator& evaluator_;
  eval::SynchronousEvaluator sync_evaluator_;

  std::unordered_map<std::uint64_t, TTEntry> tt_;
  std::uint64_t nodes_{0};
  std::atomic_bool* stop_{nullptr};
  bool stopped_{false};

  static constexpr int kMaxPly = 128;
  chess::Move killer_moves_[kMaxPly][2]{};
  int history_[2][64][64]{};
  chess::Move pv_[kMaxPly][kMaxPly]{};
  int pv_length_[kMaxPly]{};

  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point deadline_;
  bool has_deadline_{false};
};

}  // namespace chessmoe::search
