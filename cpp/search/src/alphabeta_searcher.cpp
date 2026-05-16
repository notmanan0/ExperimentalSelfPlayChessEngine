#include <chessmoe/search/alphabeta_searcher.h>

#include <algorithm>
#include <climits>
#include <cmath>

#include <chessmoe/chess/move_generator.h>

namespace chessmoe::search {

namespace {

constexpr int kMateScore = 30000;
constexpr int kInfinity = 32000;
constexpr int kMaxDepth = 64;
constexpr int kNullMoveReduction = 3;
constexpr int kLmrFullDepthMoves = 4;
constexpr int kLmrReductionLimit = 3;

int mate_score(int ply) {
  return kMateScore - ply;
}

bool is_mate_score(int score) {
  return std::abs(score) > kMateScore - kMaxDepth;
}

}  // namespace

AlphaBetaSearcher::AlphaBetaSearcher(eval::IBatchEvaluator& evaluator)
    : evaluator_(evaluator), sync_evaluator_(evaluator) {}

AlphaBetaResult AlphaBetaSearcher::search(
    const chess::Position& root,
    AlphaBetaLimits limits,
    std::atomic_bool& stop_requested) {
  stop_ = &stop_requested;
  stopped_ = false;
  nodes_ = 0;
  start_time_ = std::chrono::steady_clock::now();
  has_deadline_ = false;

  if (limits.movetime_ms > 0) {
    deadline_ = start_time_ + std::chrono::milliseconds(limits.movetime_ms);
    has_deadline_ = true;
  }

  tt_.clear();
  std::memset(killer_moves_, 0, sizeof(killer_moves_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(pv_length_, 0, sizeof(pv_length_));

  const auto legal_moves = chess::MoveGenerator::legal_moves(root);
  AlphaBetaResult result{};

  if (legal_moves.empty()) {
    result.nodes = 1;
    result.score_cp = root.in_check(root.side_to_move()) ? -kMateScore : 0;
    return result;
  }

  const int max_depth = limits.depth > 0 ? std::min(limits.depth, kMaxDepth) : 4;

  chess::Position pos = root;
  int best_score = -kInfinity;

  for (int depth = 1; depth <= max_depth; ++depth) {
    if (stopped_) break;

    int alpha = -kInfinity;
    int beta = kInfinity;

    auto moves = chess::MoveGenerator::legal_moves(pos);

    chess::Move tt_move{};
    auto tt_it = tt_.find(pos.hash());
    if (tt_it != tt_.end()) {
      tt_move = tt_it->second.best_move;
    }
  order_moves(moves, tt_move, 0);

    int best_move_idx = 0;
    const int root_static_eval = static_eval(pos);
    for (int i = 0; i < static_cast<int>(moves.size()); ++i) {
      if (stopped_) break;

      auto undo = pos.make_move(moves[i]);
      const int score = -negamax(pos, depth - 1, 1, -beta, -alpha, true, -root_static_eval);
      pos.unmake_move(moves[i], undo);

      if (stopped_) break;

      if (score > best_score) {
        best_score = score;
        best_move_idx = i;
        alpha = score;

        pv_[0][0] = moves[i];
        for (int j = 1; j < pv_length_[1]; ++j) {
          pv_[0][j] = pv_[1][j];
        }
        pv_length_[0] = pv_length_[1] + 1;
      }
    }

    if (!stopped_) {
      result.best_move = moves[best_move_idx];
      result.has_best_move = true;
      result.score_cp = best_score;
      result.depth_searched = depth;
    }

    if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
      break;
    }
  }

  result.nodes = nodes_;
  result.stopped = stopped_;

  if (!stopped_) {
    auto root_moves = chess::MoveGenerator::legal_moves(root);
    const int root_se = static_eval(pos);
    for (const auto& move : root_moves) {
      if (stopped_) break;
      auto undo = pos.make_move(move);
      const int score = -negamax(pos, 0, 1, -kInfinity, kInfinity, false, -root_se);
      pos.unmake_move(move, undo);
      result.root_moves.push_back({move, score, 0});
    }

    std::sort(result.root_moves.begin(), result.root_moves.end(),
              [](const auto& a, const auto& b) { return a.score_cp > b.score_cp; });
  }

  return result;
}

int AlphaBetaSearcher::negamax(chess::Position& position, int depth, int ply,
                               int alpha, int beta, bool allow_null, int static_eval_val) {
  if (stopped_) return 0;

  if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
    stopped_ = true;
    *stop_ = true;
    return 0;
  }

  if (stop_->load(std::memory_order_relaxed)) {
    stopped_ = true;
    return 0;
  }

  pv_length_[ply] = 0;

  if (ply > 0 && position.repetition_count(position.hash()) > 1) {
    return 0;
  }

  const bool in_check = position.in_check(position.side_to_move());
  const bool is_pv = (beta - alpha) > 1;

  if (depth <= 0) {
    if (in_check) {
      depth = 1;
    } else {
      return quiescence(position, ply, alpha, beta);
    }
  }

  auto legal_moves = chess::MoveGenerator::legal_moves(position);

  if (legal_moves.empty()) {
    return in_check ? -mate_score(ply) : 0;
  }

  const auto tt_it = tt_.find(position.hash());
  chess::Move tt_move{};
  if (tt_it != tt_.end()) {
    tt_move = tt_it->second.best_move;
    if (tt_it->second.depth >= depth) {
      const int tt_score = tt_it->second.score;
      if (tt_it->second.flag == TTFlag::Exact) return tt_score;
      if (tt_it->second.flag == TTFlag::Lower && tt_score >= beta) return tt_score;
      if (tt_it->second.flag == TTFlag::Upper && tt_score <= alpha) return tt_score;
    }
  }

  order_moves(legal_moves, tt_move, ply);

  int best_score = -kInfinity;
  chess::Move best_move{};
  TTFlag flag = TTFlag::Upper;
  int move_count = 0;

  for (int i = 0; i < static_cast<int>(legal_moves.size()); ++i) {
    if (stopped_) break;

    const auto& move = legal_moves[i];
    const bool is_capture = move.is_capture();
    const bool is_promotion = move.is_promotion();
    const bool gives_check = false;  // simplified: assume no check detection

    auto undo = position.make_move(move);
    ++move_count;

    int new_depth = depth - 1;
    if (in_check) new_depth = depth - 1;

    // Late-move reductions: reduce depth for moves searched late
    int score;
    if (move_count > kLmrFullDepthMoves && depth >= kLmrReductionLimit &&
        !is_capture && !is_promotion && !in_check && !gives_check) {
      // Reduced search
      score = -negamax(position, new_depth - 1, ply + 1, -alpha - 1, -alpha,
                       true, -static_eval_val);

      // Re-search at full depth if reduced search fails high
      if (score > alpha) {
        score = -negamax(position, new_depth, ply + 1, -beta, -alpha,
                         true, -static_eval_val);
      }
    } else {
      // Full depth search
      score = -negamax(position, new_depth, ply + 1, -beta, -alpha,
                       true, -static_eval_val);
    }

    position.unmake_move(move, undo);

    if (stopped_) break;

    if (score > best_score) {
      best_score = score;
      best_move = move;

      if (score > alpha) {
        alpha = score;
        flag = TTFlag::Exact;

        pv_[ply][0] = move;
        for (int j = 0; j < pv_length_[ply + 1]; ++j) {
          pv_[ply][j + 1] = pv_[ply + 1][j];
        }
        pv_length_[ply] = pv_length_[ply + 1] + 1;
      }
    }

    if (alpha >= beta) {
      flag = TTFlag::Lower;
      if (!is_capture) {
        record_killer(move, ply);
        record_history(move, depth);
      }
      break;
    }
  }

  if (!stopped_ && best_move.from != chess::Square::None) {
    TTEntry entry{};
    entry.key = position.hash();
    entry.score = best_score;
    entry.depth = depth;
    entry.flag = flag;
    entry.best_move = best_move;
    tt_[position.hash()] = entry;
  }

  ++nodes_;
  return best_score;
}

int AlphaBetaSearcher::quiescence(chess::Position& position, int ply,
                                  int alpha, int beta) {
  if (stopped_) return 0;

  ++nodes_;

  const int stand_pat = static_eval(position);

  if (stand_pat >= beta) return beta;
  if (stand_pat > alpha) alpha = stand_pat;

  const auto legal_moves = chess::MoveGenerator::legal_moves(position);

  std::vector<chess::Move> captures;
  for (const auto& move : legal_moves) {
    if (move.is_capture() || move.is_promotion()) {
      captures.push_back(move);
    }
  }

  order_moves(captures, chess::Move{}, ply);

  for (const auto& move : captures) {
    if (stopped_) break;

    auto undo = position.make_move(move);
    const int score = -quiescence(position, ply + 1, -beta, -alpha);
    position.unmake_move(move, undo);

    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

void AlphaBetaSearcher::order_moves(std::vector<chess::Move>& moves,
                                    const chess::Move& tt_move, int ply) {
  std::stable_sort(moves.begin(), moves.end(),
                   [&](const chess::Move& a, const chess::Move& b) {
    int sa = 0, sb = 0;
    if (a == tt_move) sa = 1000000;
    if (b == tt_move) sb = 1000000;
    if (a.is_capture()) sa += 100000;
    if (b.is_capture()) sb += 100000;
    if (a.is_promotion()) sa += 50000;
    if (b.is_promotion()) sb += 50000;
    if (ply < kMaxPly) {
      if (a == killer_moves_[ply][0]) sa += 90000;
      if (a == killer_moves_[ply][1]) sa += 80000;
      if (b == killer_moves_[ply][0]) sb += 90000;
      if (b == killer_moves_[ply][1]) sb += 80000;
    }
    sa += history_[0][static_cast<int>(a.from)][static_cast<int>(a.to)];
    sb += history_[0][static_cast<int>(b.from)][static_cast<int>(b.to)];
    return sa > sb;
  });
}

void AlphaBetaSearcher::record_killer(const chess::Move& move, int ply) {
  if (ply >= kMaxPly) return;
  if (!(move == killer_moves_[ply][0])) {
    killer_moves_[ply][1] = killer_moves_[ply][0];
    killer_moves_[ply][0] = move;
  }
}

void AlphaBetaSearcher::record_history(const chess::Move& move, int depth) {
  history_[0][static_cast<int>(move.from)][static_cast<int>(move.to)] +=
      depth * depth;
}

int AlphaBetaSearcher::static_eval(const chess::Position& position) {
  const auto request = eval::EvaluationRequest::from_position(position);
  const auto result = sync_evaluator_.evaluate(request);
  return static_cast<int>(std::round(result.value * 600.0));
}

}  // namespace chessmoe::search
