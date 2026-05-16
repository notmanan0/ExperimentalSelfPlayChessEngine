#include <chessmoe/eval/pesto_evaluator.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

#include <chessmoe/chess/bitboard.h>
#include <chessmoe/chess/move_generator.h>

namespace chessmoe::eval {

namespace {

constexpr std::array<int, 6> kMgPieceValues{82, 337, 365, 477, 1025, 0};
constexpr std::array<int, 6> kEgPieceValues{94, 281, 297, 512, 936, 0};

constexpr std::array<int, 6> kGamePhaseInc{0, 1, 1, 2, 4, 0};
constexpr int kTotalPhase = kGamePhaseInc[0] * 16 + kGamePhaseInc[1] * 4 +
                            kGamePhaseInc[2] * 4 + kGamePhaseInc[3] * 4 +
                            kGamePhaseInc[4] * 2;

constexpr std::array<int, 64> kMgPawnTable{
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
};

constexpr std::array<int, 64> kEgPawnTable{
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};

constexpr std::array<int, 64> kMgKnightTable{
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
};

constexpr std::array<int, 64> kEgKnightTable{
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};

constexpr std::array<int, 64> kMgBishopTable{
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};

constexpr std::array<int, 64> kEgBishopTable{
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
};

constexpr std::array<int, 64> kMgRookTable{
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
};

constexpr std::array<int, 64> kEgRookTable{
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
};

constexpr std::array<int, 64> kMgQueenTable{
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};

constexpr std::array<int, 64> kEgQueenTable{
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};

constexpr std::array<int, 64> kMgKingTable{
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};

constexpr std::array<int, 64> kEgKingTable{
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

using PST = std::array<std::array<const int*, 6>, 2>;

const PST& pst() {
  static const PST tables = {{
      {kMgPawnTable.data(), kMgKnightTable.data(), kMgBishopTable.data(),
       kMgRookTable.data(), kMgQueenTable.data(), kMgKingTable.data()},
      {kEgPawnTable.data(), kEgKnightTable.data(), kEgBishopTable.data(),
       kEgRookTable.data(), kEgQueenTable.data(), kEgKingTable.data()},
  }};
  return tables;
}

int pst_value(int phase, chess::Color color, chess::PieceType type,
              chess::Square square) {
  const int sq = phase == 0
                     ? (color == chess::Color::White
                            ? chess::square_index(square) ^ 56
                            : chess::square_index(square))
                     : (color == chess::Color::White
                            ? chess::square_index(square) ^ 56
                            : chess::square_index(square));
  return pst()[phase][static_cast<int>(type)][sq];
}

int eval_side(const chess::Position& position, chess::Color color,
              int phase) {
  int mg = 0;
  int eg = 0;
  const auto& board = position.board();

  for (int pt = 0; pt < 6; ++pt) {
    const auto type = static_cast<chess::PieceType>(pt);
    chess::Bitboard bb = board.pieces_of(color, type);
    for (chess::BitboardIterator it(bb); it.has_next();) {
      const auto sq = it.next();
      const int sq_idx = static_cast<int>(sq);

      const int index = color == chess::Color::White ? (sq_idx ^ 56) : sq_idx;

      mg += kMgPieceValues[pt] + pst()[0][pt][index];
      eg += kEgPieceValues[pt] + pst()[1][pt][index];
    }
  }

  if (phase == 0) {
    return mg;
  }
  return eg;
}

WdlOutput value_to_wdl(double value) {
  const double decisive = std::abs(value);
  const double draw = std::max(0.0, 1.0 - decisive);
  const double win = value > 0.0 ? decisive : 0.0;
  const double loss = value < 0.0 ? decisive : 0.0;
  return {win, draw, loss};
}

}  // namespace

int PestoEvaluator::game_phase(const chess::Position& position) const {
  int phase = 0;
  const auto& board = position.board();
  for (int pt = 0; pt < 5; ++pt) {
    const auto type = static_cast<chess::PieceType>(pt);
    phase += std::popcount(board.pieces_of(chess::Color::White, type)) *
             kGamePhaseInc[pt];
    phase += std::popcount(board.pieces_of(chess::Color::Black, type)) *
             kGamePhaseInc[pt];
  }
  return std::clamp(phase, 0, kTotalPhase);
}

int PestoEvaluator::evaluate_mg(const chess::Position& position,
                                chess::Color perspective) const {
  const int white_mg = eval_side(position, chess::Color::White, 0);
  const int black_mg = eval_side(position, chess::Color::Black, 0);
  const int diff = white_mg - black_mg;
  return perspective == chess::Color::White ? diff : -diff;
}

int PestoEvaluator::evaluate_eg(const chess::Position& position,
                                chess::Color perspective) const {
  const int white_eg = eval_side(position, chess::Color::White, 1);
  const int black_eg = eval_side(position, chess::Color::Black, 1);
  const int diff = white_eg - black_eg;
  return perspective == chess::Color::White ? diff : -diff;
}

int PestoEvaluator::evaluate_tapered(const chess::Position& position,
                                     chess::Color perspective) const {
  const int phase = game_phase(position);
  const int mg = evaluate_mg(position, perspective);
  const int eg = evaluate_eg(position, perspective);
  const int tapered = (mg * phase + eg * (kTotalPhase - phase)) / kTotalPhase;
  return tapered;
}

std::vector<EvaluationResult> PestoEvaluator::evaluate_batch(
    std::span<const EvaluationRequest> requests) {
  std::vector<EvaluationResult> results;
  results.reserve(requests.size());

  for (const auto& request : requests) {
    EvaluationResult result;
    result.policy.reserve(request.legal_moves.size());

    const int cp = evaluate_tapered(request.position, request.side_to_move);

    for (const auto move : request.legal_moves) {
      double logit = 0.0;
      double prior = 1.0 / request.legal_moves.size();

      if (move.is_capture()) {
        logit += 0.5;
      }
      if (move.is_promotion()) {
        logit += 1.0;
      }
      if (move.is_castling()) {
        logit += 0.3;
      }

      result.policy.push_back({move, logit, prior});
    }

    if (!result.policy.empty()) {
      double max_logit = result.policy[0].logit;
      for (const auto& entry : result.policy) {
        max_logit = std::max(max_logit, entry.logit);
      }
      double sum = 0.0;
      for (auto& entry : result.policy) {
        entry.probability = std::exp(entry.logit - max_logit);
        sum += entry.probability;
      }
      for (auto& entry : result.policy) {
        entry.probability /= sum;
      }
    }

    result.value = std::clamp(std::tanh(static_cast<double>(cp) / 600.0), -1.0, 1.0);
    result.wdl = value_to_wdl(result.value);
    result.moves_left = 40.0;

    results.push_back(
        normalize_policy_over_legal_moves(std::move(result), request.legal_moves));
  }

  return results;
}

}  // namespace chessmoe::eval
