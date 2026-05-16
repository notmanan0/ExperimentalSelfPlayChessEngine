#include <chessmoe/chess/fen.h>
#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/eval/pesto_evaluator.h>
#include <chessmoe/search/alphabeta_searcher.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

using namespace chessmoe;

void test_search_returns_legal_move() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 3;

  const auto result = searcher.search(position, limits, stop);

  require(result.has_best_move, "search finds a best move");
  require(result.depth_searched == 3, "search reaches requested depth");
  require(result.nodes > 0, "search visits nodes");
}

void test_checkmate_returns_high_score() {
  const auto position = chess::Fen::parse(
      "6k1/5ppp/8/8/8/8/8/4K2R w - - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 4;

  const auto result = searcher.search(position, limits, stop);

  require(result.has_best_move, "white finds a move");
  require(result.score_cp > 100, "rook advantage gives positive score");
}

void test_stalemate_returns_zero() {
  const auto position = chess::Fen::parse(
      "k7/8/1K6/8/8/8/8/8 b - - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 2;

  const auto result = searcher.search(position, limits, stop);

  require(result.has_best_move, "stalemate still returns a move");
  require(result.score_cp == 0, "stalemate score is zero");
}

void test_deterministic_results() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::MaterialEvaluator evaluator;
  std::atomic_bool stop1{false}, stop2{false};

  search::AlphaBetaLimits limits;
  limits.depth = 3;

  search::AlphaBetaSearcher s1(evaluator);
  search::AlphaBetaSearcher s2(evaluator);

  const auto r1 = s1.search(position, limits, stop1);
  const auto r2 = s2.search(position, limits, stop2);

  require(r1.best_move == r2.best_move, "deterministic best move");
  require(r1.score_cp == r2.score_cp, "deterministic score");
}

void test_pesto_evaluator_works() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::PestoEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 3;

  const auto result = searcher.search(position, limits, stop);

  require(result.has_best_move, "pesto search finds a best move");
}

void test_root_moves_populated() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 2;

  const auto result = searcher.search(position, limits, stop);

  require(result.root_moves.size() == 20, "20 root moves scored");
  require(result.root_moves[0].score_cp >= result.root_moves[1].score_cp,
          "root moves sorted by score");
}

void test_depth_limited_search() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{false};

  search::AlphaBetaLimits limits;
  limits.depth = 1;

  const auto result = searcher.search(position, limits, stop);

  require(result.has_best_move, "depth 1 search returns a move");
  require(result.depth_searched == 1, "searched exactly depth 1");
}

void test_stop_requested_halts_search() {
  const auto position = chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  eval::MaterialEvaluator evaluator;
  search::AlphaBetaSearcher searcher(evaluator);
  std::atomic_bool stop{true};

  search::AlphaBetaLimits limits;
  limits.depth = 6;

  const auto result = searcher.search(position, limits, stop);

  require(result.stopped, "search was stopped");
}

}  // namespace

int main() {
  try {
    test_search_returns_legal_move();
    test_checkmate_returns_high_score();
    test_stalemate_returns_zero();
    test_deterministic_results();
    test_pesto_evaluator_works();
    test_root_moves_populated();
    test_depth_limited_search();
    test_stop_requested_halts_search();
    std::cout << "All alpha-beta tests passed.\n";
  } catch (const std::exception& e) {
    std::cerr << "alphabeta_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
