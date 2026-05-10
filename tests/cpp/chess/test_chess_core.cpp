#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/chess/perft.h>
#include <chessmoe/chess/position.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_eq(std::uint64_t actual, std::uint64_t expected,
                std::string_view message) {
  if (actual != expected) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

void require_eq(std::string_view actual, std::string_view expected,
                std::string_view message) {
  if (actual != expected) {
    throw std::runtime_error(std::string(message) + ": expected `" +
                             std::string(expected) + "`, got `" +
                             std::string(actual) + "`");
  }
}

void test_startpos_fen_round_trips() {
  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  require_eq(chessmoe::chess::Fen::to_string(position),
             "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
             "start position FEN round-trip");
  require(position.side_to_move() == chessmoe::chess::Color::White,
          "start position side to move is white");
  require_eq(position.halfmove_clock(), 0, "start position halfmove clock");
  require_eq(position.fullmove_number(), 1, "start position fullmove number");
}

void test_legal_moves_do_not_leave_king_in_check() {
  const auto position = chessmoe::chess::Fen::parse("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
  const auto moves = chessmoe::chess::MoveGenerator::legal_moves(position);

  require_eq(moves.size(), 3, "king in check has exactly three legal evasions");
  for (const auto move : moves) {
    auto next = position;
    const auto undo = next.make_move(move);
    require(!next.in_check(chessmoe::chess::Color::White),
            "legal move must not leave white in check");
    next.unmake_move(move, undo);
    require_eq(chessmoe::chess::Fen::to_string(next),
               "4k3/8/8/8/8/8/4r3/4K3 w - - 0 1",
               "unmake restores checked position");
  }
}

void test_castling_en_passant_and_promotion_are_generated() {
  {
    const auto position = chessmoe::chess::Fen::parse(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    const auto moves = chessmoe::chess::MoveGenerator::legal_moves(position);
    require(chessmoe::chess::contains_uci(moves, "e1g1"),
            "white king-side castling generated");
    require(chessmoe::chess::contains_uci(moves, "e1c1"),
            "white queen-side castling generated");
  }

  {
    const auto position = chessmoe::chess::Fen::parse("8/8/8/3pP3/8/8/8/4K2k w - d6 0 1");
    const auto moves = chessmoe::chess::MoveGenerator::legal_moves(position);
    require(chessmoe::chess::contains_uci(moves, "e5d6"),
            "en passant capture generated");
  }

  {
    const auto position = chessmoe::chess::Fen::parse("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    const auto moves = chessmoe::chess::MoveGenerator::legal_moves(position);
    require(chessmoe::chess::contains_uci(moves, "a7a8q"),
            "queen promotion generated");
    require(chessmoe::chess::contains_uci(moves, "a7a8n"),
            "knight promotion generated");
  }
}

void test_terminal_detection() {
  const auto checkmate = chessmoe::chess::Fen::parse("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1");
  require(checkmate.is_checkmate(), "checkmate detected");
  require(!checkmate.is_stalemate(), "checkmate is not stalemate");

  const auto stalemate = chessmoe::chess::Fen::parse("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
  require(stalemate.is_stalemate(), "stalemate detected");
  require(!stalemate.is_checkmate(), "stalemate is not checkmate");
}

void test_zobrist_hash_make_unmake_and_repetition_skeleton() {
  auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto initial_hash = position.hash();
  const auto initial_repetitions = position.repetition_count(initial_hash);
  const auto move = chessmoe::chess::Move::from_uci("g1f3", position);

  const auto undo = position.make_move(move);
  require(position.hash() != initial_hash, "hash changes after move");
  position.unmake_move(move, undo);

  require_eq(position.hash(), initial_hash, "hash restored after unmake");
  require_eq(position.repetition_count(initial_hash), initial_repetitions,
             "repetition skeleton restored after unmake");
}

void test_known_perft_positions() {
  struct Case {
    std::string fen;
    int depth;
    std::uint64_t nodes;
  };

  const std::vector<Case> cases = {
      {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 20},
      {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2, 400},
      {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3, 8902},
      {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1, 48},
      {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2, 2039},
      {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1, 14},
      {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2, 191},
  };

  for (const auto& c : cases) {
    require_eq(chessmoe::chess::perft(chessmoe::chess::Fen::parse(c.fen), c.depth),
               c.nodes, "perft depth " + std::to_string(c.depth) + " for " + c.fen);
  }
}

}  // namespace

int main() {
  try {
    test_startpos_fen_round_trips();
    test_legal_moves_do_not_leave_king_in_check();
    test_castling_en_passant_and_promotion_are_generated();
    test_terminal_detection();
    test_zobrist_hash_make_unmake_and_repetition_skeleton();
    test_known_perft_positions();
  } catch (const std::exception& e) {
    std::cerr << "chess_core_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
