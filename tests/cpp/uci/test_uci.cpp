#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>
#include <chessmoe/uci/uci_engine.h>

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

void require_contains(const std::vector<std::string>& lines,
                      std::string_view expected,
                      std::string_view message) {
  for (const auto& line : lines) {
    if (line == expected) {
      return;
    }
  }
  throw std::runtime_error(std::string(message));
}

void require_prefix(const std::vector<std::string>& lines,
                    std::string_view prefix,
                    std::string_view message) {
  for (const auto& line : lines) {
    if (line.rfind(std::string(prefix), 0) == 0) {
      return;
    }
  }
  throw std::runtime_error(std::string(message));
}

void require_line_contains(const std::vector<std::string>& lines,
                           std::string_view needle,
                           std::string_view message) {
  for (const auto& line : lines) {
    if (line.find(needle) != std::string::npos) {
      return;
    }
  }
  throw std::runtime_error(std::string(message));
}

std::string only_bestmove(const std::vector<std::string>& lines) {
  for (const auto& line : lines) {
    if (line.rfind("bestmove ", 0) == 0) {
      return line.substr(std::string{"bestmove "}.size());
    }
  }
  throw std::runtime_error("missing bestmove line");
}

void test_uci_handshake_and_readiness() {
  chessmoe::uci::UciEngine engine;

  const auto uci = engine.handle_line("uci");
  require_prefix(uci, "id name ", "uci response includes engine name");
  require_prefix(uci, "id author ", "uci response includes author");
  require_prefix(uci, "option name ", "uci response includes setoption skeleton");
  require_contains(uci, "uciok", "uci response ends with uciok");

  const auto ready = engine.handle_line("isready");
  require_contains(ready, "readyok", "isready returns readyok");
}

void test_position_startpos_with_moves_updates_state() {
  chessmoe::uci::UciEngine engine;

  (void)engine.handle_line("position startpos moves e2e4 e7e5 g1f3");

  require(engine.current_fen() ==
              "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
          "position startpos moves applies legal UCI moves");
}

void test_position_fen_with_castling_move() {
  chessmoe::uci::UciEngine engine;

  (void)engine.handle_line(
      "position fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1");

  require(engine.current_fen() == "r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1",
          "position fen applies castling move");
}

void test_go_depth_returns_legal_bestmove() {
  chessmoe::uci::UciEngine engine;
  (void)engine.handle_line("position startpos");

  const auto lines = engine.handle_line("go depth 1");
  const auto bestmove = only_bestmove(lines);
  const auto position = chessmoe::chess::Fen::parse(engine.current_fen());

  require(chessmoe::chess::contains_uci(
              chessmoe::chess::MoveGenerator::legal_moves(position), bestmove),
          "go depth 1 returns a legal bestmove");
}

void test_go_nodes_returns_legal_bestmove_and_reports_nodes() {
  chessmoe::uci::UciEngine engine;
  (void)engine.handle_line("position startpos");

  const auto lines = engine.handle_line("go nodes 8");
  require_prefix(lines, "info depth ", "go nodes reports search info");
  require_line_contains(lines, " nodes 8", "go nodes reports requested MCTS visits");
  require_prefix(lines, "bestmove ", "go nodes returns bestmove");

  const auto bestmove = only_bestmove(lines);
  const auto position = chessmoe::chess::Fen::parse(engine.current_fen());
  require(chessmoe::chess::contains_uci(
              chessmoe::chess::MoveGenerator::legal_moves(position), bestmove),
          "go nodes returns a legal bestmove");
}

void test_go_movetime_and_stop_are_supported() {
  chessmoe::uci::UciEngine engine;
  (void)engine.handle_line("position startpos moves e2e4");

  const auto lines = engine.handle_line("go movetime 1");
  require_prefix(lines, "bestmove ", "go movetime returns bestmove");

  const auto stop = engine.handle_line("stop");
  require(stop.empty(), "stop command is accepted without protocol noise");
}

void test_setoption_and_quit() {
  chessmoe::uci::UciEngine engine;

  require(engine.handle_line("setoption name Eval value material").empty(),
          "setoption skeleton accepts known option");
  require(engine.handle_line("quit").empty(), "quit emits no protocol line");
  require(engine.should_quit(), "quit sets quit flag");
}

}  // namespace

int main() {
  try {
    test_uci_handshake_and_readiness();
    test_position_startpos_with_moves_updates_state();
    test_position_fen_with_castling_move();
    test_go_depth_returns_legal_bestmove();
    test_go_nodes_returns_legal_bestmove_and_reports_nodes();
    test_go_movetime_and_stop_are_supported();
    test_setoption_and_quit();
  } catch (const std::exception& e) {
    std::cerr << "uci_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
