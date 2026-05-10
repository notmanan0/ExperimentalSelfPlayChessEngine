#include <chessmoe/uci/uci_engine.h>

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace chessmoe::uci {

namespace {

bool starts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

}  // namespace

UciEngine::UciEngine()
    : evaluator_(batch_evaluator_), searcher_(evaluator_) {}

std::vector<std::string> UciEngine::handle_line(std::string_view line) {
  const std::string command = trim(line);
  if (command.empty()) {
    return {};
  }

  try {
    if (command == "uci") {
      return {
          "id name chessmoe phase4",
          "id author chessmoe",
          "option name Eval type combo default material var material",
          "uciok",
      };
    }
    if (command == "isready") {
      return {"readyok"};
    }
    if (command == "ucinewgame") {
      state_.new_game();
      stop_requested_.store(false, std::memory_order_relaxed);
      return {};
    }
    if (starts_with(command, "position ")) {
      state_.set_position_from_command(std::string_view{command}.substr(9));
      return {};
    }
    if (starts_with(command, "go")) {
      return handle_go(command.size() > 2 ? std::string_view{command}.substr(2) : "");
    }
    if (command == "stop") {
      stop_requested_.store(true, std::memory_order_relaxed);
      return {};
    }
    if (starts_with(command, "setoption ")) {
      return {};
    }
    if (command == "quit") {
      stop_requested_.store(true, std::memory_order_relaxed);
      quit_ = true;
      return {};
    }
  } catch (const std::exception& e) {
    return {std::string{"info string error: "} + e.what()};
  }

  return {std::string{"info string ignored unsupported command: "} + command};
}

bool UciEngine::should_quit() const {
  return quit_;
}

std::string UciEngine::current_fen() const {
  return state_.current_fen();
}

std::vector<std::string> UciEngine::handle_go(std::string_view command_tail) {
  const auto limits = parse_go_limits(command_tail);
  auto result = searcher_.search(state_.position(), limits);

  std::vector<std::string> lines;
  const int reported_depth = limits.max_depth > 0 ? limits.max_depth : 1;
  const int score_cp = static_cast<int>(result.root_value * 1000.0);
  lines.push_back("info depth " + std::to_string(reported_depth) + " score cp " +
                  std::to_string(score_cp) + " nodes " +
                  std::to_string(result.root_visits));
  lines.push_back("bestmove " +
                  (result.has_best_move ? result.best_move.to_uci() : "0000"));
  return lines;
}

search::MctsLimits UciEngine::parse_go_limits(std::string_view command_tail) {
  search::MctsLimits limits;
  std::istringstream input{std::string{command_tail}};
  std::string token;
  while (input >> token) {
    if (token == "depth") {
      if (!(input >> limits.max_depth) || limits.max_depth < 1) {
        limits.max_depth = 1;
      }
      if (limits.visits < 1) {
        limits.visits = 1;
      }
      limits.visits = std::max(limits.visits, limits.max_depth * 32);
    } else if (token == "nodes") {
      if (!(input >> limits.visits) || limits.visits < 1) {
        limits.visits = 1;
      }
    } else if (token == "movetime") {
      int ignored_movetime_ms = 0;
      input >> ignored_movetime_ms;
      limits.visits = std::max(limits.visits, 32);
    }
  }
  return limits;
}

std::string UciEngine::trim(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return std::string{text.substr(first, last - first + 1)};
}

}  // namespace chessmoe::uci
