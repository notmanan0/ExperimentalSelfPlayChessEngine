#include <chessmoe/uci/engine_state.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move_generator.h>

namespace chessmoe::uci {

namespace {

constexpr std::string_view kStartFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::vector<std::string> split_words(std::string_view text) {
  std::istringstream input{std::string{text}};
  std::vector<std::string> words;
  std::string word;
  while (input >> word) {
    words.push_back(word);
  }
  return words;
}

}  // namespace

EngineState::EngineState() {
  set_start_position();
}

void EngineState::new_game() {
  set_start_position();
}

void EngineState::set_position_from_command(std::string_view command_tail) {
  const auto words = split_words(command_tail);
  if (words.empty()) {
    throw std::runtime_error("position command missing argument");
  }

  std::size_t index = 0;
  if (words[index] == "startpos") {
    set_start_position();
    ++index;
  } else if (words[index] == "fen") {
    if (words.size() < index + 7) {
      throw std::runtime_error("position fen requires six FEN fields");
    }

    std::string fen = words[index + 1];
    for (std::size_t i = index + 2; i < index + 7; ++i) {
      fen += ' ';
      fen += words[i];
    }
    position_ = chess::Fen::parse(fen);
    index += 7;
  } else {
    throw std::runtime_error("unsupported position command");
  }

  if (index < words.size()) {
    if (words[index] != "moves") {
      throw std::runtime_error("expected moves token in position command");
    }
    ++index;
    while (index < words.size()) {
      apply_uci_move(words[index]);
      ++index;
    }
  }
}

const chess::Position& EngineState::position() const {
  return position_;
}

std::string EngineState::current_fen() const {
  return chess::Fen::to_string(position_);
}

void EngineState::set_start_position() {
  position_ = chess::Fen::parse(kStartFen);
}

void EngineState::apply_uci_move(std::string_view move_text) {
  const auto legal_moves = chess::MoveGenerator::legal_moves(position_);
  for (const auto move : legal_moves) {
    if (move.to_uci() == move_text) {
      position_.make_move(move);
      return;
    }
  }

  throw std::runtime_error("illegal UCI move in position command");
}

}  // namespace chessmoe::uci
