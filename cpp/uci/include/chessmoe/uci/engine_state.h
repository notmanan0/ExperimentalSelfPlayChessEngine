#pragma once

#include <string>
#include <string_view>

#include <chessmoe/chess/position.h>

namespace chessmoe::uci {

class EngineState {
 public:
  EngineState();

  void new_game();
  void set_position_from_command(std::string_view command_tail);

  [[nodiscard]] const chess::Position& position() const;
  [[nodiscard]] std::string current_fen() const;

 private:
  void set_start_position();
  void apply_uci_move(std::string_view move_text);

  chess::Position position_;
};

}  // namespace chessmoe::uci
