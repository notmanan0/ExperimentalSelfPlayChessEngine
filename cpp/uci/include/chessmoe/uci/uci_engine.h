#pragma once

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

#include <chessmoe/eval/material_evaluator.h>
#include <chessmoe/search/mcts_searcher.h>
#include <chessmoe/uci/engine_state.h>

namespace chessmoe::uci {

class UciEngine {
 public:
  UciEngine();

  [[nodiscard]] std::vector<std::string> handle_line(std::string_view line);
  [[nodiscard]] bool should_quit() const;
  [[nodiscard]] std::string current_fen() const;

 private:
  [[nodiscard]] std::vector<std::string> handle_go(std::string_view command_tail);
  [[nodiscard]] static search::MctsLimits parse_go_limits(std::string_view command_tail);
  [[nodiscard]] static std::string trim(std::string_view text);

  EngineState state_;
  eval::MaterialEvaluator batch_evaluator_;
  eval::SynchronousEvaluator evaluator_;
  search::MctsSearcher searcher_;
  std::atomic_bool stop_requested_{false};
  bool quit_{false};
};

}  // namespace chessmoe::uci
