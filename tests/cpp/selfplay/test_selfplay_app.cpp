#include <chessmoe/chess/fen.h>
#include <chessmoe/eval/evaluator.h>
#include <chessmoe/selfplay/selfplay_app.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void require_contains(std::string_view text, std::string_view needle,
                      std::string_view message) {
  if (text.find(needle) == std::string_view::npos) {
    throw std::runtime_error(std::string(message) + ": missing '" +
                             std::string(needle) + "' in '" +
                             std::string(text) + "'");
  }
}

std::filesystem::path write_config(std::string_view body) {
  const auto path = std::filesystem::temp_directory_path() /
                    "chessmoe_selfplay_app_test_config.json";
  std::ofstream output(path);
  output << body;
  return path;
}

void test_json_config_loads_and_cli_overrides_values() {
  const auto path = write_config(R"json({
    "evaluator": "material",
    "games": 3,
    "concurrent_games": 2,
    "fixed_batch": 8,
    "visits": 5,
    "max_plies": 9,
    "write_replay": true,
    "output_dir": "data/replay/from_config",
    "model_version": 4,
    "progress_interval": 2
  })json");
  const auto path_text = path.string();

  const char* argv[] = {
      "selfplay",
      "--config",
      path_text.c_str(),
      "--games",
      "11",
      "--output-dir",
      "data/replay/from_cli",
  };

  const auto options =
      chessmoe::selfplay::parse_selfplay_options(7, const_cast<char**>(argv));

  require(options.evaluator_mode == chessmoe::selfplay::EvaluatorMode::Material,
          "config selects material evaluator");
  require(options.pipeline.total_games == 11, "CLI games override config");
  require(options.pipeline.concurrent_games == 2,
          "config concurrent games are preserved");
  require(options.pipeline.fixed_batch_size == 8,
          "config fixed batch is loaded");
  require(options.pipeline.game.search_visits == 5, "config visits are loaded");
  require(options.pipeline.game.max_plies == 9, "config max plies are loaded");
  require(options.pipeline.write_replay, "config write_replay is loaded");
  require(options.pipeline.replay_output_dir == "data/replay/from_cli",
          "CLI output dir overrides config");
  require(options.pipeline.replay_options.model_version == 4,
          "config model version is loaded");
  require(options.progress_interval == 2, "config progress interval is loaded");

  std::filesystem::remove(path);
}

void test_material_factory_returns_working_batch_evaluator() {
  chessmoe::selfplay::SelfPlayAppOptions options;
  options.evaluator_mode = chessmoe::selfplay::EvaluatorMode::Material;
  auto evaluator = chessmoe::selfplay::create_batch_evaluator(options);

  const auto position = chessmoe::chess::Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const std::array requests{
      chessmoe::eval::EvaluationRequest::from_position(position)};
  const auto results = evaluator->evaluate_batch(requests);

  require(results.size() == 1, "material evaluator returns one result");
  require(!results.front().policy.empty(),
          "material evaluator returns legal policy entries");
}

void test_tensorrt_requires_engine_path() {
  chessmoe::selfplay::SelfPlayAppOptions options;
  options.evaluator_mode = chessmoe::selfplay::EvaluatorMode::TensorRT;

  try {
    (void)chessmoe::selfplay::create_batch_evaluator(options);
  } catch (const std::exception& e) {
    require_contains(e.what(), "--engine", "TensorRT error names missing flag");
    return;
  }
  throw std::runtime_error("TensorRT without engine path should fail");
}

void test_onnx_mode_reports_unavailable_backend() {
  chessmoe::selfplay::SelfPlayAppOptions options;
  options.evaluator_mode = chessmoe::selfplay::EvaluatorMode::Onnx;

  try {
    (void)chessmoe::selfplay::create_batch_evaluator(options);
  } catch (const std::exception& e) {
    require_contains(e.what(), "ONNX", "ONNX error names backend");
    require_contains(e.what(), "unavailable", "ONNX error is explicit");
    return;
  }
  throw std::runtime_error("ONNX evaluator should be unavailable");
}

void test_zero_progress_format_has_no_divide_by_zero() {
  chessmoe::selfplay::ProgressSnapshot snapshot;
  snapshot.total_games = 0;
  snapshot.completed_games = 0;
  snapshot.elapsed_ms = 0.0;
  snapshot.output_dir = "data/replay/empty";
  snapshot.evaluator_mode = chessmoe::selfplay::EvaluatorMode::Material;

  const auto text = chessmoe::selfplay::format_progress(snapshot);

  require_contains(text, "0/0", "progress includes completed and total games");
  require_contains(text, "0.0%", "progress percentage is finite");
  require_contains(text, "games/sec=0.00", "progress games/sec is finite");
  require_contains(text, "ETA=unknown", "progress ETA is explicit");
}

}  // namespace

int main() {
  try {
    test_json_config_loads_and_cli_overrides_values();
    test_material_factory_returns_working_batch_evaluator();
    test_tensorrt_requires_engine_path();
    test_onnx_mode_reports_unavailable_backend();
    test_zero_progress_format_has_no_divide_by_zero();
  } catch (const std::exception& e) {
    std::cerr << "selfplay_app_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
