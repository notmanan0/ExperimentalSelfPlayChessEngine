#include <chessmoe/selfplay/reanalysis_config.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void test_valid_reanalysis_config_accepts_append_records_mode() {
  chessmoe::selfplay::ReanalysisConfig config;
  config.replay_index = "replay.sqlite";
  config.output_index = "replay.sqlite";
  config.source_model_versions = {1, 2};
  config.current_model_version = 7;
  config.search_budget = 128;
  config.minimum_sampling_priority = 0.2;

  chessmoe::selfplay::validate_reanalysis_config(config);

  require(chessmoe::selfplay::to_string(config.storage_mode) ==
              "append_target_records",
          "default storage mode appends target records");
}

void test_invalid_reanalysis_config_rejects_missing_model_version() {
  chessmoe::selfplay::ReanalysisConfig config;
  config.replay_index = "replay.sqlite";
  config.output_index = "replay.sqlite";
  config.search_budget = 1;

  bool threw = false;
  try {
    chessmoe::selfplay::validate_reanalysis_config(config);
  } catch (const std::invalid_argument&) {
    threw = true;
  }

  require(threw, "current model version is mandatory");
}

}  // namespace

int main() {
  try {
    test_valid_reanalysis_config_accepts_append_records_mode();
    test_invalid_reanalysis_config_rejects_missing_model_version();
  } catch (const std::exception& e) {
    std::cerr << "reanalysis_config_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
