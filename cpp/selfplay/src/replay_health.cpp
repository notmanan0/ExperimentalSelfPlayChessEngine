#include <chessmoe/selfplay/replay_health.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

#include <chessmoe/selfplay/replay_writer.h>

namespace chessmoe::selfplay {
namespace {

struct GameStats {
  int samples{0};
  int terminal_reason{0};  // 0=checkmate,1=stalemate,2=repetition,3=50move,4=maxplies
  bool is_draw{false};
};

}  // namespace

ReplayHealthReport analyze_replay_health(
    const std::vector<std::filesystem::path>& paths,
    const ReplayHealthConfig& config) {
  ReplayHealthReport report;

  int checkmate = 0;
  int stalemate = 0;
  int repetition = 0;
  int fifty_move = 0;
  int max_plies = 0;
  int draws = 0;
  int total_games = 0;
  int total_samples = 0;

  for (const auto& path : paths) {
    if (!std::filesystem::exists(path)) {
      report.warnings.push_back("missing replay file: " + path.string());
      continue;
    }

    const auto file_size = std::filesystem::file_size(path);
    if (file_size < config.min_file_bytes) {
      report.warnings.push_back("tiny replay file: " + path.string() +
                                " (" + std::to_string(file_size) + " bytes)");
    }

    if (file_size < static_cast<std::uint64_t>(kReplayHeaderSize)) {
      report.errors.push_back("file too small for header: " + path.string());
      report.passed = false;
      continue;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
      report.errors.push_back("cannot open: " + path.string());
      report.passed = false;
      continue;
    }

    std::vector<std::uint8_t> data(file_size);
    input.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(file_size));

    if (data.size() < kReplayHeaderSize) {
      continue;
    }

    std::uint32_t sample_count = 0;
    std::memcpy(&sample_count, data.data() + 12, sizeof(std::uint32_t));

    total_games++;
    total_samples += static_cast<int>(sample_count);

    if (sample_count > 0) {
      const std::uint8_t last_result = data[data.size() - 1];
      if (last_result == 1) {
        draws++;
      }
    }
  }

  report.total_games = total_games;
  report.total_samples = total_samples;

  if (total_games > 0) {
    report.average_plies =
        static_cast<double>(total_samples) / static_cast<double>(total_games);
    report.draw_rate = static_cast<double>(draws) / static_cast<double>(total_games);
  }

  if (total_samples < config.min_samples) {
    report.warnings.push_back(
        "total samples (" + std::to_string(total_samples) +
        ") below minimum (" + std::to_string(config.min_samples) + ")");
  }
  if (report.average_plies < config.min_average_plies && total_games > 0) {
    report.warnings.push_back(
        "average plies/game (" + std::to_string(report.average_plies) +
        ") below minimum (" + std::to_string(config.min_average_plies) + ")");
  }
  if (report.draw_rate > config.max_draw_rate && total_games > 10) {
    report.warnings.push_back(
        "draw rate (" + std::to_string(report.draw_rate) +
        ") exceeds threshold (" + std::to_string(config.max_draw_rate) + ")");
  }

  return report;
}

void write_replay_health_report(const ReplayHealthReport& report,
                                const std::filesystem::path& path) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) {
    return;
  }

  out << "{\n";
  out << "  \"total_samples\": " << report.total_samples << ",\n";
  out << "  \"total_games\": " << report.total_games << ",\n";
  out << "  \"average_plies\": " << report.average_plies << ",\n";
  out << "  \"draw_rate\": " << report.draw_rate << ",\n";
  out << "  \"checkmate_rate\": " << report.checkmate_rate << ",\n";
  out << "  \"stalemate_rate\": " << report.stalemate_rate << ",\n";
  out << "  \"repetition_rate\": " << report.repetition_rate << ",\n";
  out << "  \"fifty_move_rate\": " << report.fifty_move_rate << ",\n";
  out << "  \"max_plies_rate\": " << report.max_plies_rate << ",\n";
  out << "  \"passed\": " << (report.passed ? "true" : "false") << ",\n";

  out << "  \"warnings\": [";
  for (std::size_t i = 0; i < report.warnings.size(); ++i) {
    if (i > 0) out << ", ";
    out << "\"" << report.warnings[i] << "\"";
  }
  out << "],\n";

  out << "  \"errors\": [";
  for (std::size_t i = 0; i < report.errors.size(); ++i) {
    if (i > 0) out << ", ";
    out << "\"" << report.errors[i] << "\"";
  }
  out << "]\n";
  out << "}\n";
}

void print_replay_health_summary(const ReplayHealthReport& report) {
  std::cout << "=== Replay Health ===" << '\n';
  std::cout << "Total games: " << report.total_games << '\n';
  std::cout << "Total samples: " << report.total_samples << '\n';
  std::cout << "Average plies/game: " << report.average_plies << '\n';
  std::cout << "Draw rate: " << report.draw_rate << '\n';
  std::cout << "Status: " << (report.passed ? "PASSED" : "FAILED") << '\n';

  if (!report.warnings.empty()) {
    std::cout << "Warnings:" << '\n';
    for (const auto& w : report.warnings) {
      std::cout << "  - " << w << '\n';
    }
  }
  if (!report.errors.empty()) {
    std::cout << "Errors:" << '\n';
    for (const auto& e : report.errors) {
      std::cout << "  - " << e << '\n';
    }
  }
}

}  // namespace chessmoe::selfplay
