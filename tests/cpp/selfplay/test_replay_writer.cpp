#include <chessmoe/chess/fen.h>
#include <chessmoe/chess/move.h>
#include <chessmoe/selfplay/replay_writer.h>
#include <chessmoe/selfplay/self_play_generator.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::uint16_t read_u16_le(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  return static_cast<std::uint16_t>(bytes[offset]) |
         (static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

std::uint32_t read_u32_le(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t read_u64_le(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  std::uint64_t value = 0;
  for (int i = 7; i >= 0; --i) {
    value = (value << 8) | bytes[offset + static_cast<std::size_t>(i)];
  }
  return value;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "replay chunk can be opened");
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

void test_writer_emits_chunk_header_and_payload() {
  constexpr std::string_view start_fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  const auto position = chessmoe::chess::Fen::parse(start_fen);

  chessmoe::selfplay::SelfPlaySample sample;
  sample.board_fen = std::string(start_fen);
  sample.side_to_move = chessmoe::chess::Color::White;
  sample.legal_moves = {
      chessmoe::chess::Move::from_uci("e2e4", position),
      chessmoe::chess::Move::from_uci("g1f3", position),
  };
  sample.visit_distribution = {
      {sample.legal_moves[0], 7, 0.7},
      {sample.legal_moves[1], 3, 0.3},
  };
  sample.final_result = chessmoe::selfplay::GameResult::Draw;
  sample.root_value = 0.125;
  sample.search_budget = 10;

  chessmoe::selfplay::SelfPlayGame game;
  game.samples.push_back(sample);
  game.result = chessmoe::selfplay::GameResult::Draw;

  const auto path = std::filesystem::temp_directory_path() /
                    "chessmoe_replay_writer_test.cmrep";
  chessmoe::selfplay::ReplayChunkOptions options;
  options.model_version = 17;
  options.generator_version = 3;
  options.creation_timestamp_ms = 1'715'000'000'000ULL;
  options.game_id = 42;

  chessmoe::selfplay::write_replay_chunk(path, game, options);

  const auto bytes = read_file(path);
  require(bytes.size() > chessmoe::selfplay::kReplayHeaderSize,
          "chunk includes payload after fixed header");
  require(std::string(bytes.begin(), bytes.begin() + 8) == "CMREPLAY",
          "chunk magic is fixed");
  require(read_u16_le(bytes, 8) == chessmoe::selfplay::kReplayFormatVersion,
          "format version is current");
  require(read_u16_le(bytes, 10) == chessmoe::selfplay::kReplayHeaderSize,
          "header size is stored");
  require(read_u32_le(bytes, 16) == 1, "sample count is stored");
  require(read_u32_le(bytes, 24) == 17, "model version is stored");
  require(read_u32_le(bytes, 28) == 3, "generator version is stored");
  require(read_u64_le(bytes, 32) == 1'715'000'000'000ULL,
          "creation timestamp is stored");
  require(read_u64_le(bytes, 40) == bytes.size() -
                                      chessmoe::selfplay::kReplayHeaderSize,
          "payload size matches file size");
  require(read_u32_le(bytes, 48) != 0, "payload checksum is written");

  std::filesystem::remove(path);
}

}  // namespace

int main() {
  try {
    test_writer_emits_chunk_header_and_payload();
  } catch (const std::exception& e) {
    std::cerr << "replay_writer_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
