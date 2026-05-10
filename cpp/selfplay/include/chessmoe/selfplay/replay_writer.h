#pragma once

#include <cstdint>
#include <filesystem>

#include <chessmoe/selfplay/self_play_generator.h>

namespace chessmoe::selfplay {

inline constexpr std::uint16_t kReplayFormatVersion = 1;
inline constexpr std::uint16_t kReplayHeaderSize = 64;

struct ReplayChunkOptions {
  std::uint32_t model_version{0};
  std::uint32_t generator_version{0};
  std::uint64_t creation_timestamp_ms{0};
  std::uint64_t game_id{0};
  std::uint32_t starting_ply_index{0};
  bool compressed{false};
};

void write_replay_chunk(const std::filesystem::path& path,
                        const SelfPlayGame& game,
                        ReplayChunkOptions options);

}  // namespace chessmoe::selfplay
