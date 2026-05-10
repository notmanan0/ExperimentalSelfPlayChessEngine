#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include <chessmoe/chess/move.h>
#include <chessmoe/chess/position.h>

namespace chessmoe::inference {

struct TensorLayout {
  std::string_view input_name{"board"};
  std::string_view policy_output_name{"policy_logits"};
  std::string_view wdl_output_name{"wdl_logits"};
  std::string_view moves_left_output_name{"moves_left"};
  std::size_t channels{18};
  std::size_t height{8};
  std::size_t width{8};
  std::size_t policy_buckets{20480};

  [[nodiscard]] static constexpr TensorLayout tiny_baseline() {
    return TensorLayout{};
  }

  [[nodiscard]] constexpr std::size_t input_elements_per_position() const {
    return channels * height * width;
  }
};

[[nodiscard]] std::size_t policy_index_from_uci(std::string_view uci);
[[nodiscard]] std::size_t policy_index_from_move(const chess::Move& move);
[[nodiscard]] std::vector<float> encode_position_nchw(
    const chess::Position& position, const TensorLayout& layout);

}  // namespace chessmoe::inference
