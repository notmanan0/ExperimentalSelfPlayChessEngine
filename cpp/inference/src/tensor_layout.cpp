#include <chessmoe/inference/tensor_layout.h>

#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace chessmoe::inference {
namespace {

constexpr std::size_t kFromToMoves = 64 * 64;
constexpr std::array<char, 4> kPromotionPieces{'q', 'r', 'b', 'n'};

int square_index_from_text(std::string_view square) {
  if (square.size() != 2 || square[0] < 'a' || square[0] > 'h' ||
      square[1] < '1' || square[1] > '8') {
    throw std::invalid_argument("invalid square text");
  }
  const int file = square[0] - 'a';
  const int rank = square[1] - '1';
  return rank * 8 + file;
}

std::size_t promotion_offset(char promotion) {
  for (std::size_t i = 0; i < kPromotionPieces.size(); ++i) {
    if (promotion == kPromotionPieces[i]) {
      return i;
    }
  }
  throw std::invalid_argument("invalid promotion piece");
}

std::size_t plane_offset(const TensorLayout& layout, std::size_t channel,
                         std::size_t rank, std::size_t file) {
  return channel * layout.height * layout.width + rank * layout.width + file;
}

std::size_t piece_channel(chess::Piece piece) {
  const std::size_t base = piece.color == chess::Color::White ? 0 : 6;
  return base + static_cast<std::size_t>(piece.type);
}

}  // namespace

std::size_t policy_index_from_uci(std::string_view uci) {
  if (uci.size() != 4 && uci.size() != 5) {
    throw std::invalid_argument("invalid UCI move length");
  }

  const std::size_t from = static_cast<std::size_t>(square_index_from_text(uci.substr(0, 2)));
  const std::size_t to = static_cast<std::size_t>(square_index_from_text(uci.substr(2, 2)));
  const std::size_t base = from * 64 + to;

  if (uci.size() == 4) {
    return base;
  }

  return kFromToMoves + promotion_offset(uci[4]) * kFromToMoves + base;
}

std::size_t policy_index_from_move(const chess::Move& move) {
  return policy_index_from_uci(move.to_uci());
}

std::vector<float> encode_position_nchw(const chess::Position& position,
                                        const TensorLayout& layout) {
  if (layout.channels != 18 || layout.height != 8 || layout.width != 8) {
    throw std::invalid_argument("only the tiny baseline 18x8x8 layout is supported");
  }

  std::vector<float> encoded(layout.input_elements_per_position(), 0.0F);

  for (int index = 0; index < 64; ++index) {
    const auto square = chess::square_from_index(index);
    const auto piece = position.piece_at(square);
    if (!piece.has_value()) {
      continue;
    }
    encoded[plane_offset(layout, piece_channel(*piece),
                         static_cast<std::size_t>(chess::rank_of(square)),
                         static_cast<std::size_t>(chess::file_of(square)))] = 1.0F;
  }

  if (position.side_to_move() == chess::Color::White) {
    for (std::size_t i = 0; i < layout.height * layout.width; ++i) {
      encoded[plane_offset(layout, 12, 0, 0) + i] = 1.0F;
    }
  }

  const std::array<std::pair<chess::CastlingRight, std::size_t>, 4> castling{
      std::pair<chess::CastlingRight, std::size_t>{chess::WhiteKingSide, 13},
      std::pair<chess::CastlingRight, std::size_t>{chess::WhiteQueenSide, 14},
      std::pair<chess::CastlingRight, std::size_t>{chess::BlackKingSide, 15},
      std::pair<chess::CastlingRight, std::size_t>{chess::BlackQueenSide, 16},
  };
  for (const auto [right, channel] : castling) {
    if (!position.can_castle(right)) {
      continue;
    }
    for (std::size_t i = 0; i < layout.height * layout.width; ++i) {
      encoded[plane_offset(layout, channel, 0, 0) + i] = 1.0F;
    }
  }

  const auto en_passant = position.en_passant_square();
  if (en_passant != chess::Square::None) {
    encoded[plane_offset(layout, 17,
                         static_cast<std::size_t>(chess::rank_of(en_passant)),
                         static_cast<std::size_t>(chess::file_of(en_passant)))] = 1.0F;
  }

  return encoded;
}

}  // namespace chessmoe::inference
