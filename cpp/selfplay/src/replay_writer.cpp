#include <chessmoe/selfplay/replay_writer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <chessmoe/chess/fen.h>

namespace chessmoe::selfplay {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic{'C', 'M', 'R', 'E',
                                             'P', 'L', 'A', 'Y'};

void append_u8(std::vector<std::uint8_t>& bytes, std::uint8_t value) {
  bytes.push_back(value);
}

void append_u16(std::vector<std::uint8_t>& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value & 0xFFU));
  bytes.push_back(static_cast<std::uint8_t>((value >> 8U) & 0xFFU));
}

void append_u32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFU));
  }
}

void append_u64(std::vector<std::uint8_t>& bytes, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xFFULL));
  }
}

void append_f32(std::vector<std::uint8_t>& bytes, float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t raw = 0;
  std::memcpy(&raw, &value, sizeof(float));
  append_u32(bytes, raw);
}

std::uint64_t now_ms() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::uint8_t piece_code(chess::Piece piece) {
  return static_cast<std::uint8_t>(
      1 + chess::color_index(piece.color) * 6 + chess::piece_index(piece.type));
}

std::uint8_t result_code(GameResult result) {
  switch (result) {
    case GameResult::BlackWin:
      return 0;
    case GameResult::Draw:
      return 1;
    case GameResult::WhiteWin:
      return 2;
    case GameResult::Unknown:
      return 3;
  }
  return 3;
}

std::uint16_t checked_u16(int value, const char* field) {
  if (value < 0 || value > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error(std::string(field) + " exceeds uint16 range");
  }
  return static_cast<std::uint16_t>(value);
}

std::uint32_t checked_u32(int value, const char* field) {
  if (value < 0) {
    throw std::runtime_error(std::string(field) + " cannot be negative");
  }
  return static_cast<std::uint32_t>(value);
}

std::uint16_t promotion_code(chess::Move move) {
  if (!move.is_promotion()) {
    return 0;
  }
  switch (move.promotion) {
    case chess::PieceType::Knight:
      return 1;
    case chess::PieceType::Bishop:
      return 2;
    case chess::PieceType::Rook:
      return 3;
    case chess::PieceType::Queen:
      return 4;
    case chess::PieceType::Pawn:
    case chess::PieceType::King:
      break;
  }
  throw std::runtime_error("invalid promotion piece in replay move");
}

std::uint16_t encode_move(chess::Move move) {
  if (move.from == chess::Square::None || move.to == chess::Square::None) {
    throw std::runtime_error("cannot encode null replay move");
  }
  return static_cast<std::uint16_t>(
      chess::square_index(move.from) |
      (chess::square_index(move.to) << 6) |
      (promotion_code(move) << 12));
}

void append_board(std::vector<std::uint8_t>& bytes,
                  const chess::Position& position) {
  for (int square = 0; square < 64; ++square) {
    const auto piece = position.piece_at(chess::square_from_index(square));
    append_u8(bytes, piece.has_value() ? piece_code(*piece) : 0);
  }
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes) {
  std::uint32_t crc = 0xFFFFFFFFU;
  for (const auto byte : bytes) {
    crc ^= byte;
    for (int bit = 0; bit < 8; ++bit) {
      const std::uint32_t mask = 0U - (crc & 1U);
      crc = (crc >> 1U) ^ (0xEDB88320U & mask);
    }
  }
  return ~crc;
}

void append_sample(std::vector<std::uint8_t>& payload,
                   const SelfPlaySample& sample,
                   std::uint64_t game_id,
                   std::uint32_t ply_index) {
  const auto position = chess::Fen::parse(sample.board_fen);
  std::vector<std::uint8_t> body;
  body.reserve(96 + sample.legal_moves.size() * 2 +
               sample.visit_distribution.size() * 10);

  append_board(body, position);
  append_u8(body, position.side_to_move() == chess::Color::White ? 0 : 1);
  append_u8(body, position.castling_rights());
  append_u8(body, position.en_passant_square() == chess::Square::None
                      ? 64
                      : static_cast<std::uint8_t>(
                            chess::square_index(position.en_passant_square())));
  append_u16(body, checked_u16(position.halfmove_clock(), "halfmove clock"));
  append_u16(body, checked_u16(position.fullmove_number(), "fullmove number"));
  append_u8(body, result_code(sample.final_result));
  append_f32(body, static_cast<float>(sample.root_value));
  append_u32(body, checked_u32(sample.search_budget, "search budget"));
  append_u64(body, game_id);
  append_u32(body, ply_index);

  if (sample.legal_moves.size() >
      std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("too many legal moves for replay sample");
  }
  if (sample.visit_distribution.size() >
      std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("too many policy entries for replay sample");
  }
  append_u16(body, static_cast<std::uint16_t>(sample.legal_moves.size()));
  append_u16(body,
             static_cast<std::uint16_t>(sample.visit_distribution.size()));

  for (const auto move : sample.legal_moves) {
    append_u16(body, encode_move(move));
  }
  for (const auto& entry : sample.visit_distribution) {
    append_u16(body, encode_move(entry.move));
    append_u32(body, checked_u32(entry.visit_count, "visit count"));
    append_f32(body, static_cast<float>(entry.probability));
  }

  if (body.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("replay sample is too large");
  }
  append_u32(payload, static_cast<std::uint32_t>(body.size()));
  payload.insert(payload.end(), body.begin(), body.end());
}

}  // namespace

void write_replay_chunk(const std::filesystem::path& path,
                        const SelfPlayGame& game,
                        ReplayChunkOptions options) {
  if (options.compressed) {
    throw std::runtime_error("replay compression is reserved but not implemented");
  }
  if (game.samples.size() > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("too many samples for replay chunk");
  }

  std::vector<std::uint8_t> payload;
  for (std::size_t i = 0; i < game.samples.size(); ++i) {
    const auto ply_index = options.starting_ply_index +
                           static_cast<std::uint32_t>(i);
    append_sample(payload, game.samples[i], options.game_id, ply_index);
  }

  if (payload.size() > std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("replay payload is too large");
  }

  std::vector<std::uint8_t> header;
  header.reserve(kReplayHeaderSize);
  header.insert(header.end(), kMagic.begin(), kMagic.end());
  append_u16(header, kReplayFormatVersion);
  append_u16(header, kReplayHeaderSize);
  append_u32(header, 0);
  append_u32(header, static_cast<std::uint32_t>(game.samples.size()));
  append_u32(header, 0);
  append_u32(header, options.model_version);
  append_u32(header, options.generator_version);
  append_u64(header, options.creation_timestamp_ms == 0
                         ? now_ms()
                         : options.creation_timestamp_ms);
  append_u64(header, static_cast<std::uint64_t>(payload.size()));
  append_u32(header, crc32(payload));
  header.resize(kReplayHeaderSize, 0);

  std::ofstream output(path, std::ios::binary);
  if (!output.good()) {
    throw std::runtime_error("failed to open replay chunk for writing");
  }
  output.write(reinterpret_cast<const char*>(header.data()),
               static_cast<std::streamsize>(header.size()));
  output.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
  if (!output.good()) {
    throw std::runtime_error("failed to write replay chunk");
  }
}

}  // namespace chessmoe::selfplay
