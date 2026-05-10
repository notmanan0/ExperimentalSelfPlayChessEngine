#include <chessmoe/chess/fen.h>

#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chessmoe::chess {

namespace {

std::vector<std::string> split_fields(std::string_view text) {
  std::istringstream input{std::string{text}};
  std::vector<std::string> fields;
  std::string field;
  while (input >> field) {
    fields.push_back(field);
  }
  return fields;
}

int parse_nonnegative_int(const std::string& text, std::string_view field_name) {
  if (text.empty()) {
    throw std::runtime_error(std::string{"empty "} + std::string{field_name});
  }
  for (const char c : text) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      throw std::runtime_error(std::string{"invalid "} + std::string{field_name});
    }
  }
  return std::stoi(text);
}

}  // namespace

Position Fen::parse(std::string_view fen) {
  const auto fields = split_fields(fen);
  if (fields.size() != 6) {
    throw std::runtime_error("FEN must contain six fields");
  }

  Position position;
  position.board().clear();

  int rank = 7;
  int file = 0;
  int white_kings = 0;
  int black_kings = 0;

  for (const char c : fields[0]) {
    if (c == '/') {
      if (file != 8) {
        throw std::runtime_error("FEN rank does not contain eight files");
      }
      --rank;
      file = 0;
      continue;
    }

    if (rank < 0) {
      throw std::runtime_error("FEN contains too many ranks");
    }

    if (std::isdigit(static_cast<unsigned char>(c))) {
      const int empty = c - '0';
      if (empty < 1 || empty > 8 || file + empty > 8) {
        throw std::runtime_error("invalid FEN empty-square count");
      }
      file += empty;
      continue;
    }

    const auto piece = piece_from_fen(c);
    if (!piece.has_value() || file >= 8) {
      throw std::runtime_error("invalid FEN piece placement");
    }

    if (piece->type == PieceType::King) {
      if (piece->color == Color::White) {
        ++white_kings;
      } else {
        ++black_kings;
      }
    }

    position.board().set_piece(square_from_index(rank * 8 + file), *piece);
    ++file;
  }

  if (rank != 0 || file != 8 || white_kings != 1 || black_kings != 1) {
    throw std::runtime_error("FEN piece placement is incomplete or invalid");
  }

  if (fields[1] == "w") {
    position.set_side_to_move(Color::White);
  } else if (fields[1] == "b") {
    position.set_side_to_move(Color::Black);
  } else {
    throw std::runtime_error("invalid FEN side to move");
  }

  std::uint8_t castling = 0;
  if (fields[2] != "-") {
    for (const char c : fields[2]) {
      switch (c) {
        case 'K':
          castling |= WhiteKingSide;
          break;
        case 'Q':
          castling |= WhiteQueenSide;
          break;
        case 'k':
          castling |= BlackKingSide;
          break;
        case 'q':
          castling |= BlackQueenSide;
          break;
        default:
          throw std::runtime_error("invalid FEN castling rights");
      }
    }
  }
  position.set_castling_rights(castling);

  if (fields[3] == "-") {
    position.set_en_passant_square(Square::None);
  } else {
    const auto square = square_from_string(fields[3]);
    if (square == Square::None || (rank_of(square) != 2 && rank_of(square) != 5)) {
      throw std::runtime_error("invalid FEN en passant square");
    }
    position.set_en_passant_square(square);
  }

  position.set_halfmove_clock(parse_nonnegative_int(fields[4], "halfmove clock"));
  const int fullmove = parse_nonnegative_int(fields[5], "fullmove number");
  if (fullmove < 1) {
    throw std::runtime_error("FEN fullmove number must be positive");
  }
  position.set_fullmove_number(fullmove);
  position.refresh_hash_and_history();

  return position;
}

std::string Fen::to_string(const Position& position) {
  std::ostringstream output;

  for (int rank = 7; rank >= 0; --rank) {
    int empty = 0;
    for (int file = 0; file < 8; ++file) {
      const auto piece = position.piece_at(square_from_index(rank * 8 + file));
      if (!piece.has_value()) {
        ++empty;
        continue;
      }
      if (empty != 0) {
        output << empty;
        empty = 0;
      }
      output << piece_to_fen(*piece);
    }
    if (empty != 0) {
      output << empty;
    }
    if (rank != 0) {
      output << '/';
    }
  }

  output << ' ' << (position.side_to_move() == Color::White ? 'w' : 'b') << ' ';

  const auto rights = position.castling_rights();
  if (rights == 0) {
    output << '-';
  } else {
    if ((rights & WhiteKingSide) != 0) {
      output << 'K';
    }
    if ((rights & WhiteQueenSide) != 0) {
      output << 'Q';
    }
    if ((rights & BlackKingSide) != 0) {
      output << 'k';
    }
    if ((rights & BlackQueenSide) != 0) {
      output << 'q';
    }
  }

  output << ' ';
  output << (position.en_passant_square() == Square::None
                 ? "-"
                 : square_to_string(position.en_passant_square()));
  output << ' ' << position.halfmove_clock();
  output << ' ' << position.fullmove_number();

  return output.str();
}

}  // namespace chessmoe::chess
