#include <chessmoe/chess/bitboard.h>
#include <chessmoe/chess/board.h>
#include <chessmoe/chess/fen.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

using namespace chessmoe::chess;

void test_square_bb_matches_bitboard() {
  require(square_bb(Square::A1) == 1ULL, "A1 is bit 0");
  require(square_bb(Square::H8) == (1ULL << 63), "H8 is bit 63");
  require(square_bb(Square::E4) == (1ULL << 28), "E4 is bit 28");
  require(square_bb(Square::None) == 0ULL, "None square is zero");
}

void test_popcount() {
  require(popcount(0ULL) == 0, "empty board has 0 bits");
  require(popcount(0xFFFFFFFFFFFFFFFFULL) == 64, "full board has 64 bits");
  require(popcount(0x0101010101010101ULL) == 8, "file A has 8 bits");
  require(popcount(0x00000000000000FFULL) == 8, "rank 1 has 8 bits");
  require(popcount(0x0000001818000000ULL) == 4, "center 4 bits");
}

void test_lsb_msb() {
  require(lsb_index(0x0101010101010101ULL) == 0, "lsb of file A is bit 0");
  require(lsb_index(0x000000000000FF00ULL) == 8, "lsb of rank 2 is bit 8");
  require(msb_index(0x00000000000000FFULL) == 7, "msb of rank 1 is bit 7");
  require(msb_index(0xFF00000000000000ULL) == 63, "msb of rank 8 is bit 63");
  require(lsb_square(0x0000000000000010ULL) == Square::E1, "lsb is E1");
  require(msb_square(0x0000000000000010ULL) == Square::E1, "msb is E1");
}

void test_pop_lsb() {
  Bitboard bb = 0x00000000000000A5ULL;  // bits 0,2,5,7
  require(pop_lsb(bb) == 0x0000000000000001ULL, "first pop gets bit 0");
  require(pop_lsb(bb) == 0x0000000000000004ULL, "second pop gets bit 2");
  require(pop_lsb(bb) == 0x0000000000000020ULL, "third pop gets bit 5");
  require(pop_lsb(bb) == 0x0000000000000080ULL, "fourth pop gets bit 7");
  require(bb == 0ULL, "all bits consumed");
}

void test_file_rank_masks() {
  require(file_mask(0) == kFileA, "file 0 is file A");
  require(file_mask(7) == kFileH, "file 7 is file H");
  require(rank_mask(0) == kRank1, "rank 0 is rank 1");
  require(rank_mask(7) == kRank8, "rank 7 is rank 8");
  require(popcount(file_mask(3)) == 8, "file mask has 8 bits");
  require(popcount(rank_mask(4)) == 8, "rank mask has 8 bits");
}

void test_shift_helpers() {
  const Bitboard e4 = square_bb(Square::E4);
  require(shift_n(e4) == square_bb(Square::E5), "north of E4 is E5");
  require(shift_s(e4) == square_bb(Square::E3), "south of E4 is E3");
  require(shift_e(e4) == square_bb(Square::F4), "east of E4 is F4");
  require(shift_w(e4) == square_bb(Square::D4), "west of E4 is D4");
  require(shift_ne(e4) == square_bb(Square::F5), "NE of E4 is F5");
  require(shift_nw(e4) == square_bb(Square::D5), "NW of E4 is D5");
  require(shift_se(e4) == square_bb(Square::F3), "SE of E4 is F3");
  require(shift_sw(e4) == square_bb(Square::D3), "SW of E4 is D3");

  const Bitboard h1 = square_bb(Square::H1);
  require(shift_e(h1) == 0ULL, "east of H-file wraps to zero");
  require(shift_ne(h1) == 0ULL, "NE of H1 wraps to zero");

  const Bitboard a8 = square_bb(Square::A8);
  require(shift_n(a8) == 0ULL, "north of rank 8 is zero");
}

void test_knight_attacks() {
  const auto attacks = knight_attacks(Square::E4);
  require(popcount(attacks) == 8, "knight on E4 has 8 attacks");
  require((attacks & square_bb(Square::D2)) != 0, "knight E4 attacks D2");
  require((attacks & square_bb(Square::F2)) != 0, "knight E4 attacks F2");
  require((attacks & square_bb(Square::C3)) != 0, "knight E4 attacks C3");
  require((attacks & square_bb(Square::G3)) != 0, "knight E4 attacks G3");
  require((attacks & square_bb(Square::C5)) != 0, "knight E4 attacks C5");
  require((attacks & square_bb(Square::G5)) != 0, "knight E4 attacks G5");
  require((attacks & square_bb(Square::D6)) != 0, "knight E4 attacks D6");
  require((attacks & square_bb(Square::F6)) != 0, "knight E4 attacks F6");

  const auto corner = knight_attacks(Square::A1);
  require(popcount(corner) == 2, "knight on A1 has 2 attacks");
}

void test_king_attacks() {
  const auto attacks = king_attacks(Square::E4);
  require(popcount(attacks) == 8, "king on E4 has 8 attacks");

  const auto corner = king_attacks(Square::A1);
  require(popcount(corner) == 3, "king on A1 has 3 attacks");

  const auto center = king_attacks(Square::D4);
  require((center & square_bb(Square::E4)) != 0, "king D4 attacks E4");
  require((center & square_bb(Square::C3)) != 0, "king D4 attacks C3");
}

void test_bitboard_iterator() {
  Bitboard bb = 0x00000000000000A5ULL;  // bits 0,2,5,7
  BitboardIterator it(bb);
  require(it.has_next(), "iterator has bits");
  require(it.next() == Square::A1, "first bit is A1");
  require(it.next() == Square::C1, "second bit is C1");
  require(it.next() == Square::F1, "third bit is F1");
  require(it.next() == Square::H1, "fourth bit is H1");
  require(!it.has_next(), "iterator exhausted");
}

void test_board_occupancy_with_bitboards() {
  const auto position = Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto& board = position.board();

  const Bitboard white_occ = board.occupancy(Color::White);
  require(popcount(white_occ) == 16, "white has 16 pieces");

  const Bitboard black_occ = board.occupancy(Color::Black);
  require(popcount(black_occ) == 16, "black has 16 pieces");

  const Bitboard all_occ = board.occupancy();
  require(popcount(all_occ) == 32, "32 total pieces");
  require(all_occ == (white_occ | black_occ), "occupancy is union");

  const Bitboard white_pawns = board.pieces_of(Color::White, PieceType::Pawn);
  require(popcount(white_pawns) == 8, "white has 8 pawns");
  require(white_pawns == kRank2, "white pawns on rank 2");
}

void test_piece_iteration_with_bitboard_iterator() {
  const auto position = Fen::parse(
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  const auto& board = position.board();

  const Bitboard white_pawns = board.pieces_of(Color::White, PieceType::Pawn);
  int count = 0;
  for (BitboardIterator it(white_pawns); it.has_next();) {
    const Square sq = it.next();
    const auto piece = board.piece_at(sq);
    require(piece.has_value(), "square has piece");
    require(piece->color == Color::White, "piece is white");
    require(piece->type == PieceType::Pawn, "piece is pawn");
    require(rank_of(sq) == 1, "pawn on rank 2 (index 1)");
    ++count;
  }
  require(count == 8, "iterated over 8 white pawns");
}

}  // namespace

int main() {
  try {
    test_square_bb_matches_bitboard();
    test_popcount();
    test_lsb_msb();
    test_pop_lsb();
    test_file_rank_masks();
    test_shift_helpers();
    test_knight_attacks();
    test_king_attacks();
    test_bitboard_iterator();
    test_board_occupancy_with_bitboards();
    test_piece_iteration_with_bitboard_iterator();
    std::cout << "All bitboard tests passed.\n";
  } catch (const std::exception& e) {
    std::cerr << "bitboard_tests failed: " << e.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
