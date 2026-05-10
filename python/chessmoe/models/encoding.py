from __future__ import annotations

from dataclasses import dataclass

import torch

BOARD_CHANNELS = 18
BOARD_SIZE = 8
BOARD_SHAPE = (BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)

PROMOTION_PIECES = ("q", "r", "b", "n")
NUM_FROM_TO_MOVES = 64 * 64
NUM_PROMOTION_MOVES = 64 * 64 * len(PROMOTION_PIECES)
NUM_MOVE_BUCKETS = NUM_FROM_TO_MOVES + NUM_PROMOTION_MOVES

PIECE_TO_CHANNEL = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


@dataclass(frozen=True)
class FenParts:
    placement: str
    side_to_move: str
    castling: str
    en_passant: str
    halfmove: int
    fullmove: int


def parse_fen(fen: str) -> FenParts:
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must contain six fields")
    return FenParts(
        placement=fields[0],
        side_to_move=fields[1],
        castling=fields[2],
        en_passant=fields[3],
        halfmove=int(fields[4]),
        fullmove=int(fields[5]),
    )


def square_to_index(square: str) -> int:
    if len(square) != 2 or square[0] not in "abcdefgh" or square[1] not in "12345678":
        raise ValueError(f"invalid square: {square}")
    file_idx = ord(square[0]) - ord("a")
    rank_idx = int(square[1]) - 1
    return rank_idx * 8 + file_idx


def move_to_index(uci: str) -> int:
    if len(uci) not in (4, 5):
        raise ValueError(f"invalid UCI move: {uci}")
    from_sq = square_to_index(uci[:2])
    to_sq = square_to_index(uci[2:4])
    base = from_sq * 64 + to_sq
    if len(uci) == 4:
        return base
    promotion = uci[4].lower()
    if promotion not in PROMOTION_PIECES:
        raise ValueError(f"invalid promotion piece: {uci}")
    promotion_offset = PROMOTION_PIECES.index(promotion)
    return NUM_FROM_TO_MOVES + promotion_offset * NUM_FROM_TO_MOVES + base


def encode_fen(fen: str) -> torch.Tensor:
    """Encode FEN to float32 tensor shaped [18, 8, 8].

    Channels 0-11 are piece planes, 12 is side-to-move, 13-16 are KQkq
    castling rights, and 17 marks the en-passant target square when present.
    Row 0 corresponds to rank 1, column 0 to file a.
    """
    parts = parse_fen(fen)
    tensor = torch.zeros(BOARD_SHAPE, dtype=torch.float32)

    ranks = parts.placement.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN placement must have eight ranks")

    for fen_rank, rank_text in enumerate(ranks):
        rank = 7 - fen_rank
        file_idx = 0
        for char in rank_text:
            if char.isdigit():
                file_idx += int(char)
                continue
            if char not in PIECE_TO_CHANNEL:
                raise ValueError(f"invalid FEN piece: {char}")
            if file_idx >= 8:
                raise ValueError("too many files in FEN rank")
            tensor[PIECE_TO_CHANNEL[char], rank, file_idx] = 1.0
            file_idx += 1
        if file_idx != 8:
            raise ValueError("FEN rank does not contain eight files")

    if parts.side_to_move == "w":
        tensor[12].fill_(1.0)
    elif parts.side_to_move != "b":
        raise ValueError("FEN side to move must be w or b")

    for offset, flag in enumerate("KQkq", start=13):
        if flag in parts.castling:
            tensor[offset].fill_(1.0)

    if parts.en_passant != "-":
        square = square_to_index(parts.en_passant)
        tensor[17, square // 8, square % 8] = 1.0

    return tensor


def policy_target_from_visits(visits: dict[str, float]) -> torch.Tensor:
    target = torch.zeros(NUM_MOVE_BUCKETS, dtype=torch.float32)
    total = float(sum(max(0.0, value) for value in visits.values()))
    if total <= 0.0:
        raise ValueError("visit distribution must have positive mass")
    for move, count in visits.items():
        target[move_to_index(move)] = max(0.0, float(count)) / total
    return target

