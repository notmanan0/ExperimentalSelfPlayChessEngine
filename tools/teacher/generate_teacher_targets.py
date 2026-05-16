"""Teacher target generation using PeSTO alpha-beta search.

Reads FEN positions and produces JSONL targets with soft policy and value
derived from a classical alpha-beta search with PeSTO evaluation.

Usage:
    python tools/teacher/generate_teacher_targets.py \
        --fen-file data/openings/bootstrap_fens.txt \
        --output data/teacher/pesto_ab_targets.jsonl \
        --depth 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

try:
    import chess
except ImportError:
    print("ERROR: python-chess is required. Install with: pip install python-chess")
    sys.exit(1)

# ---------------------------------------------------------------------------
# PeSTO Piece-Square Tables (from https://www.chessprogramming.org/PeSTO's_Evaluation_Function)
# ---------------------------------------------------------------------------

MG_PAWN = [
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
]

EG_PAWN = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

MG_KNIGHT = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]

EG_KNIGHT = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

MG_BISHOP = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]

EG_BISHOP = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
]

MG_ROOK = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]

EG_ROOK = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]

MG_QUEEN = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]

EG_QUEEN = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

MG_KING = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]

EG_KING = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

MG_PST = {
    chess.PAWN: MG_PAWN, chess.KNIGHT: MG_KNIGHT, chess.BISHOP: MG_BISHOP,
    chess.ROOK: MG_ROOK, chess.QUEEN: MG_QUEEN, chess.KING: MG_KING,
}
EG_PST = {
    chess.PAWN: EG_PAWN, chess.KNIGHT: EG_KNIGHT, chess.BISHOP: EG_BISHOP,
    chess.ROOK: EG_ROOK, chess.QUEEN: EG_QUEEN, chess.KING: EG_KING,
}

MG_VALUES = {chess.PAWN: 82, chess.KNIGHT: 337, chess.BISHOP: 365,
             chess.ROOK: 477, chess.QUEEN: 1025, chess.KING: 0}
EG_VALUES = {chess.PAWN: 94, chess.KNIGHT: 281, chess.BISHOP: 297,
             chess.ROOK: 512, chess.QUEEN: 936, chess.KING: 0}

PHASE_INC = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 1,
             chess.ROOK: 2, chess.QUEEN: 4, chess.KING: 0}
TOTAL_PHASE = sum(PHASE_INC.values()) * 4  # 24


def pst_index(square: int, color: chess.Color) -> int:
    if color == chess.WHITE:
        return square ^ 56
    return square


def evaluate_board(board: chess.Board) -> int:
    phase = 0
    mg_score = 0
    eg_score = 0

    for piece_type in chess.PIECE_TYPES:
        for sq in board.pieces(piece_type, chess.WHITE):
            idx = pst_index(sq, chess.WHITE)
            mg_score += MG_VALUES[piece_type] + MG_PST[piece_type][idx]
            eg_score += EG_VALUES[piece_type] + EG_PST[piece_type][idx]
            phase += PHASE_INC[piece_type]
        for sq in board.pieces(piece_type, chess.BLACK):
            idx = pst_index(sq, chess.BLACK)
            mg_score -= MG_VALUES[piece_type] + MG_PST[piece_type][idx]
            eg_score -= EG_VALUES[piece_type] + EG_PST[piece_type][idx]
            phase += PHASE_INC[piece_type]

    phase = min(phase, TOTAL_PHASE)
    score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    if board.turn == chess.BLACK:
        score = -score
    return score


def quiescence(board: chess.Board, alpha: int, beta: int, nodes: list[int]) -> int:
    nodes[0] += 1
    stand_pat = evaluate_board(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha, nodes)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


MATE_SCORE = 30000


def alpha_beta(board: chess.Board, depth: int, alpha: int, beta: int,
               nodes: list[int]) -> int:
    if depth <= 0:
        return quiescence(board, alpha, beta, nodes)

    nodes[0] += 1
    in_check = board.is_check()
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        if in_check:
            return -MATE_SCORE
        return 0

    if board.is_repetition(2):
        return 0

    best_score = -MATE_SCORE - 1

    for move in legal_moves:
        board.push(move)
        score = -alpha_beta(board, depth - 1, -beta, -alpha, nodes)
        board.pop()

        if score > best_score:
            best_score = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    return best_score


def search_root(board: chess.Board, depth: int) -> tuple[chess.Move, int, list[tuple[str, int]], int]:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return chess.Move.null(), 0, [], 0

    nodes = [0]
    best_move = legal_moves[0]
    best_score = -MATE_SCORE - 1
    move_scores = []

    for move in legal_moves:
        board.push(move)
        score = -alpha_beta(board, depth - 1, -MATE_SCORE - 1, MATE_SCORE + 1, nodes)
        board.pop()

        move_scores.append((move.uci(), score))
        if score > best_score:
            best_score = score
            best_move = move

    move_scores.sort(key=lambda x: x[1], reverse=True)
    return best_move, best_score, move_scores, nodes[0]


def score_to_value(score_cp: int, scale: float = 600.0) -> float:
    return max(-1.0, min(1.0, math.tanh(score_cp / scale)))


def scores_to_policy(move_scores: list[tuple[str, int]], temperature: float) -> dict[str, float]:
    if not move_scores:
        return {}

    best_score = move_scores[0][1]
    policy = {}
    for uci, score in move_scores:
        policy[uci] = math.exp((score - best_score) / temperature)

    total = sum(policy.values())
    if total > 0:
        for uci in policy:
            policy[uci] /= total

    return policy


def generate_targets(fen_file: Path, output: Path, depth: int,
                     temperature: float, value_scale: float,
                     teacher_name: str, teacher_version: int) -> int:
    fens = [line.strip() for line in fen_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")]

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(output, "w") as f:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)

            best_move, score_cp, move_scores, nodes = search_root(board, depth)
            policy = scores_to_policy(move_scores, temperature)
            value = score_to_value(score_cp, value_scale)

            legal_moves = [m.uci() for m in board.legal_moves]

            entry = {
                "fen": fen,
                "legal_moves": legal_moves,
                "root_move_scores_cp": {uci: s for uci, s in move_scores},
                "soft_policy": policy,
                "value": value,
                "depth": depth,
                "nodes": nodes,
                "teacher": teacher_name,
                "teacher_version": teacher_version,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

            if (i + 1) % 10 == 0:
                print(f"  processed {i + 1}/{len(fens)} positions...")

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher targets using PeSTO alpha-beta search")
    parser.add_argument("--fen-file", default=None, help="Input FEN file (one FEN per line)")
    parser.add_argument("--opening-book", default=None, help="Polyglot .bin opening book")
    parser.add_argument("--generate-openings", type=int, default=0,
                        help="Generate N diverse opening FENs (default: 0 = use fen-file)")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--depth", type=int, default=4, help="Search depth (default: 4)")
    parser.add_argument("--temperature", type=float, default=100.0,
                        help="Temperature in centipawns for soft policy (default: 100)")
    parser.add_argument("--value-scale", type=float, default=600.0,
                        help="Centipawn scale for tanh value conversion (default: 600)")
    parser.add_argument("--teacher-name", default="pesto_alphabeta",
                        help="Teacher identifier (default: pesto_alphabeta)")
    parser.add_argument("--teacher-version", type=int, default=1,
                        help="Teacher version (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for opening generation")

    args = parser.parse_args()

    output_path = Path(args.output)

    # Determine FEN source
    fens: list[str] = []

    if args.generate_openings > 0:
        # Generate diverse openings
        from tools.teacher.opening_book import generate_diverse_fens
        fen_lines = []
        if args.fen_file:
            fen_path = Path(args.fen_file)
            if fen_path.exists():
                fen_lines = fen_path.read_text().splitlines()
        fens = generate_diverse_fens(
            book_path=Path(args.opening_book) if args.opening_book else None,
            fen_lines=fen_lines,
            count=args.generate_openings,
            seed=args.seed,
        )
        # Write generated FENs to a temp file for reference
        temp_fen = output_path.with_suffix(".fens.txt")
        temp_fen.parent.mkdir(parents=True, exist_ok=True)
        temp_fen.write_text("\n".join(fens) + "\n")
        print(f"Generated {len(fens)} opening FENs -> {temp_fen}")
    elif args.fen_file:
        fen_path = Path(args.fen_file)
        if not fen_path.exists():
            print(f"ERROR: FEN file not found: {fen_path}")
            sys.exit(1)
        fens = [line.strip() for line in fen_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    else:
        print("ERROR: Provide --fen-file or --generate-openings")
        sys.exit(1)

    print(f"Generating teacher targets...")
    print(f"  FENs: {len(fens)}")
    print(f"  Output: {output_path}")
    print(f"  Depth: {args.depth}")
    print(f"  Temperature: {args.temperature} cp")
    print(f"  Value scale: {args.value_scale} cp")

    # Write FENs to temp file for generate_targets
    temp_fen_file = output_path.with_suffix(".input.fens")
    temp_fen_file.write_text("\n".join(fens) + "\n")

    start = time.time()
    count = generate_targets(
        temp_fen_file, output_path, args.depth, args.temperature,
        args.value_scale, args.teacher_name, args.teacher_version,
    )
    elapsed = time.time() - start

    print(f"  Generated {count} targets in {elapsed:.1f}s ({count/max(elapsed,0.01):.1f} positions/sec)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
