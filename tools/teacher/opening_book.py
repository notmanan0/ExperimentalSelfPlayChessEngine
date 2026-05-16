"""Opening book support for generating diverse teacher targets.

Supports:
1. Polyglot .bin opening books
2. Simple text files with move sequences (one per line)
3. Built-in common openings
"""

from __future__ import annotations

import random
from pathlib import Path

try:
    import chess
    import chess.polyglot
except ImportError:
    print("ERROR: python-chess is required. Install with: pip install python-chess")
    raise


# Built-in common openings in UCI move notation
BUILTIN_OPENINGS: list[list[str]] = [
    # Italian Game
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"],
    # Sicilian Najdorf
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6"],
    # Queen's Gambit
    ["d2d4", "d7d5", "c2c4"],
    # King's Indian
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    # French Defense
    ["e2e4", "e7e6", "d2d4", "d7d5"],
    # Caro-Kann
    ["e2e4", "c7c6", "d2d4", "d7d5"],
    # Ruy Lopez
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"],
    # Nimzo-Indian
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    # Slav Defense
    ["d2d4", "d7d5", "c2c4", "c7c6"],
    # English Opening
    ["c2c4"],
    # Reti Opening
    ["g1f3"],
    # Bird's Opening
    ["f2f4"],
    # London System
    ["d2d4", "g8f6", "c1f4"],
    # Catalan
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3"],
    # Grunfeld
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    # Benoni
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5"],
    # Dutch Defense
    ["d2d4", "f7f5"],
    # Scandinavian
    ["e2e4", "d7d5"],
    # Pirc Defense
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
    # Alekhine
    ["e2e4", "g8f6"],
]


def generate_opening_fens_from_book(
    book_path: Path,
    max_plies: int = 10,
    positions_per_opening: int = 1,
    seed: int = 42,
) -> list[str]:
    """Generate FENs from a polyglot opening book."""
    rng = random.Random(seed)
    fens: list[str] = []

    try:
        with chess.polyglot.open_reader(str(book_path)) as reader:
            for _ in range(positions_per_opening * 100):
                board = chess.Board()
                plies = rng.randint(2, max_plies)

                for ply in range(plies):
                    try:
                        entries = list(reader.find_all(board))
                        if not entries:
                            break
                        entry = rng.choice(entries)
                        board.push(entry.move)
                    except Exception:
                        break

                if board.is_valid() and not board.is_game_over():
                    fens.append(board.fen())

                if len(fens) >= positions_per_opening * 100:
                    break
    except Exception as e:
        print(f"Warning: Could not read polyglot book: {e}")
        return []

    return fens[:positions_per_opening * 100]


def generate_opening_fens_from_lines(
    lines: list[str],
    max_plies: int = 10,
    seed: int = 42,
) -> list[str]:
    """Generate FENs from move sequence lines."""
    rng = random.Random(seed)
    fens: list[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        moves = line.split()
        board = chess.Board()

        for move_uci in moves[:max_plies]:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    break
            except Exception:
                break

        if board.is_valid() and not board.is_game_over():
            fens.append(board.fen())

    return fens


def generate_builtin_opening_fens(
    max_plies: int = 10,
    seed: int = 42,
) -> list[str]:
    """Generate FENs from built-in common openings."""
    return generate_opening_fens_from_lines(
        [" ".join(moves) for moves in BUILTIN_OPENINGS],
        max_plies=max_plies,
        seed=seed,
    )


def generate_diverse_fens(
    book_path: Path | None = None,
    fen_lines: list[str] | None = None,
    max_plies: int = 10,
    count: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Generate a diverse set of FENs from multiple sources."""
    rng = random.Random(seed)
    all_fens: list[str] = []

    # From polyglot book
    if book_path and book_path.exists():
        book_fens = generate_opening_fens_from_book(
            book_path, max_plies=max_plies, positions_per_opening=count // 3, seed=seed
        )
        all_fens.extend(book_fens)

    # From text file
    if fen_lines:
        line_fens = generate_opening_fens_from_lines(fen_lines, max_plies=max_plies, seed=seed)
        all_fens.extend(line_fens)

    # From built-in openings
    builtin_fens = generate_builtin_opening_fens(max_plies=max_plies, seed=seed)
    all_fens.extend(builtin_fens)

    # Add starting position
    all_fens.append(chess.STARTING_FEN)

    # Deduplicate
    unique_fens = list(dict.fromkeys(all_fens))

    # Shuffle and limit
    rng.shuffle(unique_fens)
    return unique_fens[:count]


def generate_book_fens(
    book_path: Path | None = None,
    fen_file: Path | None = None,
    output: Path | None = None,
    max_plies: int = 10,
    count: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Generate FENs from available sources and optionally write to file."""
    fen_lines: list[str] = []
    if fen_file and fen_file.exists():
        fen_lines = fen_file.read_text().splitlines()

    fens = generate_diverse_fens(
        book_path=book_path,
        fen_lines=fen_lines,
        max_plies=max_plies,
        count=count,
        seed=seed,
    )

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(fens) + "\n")
        print(f"Generated {len(fens)} opening FENs -> {output}")

    return fens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate opening FENs for teacher targets")
    parser.add_argument("--book", type=str, default=None, help="Polyglot .bin opening book")
    parser.add_argument("--fen-file", type=str, default=None, help="Existing FEN file")
    parser.add_argument("--output", required=True, help="Output FEN file")
    parser.add_argument("--max-plies", type=int, default=10, help="Max plies per opening")
    parser.add_argument("--count", type=int, default=1000, help="Target FEN count")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_book_fens(
        book_path=Path(args.book) if args.book else None,
        fen_file=Path(args.fen_file) if args.fen_file else None,
        output=Path(args.output),
        max_plies=args.max_plies,
        count=args.count,
        seed=args.seed,
    )
