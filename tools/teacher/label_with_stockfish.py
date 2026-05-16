"""Optional Stockfish distillation hook.

Labels positions using an external Stockfish binary via UCI protocol.
Produces the same JSONL format as the internal alpha-beta teacher.

Usage:
    python tools/teacher/label_with_stockfish.py \
        --fen-file data/openings/bootstrap_fens.txt \
        --stockfish-path /path/to/stockfish \
        --output data/teacher/sf_targets.jsonl \
        --depth 12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    import chess
    import chess.engine
except ImportError:
    print("ERROR: python-chess is required. Install with: pip install python-chess")
    sys.exit(1)


def label_position(engine: chess.engine.SimpleEngine, board: chess.Board,
                   depth: int, multipv: int) -> dict:
    results = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)

    legal_moves = [m.uci() for m in board.legal_moves]
    move_scores: dict[str, float] = {}

    for info in results:
        if "pv" not in info or not info["pv"]:
            continue
        move = info["pv"][0]
        score = info["score"].pov(board.turn)
        cp = score.score(mate_score=30000)
        if cp is not None:
            move_scores[move.uci()] = cp

    for m in legal_moves:
        if m not in move_scores:
            move_scores[m] = 0

    best_score = max(move_scores.values()) if move_scores else 0
    import math
    policy = {}
    for uci, cp in move_scores.items():
        policy[uci] = math.exp((cp - best_score) / 100.0)
    total = sum(policy.values())
    if total > 0:
        for uci in policy:
            policy[uci] /= total

    best_cp = best_score if board.turn == chess.WHITE else -best_score
    value = max(-1.0, min(1.0, math.tanh(best_cp / 600.0)))

    return {
        "fen": board.fen(),
        "legal_moves": legal_moves,
        "root_move_scores_cp": move_scores,
        "soft_policy": policy,
        "value": value,
        "depth": depth,
        "nodes": 0,
        "teacher": "stockfish",
        "teacher_version": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label positions with Stockfish")
    parser.add_argument("--fen-file", required=True, help="Input FEN file")
    parser.add_argument("--stockfish-path", required=True, help="Path to Stockfish binary")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--depth", type=int, default=12, help="Search depth (default: 12)")
    parser.add_argument("--multipv", type=int, default=1, help="MultiPV lines (default: 1)")

    args = parser.parse_args()

    sf_path = Path(args.stockfish_path)
    if not sf_path.exists():
        print(f"ERROR: Stockfish binary not found: {sf_path}")
        print("Provide the path to a Stockfish executable via --stockfish-path")
        sys.exit(1)

    fen_path = Path(args.fen_file)
    output_path = Path(args.output)

    if not fen_path.exists():
        print(f"ERROR: FEN file not found: {fen_path}")
        sys.exit(1)

    fens = [line.strip() for line in fen_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Labeling with Stockfish...")
    print(f"  FEN file: {fen_path}")
    print(f"  Stockfish: {sf_path}")
    print(f"  Output: {output_path}")
    print(f"  Depth: {args.depth}")

    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))

    start = time.time()
    count = 0

    with open(output_path, "w") as f:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            entry = label_position(engine, board, args.depth, args.multipv)
            f.write(json.dumps(entry) + "\n")
            count += 1

            if (i + 1) % 10 == 0:
                print(f"  processed {i + 1}/{len(fens)} positions...")

    engine.quit()
    elapsed = time.time() - start

    print(f"  Labeled {count} positions in {elapsed:.1f}s")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
