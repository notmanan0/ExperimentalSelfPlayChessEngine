from __future__ import annotations

import random
import json
from pathlib import Path

from chessmoe.analysis.neural_arena import (
    MctsArenaConfig,
    NeuralMatchBackend,
    OnnxModelEvaluator,
    PytorchModelEvaluator,
    _get_legal_moves_from_fen,
    _apply_move,
)


STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_get_legal_moves_starting():
    moves = _get_legal_moves_from_fen(STARTING_FEN)
    assert len(moves) == 20
    assert "e2e4" in moves
    assert "d2d4" in moves


def test_apply_move_e4():
    fen = _apply_move(STARTING_FEN, "e2e4")
    assert "4P3" in fen
    assert fen.startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR")


def test_apply_move_invalid():
    import pytest
    with pytest.raises(Exception):
        _apply_move(STARTING_FEN, "e2e9")


class MockEvaluator:
    def __init__(self, bias_move: str | None = None):
        self.bias_move = bias_move

    def evaluate(self, fen, legal_moves):
        probs = {}
        for m in legal_moves:
            probs[m] = 1.0 / len(legal_moves)
        if self.bias_move and self.bias_move in probs:
            probs[self.bias_move] = 0.9
            remaining = 0.1 / max(1, len(probs) - 1)
            for m in probs:
                if m != self.bias_move:
                    probs[m] = remaining
        total = sum(probs.values())
        probs = {m: v / total for m, v in probs.items()}
        return probs, 0.5


def test_neural_backend_deterministic():
    candidate = MockEvaluator("e2e4")
    best = MockEvaluator("d2d4")
    backend = NeuralMatchBackend(candidate, best, MctsArenaConfig(visits=1))

    from chessmoe.analysis.arena import ArenaGameSpec
    game = ArenaGameSpec(
        game_id=0, opening_fen=STARTING_FEN,
        candidate_color="white", search_budget=1, seed=42,
    )
    result1 = backend.play(game, Path("c.pt"), Path("b.pt"))
    result2 = backend.play(game, Path("c.pt"), Path("b.pt"))
    assert result1.result == result2.result


def test_neural_backend_returns_valid_result():
    candidate = MockEvaluator()
    best = MockEvaluator()
    backend = NeuralMatchBackend(candidate, best, MctsArenaConfig(visits=1))

    from chessmoe.analysis.arena import ArenaGameSpec
    game = ArenaGameSpec(
        game_id=0, opening_fen=STARTING_FEN,
        candidate_color="white", search_budget=1, seed=1,
    )
    result = backend.play(game, Path("c.pt"), Path("b.pt"))
    assert result.result in ("win", "loss", "draw")
    assert result.game_id == 0
    assert result.candidate_color == "white"


class PolicyTrapEvaluator:
    def __init__(self, root_fen: str, policy_trap: str, value_best: str):
        self.root_fen = root_fen
        self.policy_trap = policy_trap
        self.value_best = value_best

    def evaluate(self, fen, legal_moves):
        import chess

        probs = {m: 1.0 for m in legal_moves}
        if fen == self.root_fen and self.policy_trap in probs:
            probs[self.policy_trap] = 5.0
        total = sum(probs.values())
        probs = {m: v / total for m, v in probs.items()}

        board = chess.Board(fen)
        if board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE):
            value = -0.95 if board.turn == chess.BLACK else 0.95
        elif board.piece_at(chess.D4) == chess.Piece(chess.PAWN, chess.WHITE):
            value = 0.95 if board.turn == chess.BLACK else -0.95
        else:
            value = 0.0
        return probs, value


def test_neural_backend_search_uses_root_value_not_policy_only():
    evaluator = PolicyTrapEvaluator(
        STARTING_FEN,
        policy_trap="d2d4",
        value_best="e2e4",
    )
    backend = NeuralMatchBackend(
        evaluator,
        evaluator,
        MctsArenaConfig(visits=160, cpuct=1.5, temperature=0.0),
    )

    selected = backend.select_move(
        STARTING_FEN,
        evaluator,
        rng=random.Random(7),
    )

    assert selected == "e2e4"
