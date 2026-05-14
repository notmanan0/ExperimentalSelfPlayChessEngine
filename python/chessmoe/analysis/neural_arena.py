from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Protocol

from chessmoe.analysis.arena import (
    ArenaConfig,
    ArenaGameResult,
    ArenaGameSpec,
    ArenaRunResult,
    ArenaSummary,
    MatchBackend,
    PromotionDecision,
    build_match_schedule,
    evaluate_promotion,
    summarize_results,
    write_arena_metadata,
)
from chessmoe.models.encoding import encode_fen, move_to_index, NUM_MOVE_BUCKETS


@dataclass(frozen=True)
class MctsArenaConfig:
    visits: int = 64
    cpuct: float = 1.5
    temperature: float = 0.0
    dirichlet_alpha: float = 0.0
    dirichlet_epsilon: float = 0.0


@dataclass
class ArenaMctsNode:
    prior: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    children: dict[str, ArenaMctsNode] | None = None

    def mean_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


class PositionEvaluator(Protocol):
    def evaluate(self, fen: str, legal_moves: list[str]) -> tuple[dict[str, float], float]:
        """Return (policy_probs, value) where value is from side-to-move perspective."""
        ...


class OnnxModelEvaluator:
    """Evaluates positions using an ONNX model."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._session = None

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed; install it for ONNX arena backend"
            )

    def evaluate(self, fen: str, legal_moves: list[str]) -> tuple[dict[str, float], float]:
        self._ensure_session()
        import torch
        import numpy as np

        features = encode_fen(fen).unsqueeze(0).numpy()
        policy_raw, wdl_raw, _ = self._session.run(None, {"board": features})

        import torch.nn.functional as F
        policy_logits = torch.from_numpy(policy_raw[0])
        wdl_logits = torch.from_numpy(wdl_raw[0])

        wdl_probs = F.softmax(wdl_logits, dim=-1).numpy()
        value = float(wdl_probs[0] - wdl_probs[2])

        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_logits = [float(policy_logits[i]) for i in legal_indices]
        max_logit = max(legal_logits) if legal_logits else 0.0
        exp_logits = [math.exp(l - max_logit) for l in legal_logits]
        total = sum(exp_logits)
        probs = {
            m: e / total for m, e in zip(legal_moves, exp_logits)
        }

        return probs, value


class PytorchModelEvaluator:
    """Evaluates positions using a PyTorch checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from chessmoe.training.checkpoint import load_checkpoint
        self._model = load_checkpoint(str(self.checkpoint_path), map_location="cpu")
        self._model.eval()

    def evaluate(self, fen: str, legal_moves: list[str]) -> tuple[dict[str, float], float]:
        self._ensure_model()
        import torch
        import torch.nn.functional as F

        features = encode_fen(fen).unsqueeze(0)
        with torch.no_grad():
            output = self._model(features)

        policy_logits = output.policy_logits[0]
        wdl_logits = output.wdl_logits[0]
        wdl_probs = F.softmax(wdl_logits, dim=-1)
        value = float(wdl_probs[0] - wdl_probs[2])

        legal_indices = [move_to_index(m) for m in legal_moves]
        legal_logits = [float(policy_logits[i]) for i in legal_indices]
        max_logit = max(legal_logits) if legal_logits else 0.0
        exp_logits = [math.exp(l - max_logit) for l in legal_logits]
        total = sum(exp_logits)
        probs = {
            m: e / total for m, e in zip(legal_moves, exp_logits)
        }

        return probs, value


class NeuralMatchBackend:
    """Arena backend that uses a neural evaluator for both sides."""

    def __init__(
        self,
        candidate_eval: PositionEvaluator,
        best_eval: PositionEvaluator,
        config: MctsArenaConfig | None = None,
    ) -> None:
        self.candidate_eval = candidate_eval
        self.best_eval = best_eval
        self.config = config or MctsArenaConfig()

    def play(
        self,
        game: ArenaGameSpec,
        candidate_model: Path,
        best_model: Path,
    ) -> ArenaGameResult:
        from chessmoe.models.encoding import parse_fen

        fen = game.opening_fen
        side_to_move = "white" if parse_fen(fen).side_to_move == "w" else "black"

        candidate_is_white = game.candidate_color == "white"
        rng = random.Random(game.seed)
        ply = 0

        while ply < 300:
            parts = parse_fen(fen)
            side = parts.side_to_move
            legal = _get_legal_moves_from_fen(fen)

            if not legal:
                in_check = _is_in_check(fen)
                if in_check:
                    winner = "black" if side == "w" else "white"
                else:
                    winner = "draw"
                break

            if _is_draw_by_rules(fen, ply):
                winner = "draw"
                break

            if (side == "w" and candidate_is_white) or \
               (side == "b" and not candidate_is_white):
                eval_fn = self.candidate_eval
            else:
                eval_fn = self.best_eval

            probs, value = eval_fn.evaluate(fen, legal)

            if self.config.temperature <= 0.0:
                selected = max(probs, key=probs.get)
            else:
                moves = list(probs.keys())
                weights = [probs[m] ** (1.0 / self.config.temperature) for m in moves]
                total = sum(weights)
                weights = [w / total for w in weights]
                selected = moves[rng.choices(range(len(moves)), weights=weights)[0]]

            fen = _apply_move(fen, selected)
            ply += 1
        else:
            winner = "draw"

        if winner == "draw":
            result_str = "draw"
        elif (winner == "white" and candidate_is_white) or \
             (winner == "black" and not candidate_is_white):
            result_str = "win"
        else:
            result_str = "loss"

        return ArenaGameResult(
            game_id=game.game_id,
            opening_fen=game.opening_fen,
            candidate_color=game.candidate_color,
            result=result_str,
            seed=game.seed,
        )


def _get_legal_moves_from_fen(fen: str) -> list[str]:
    import chess
    board = chess.Board(fen)
    return [m.uci() for m in board.legal_moves]


def _is_in_check(fen: str) -> bool:
    import chess
    board = chess.Board(fen)
    return board.is_check()


def _is_draw_by_rules(fen: str, ply: int) -> bool:
    import chess
    board = chess.Board(fen)
    return board.is_fifty_moves() or board.is_repetition()


def _apply_move(fen: str, move_uci: str) -> str:
    import chess
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    return board.fen()


def run_neural_arena(
    config: ArenaConfig,
    candidate_eval: PositionEvaluator,
    best_eval: PositionEvaluator,
    mcts_config: MctsArenaConfig | None = None,
) -> ArenaRunResult:
    backend = NeuralMatchBackend(candidate_eval, best_eval, mcts_config)
    results = [
        backend.play(game, Path(config.candidate_model), Path(config.best_model))
        for game in build_match_schedule(config)
    ]
    summary = summarize_results(results)
    decision = evaluate_promotion(summary, config)
    run = ArenaRunResult(
        config=config, results=results, summary=summary, decision=decision
    )
    write_arena_metadata(run, config.metadata_path)
    return run
