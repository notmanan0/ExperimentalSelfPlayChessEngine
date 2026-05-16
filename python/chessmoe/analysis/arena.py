from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
import argparse
import json
import math
import random
import time
from typing import Protocol


class PromotionDecision(str, Enum):
    PROMOTED = "promoted"
    REJECTED = "rejected"
    INSUFFICIENT_GAMES = "insufficient_games"
    SPRT_CONTINUE = "sprt_continue"


@dataclass(frozen=True)
class SprtConfig:
    enabled: bool = False
    elo0: float = 0.0
    elo1: float = 35.0
    alpha: float = 0.05
    beta: float = 0.05


@dataclass(frozen=True)
class ArenaConfig:
    candidate_model: str | Path
    best_model: str | Path
    openings: list[str]
    games_per_opening: int = 2
    search_budget: int = 64
    seed: int = 1
    minimum_games: int = 32
    minimum_score_rate: float = 0.55
    metadata_path: str | Path = "data/arena/latest_arena.json"
    registry_path: str | Path | None = None
    sprt: SprtConfig = SprtConfig()


@dataclass(frozen=True)
class ArenaGameSpec:
    game_id: int
    opening_fen: str
    candidate_color: str
    search_budget: int
    seed: int


@dataclass(frozen=True)
class ArenaGameResult:
    game_id: int
    opening_fen: str
    candidate_color: str
    result: str
    seed: int


@dataclass(frozen=True)
class ArenaSummary:
    wins: int
    losses: int
    draws: int
    games: int
    score_rate: float
    elo_estimate: float
    confidence_interval: tuple[float, float]


@dataclass(frozen=True)
class ArenaRunResult:
    config: ArenaConfig
    results: list[ArenaGameResult]
    summary: ArenaSummary
    decision: PromotionDecision
    candidate_path: str = ""
    best_path: str = ""
    openings: list[str] | None = None
    search_budget: int = 0
    forced: bool = False


class MatchBackend(Protocol):
    def play(
        self,
        game: ArenaGameSpec,
        candidate_model: Path,
        best_model: Path,
    ) -> ArenaGameResult:
        ...


class SeededScoreMatchBackend:
    """Deterministic backend used until model-backed arena games are wired."""

    def __init__(self, candidate_bias: float = 0.0) -> None:
        self.candidate_bias = candidate_bias

    def play(
        self,
        game: ArenaGameSpec,
        candidate_model: Path,
        best_model: Path,
    ) -> ArenaGameResult:
        del candidate_model, best_model
        if self.candidate_bias >= 1.0:
            result = "win"
        elif self.candidate_bias <= -1.0:
            result = "loss"
        else:
            roll = random.Random(game.seed).random() + self.candidate_bias
            if roll >= 0.62:
                result = "win"
            elif roll <= 0.38:
                result = "loss"
            else:
                result = "draw"
        return ArenaGameResult(
            game_id=game.game_id,
            opening_fen=game.opening_fen,
            candidate_color=game.candidate_color,
            result=result,
            seed=game.seed,
        )


class ExistingEvaluatorMatchBackend:
    """Boundary for future C++ evaluator-backed arena games."""

    def play(
        self,
        game: ArenaGameSpec,
        candidate_model: Path,
        best_model: Path,
    ) -> ArenaGameResult:
        raise NotImplementedError(
            "Phase 9 defines the C++/Python boundary; CUDA inference is not implemented"
        )


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def register_best(self, path: str | Path, model_id: str | None = None) -> None:
        data = self._read()
        data["best"] = _registry_entry(path, model_id, "best")
        self._write(data)

    def register_candidate(self, path: str | Path, model_id: str | None = None) -> None:
        data = self._read()
        entry = _registry_entry(path, model_id, "candidate")
        data.setdefault("candidate", [])
        _replace_or_append(data["candidate"], entry)
        self._write(data)

    def promote(self, candidate_path: str | Path, summary: ArenaSummary) -> None:
        data = self._read()
        entry = self._find_candidate(data, candidate_path)
        entry["status"] = "promoted"
        entry["promoted_at_ms"] = _now_ms()
        entry["summary"] = asdict(summary)
        data["best"] = {
            "model_id": entry["model_id"],
            "path": entry["path"],
            "status": "best",
            "updated_at_ms": _now_ms(),
        }
        data.setdefault("promoted", []).append(entry)
        data["candidate"] = [
            candidate
            for candidate in data.get("candidate", [])
            if candidate.get("path") != str(candidate_path)
        ]
        self._write(data)

    def reject(self, candidate_path: str | Path, summary: ArenaSummary) -> None:
        data = self._read()
        entry = self._find_candidate(data, candidate_path)
        entry["status"] = "rejected"
        entry["rejected_at_ms"] = _now_ms()
        entry["summary"] = asdict(summary)
        data.setdefault("rejected", []).append(entry)
        data["candidate"] = [
            candidate
            for candidate in data.get("candidate", [])
            if candidate.get("path") != str(candidate_path)
        ]
        self._write(data)

    def _find_candidate(self, data: dict, candidate_path: str | Path) -> dict:
        path = str(candidate_path)
        for candidate in data.get("candidate", []):
            if candidate.get("path") == path:
                return dict(candidate)
        return _registry_entry(path, None, "candidate")

    def _read(self) -> dict:
        if not self.path.exists():
            return {"best": None, "candidate": [], "rejected": [], "promoted": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def build_match_schedule(config: ArenaConfig) -> list[ArenaGameSpec]:
    if not config.openings:
        raise ValueError("arena requires at least one opening")
    if config.games_per_opening <= 0 or config.games_per_opening % 2 != 0:
        raise ValueError("games_per_opening must be positive and even")
    if config.search_budget <= 0:
        raise ValueError("search_budget must be positive")

    schedule: list[ArenaGameSpec] = []
    game_id = 0
    for opening_index, opening in enumerate(config.openings):
        for local_game in range(config.games_per_opening):
            schedule.append(
                ArenaGameSpec(
                    game_id=game_id,
                    opening_fen=opening,
                    candidate_color="white" if local_game % 2 == 0 else "black",
                    search_budget=config.search_budget,
                    seed=config.seed + opening_index * 1009 + local_game * 9173,
                )
            )
            game_id += 1
    return schedule


def run_arena(config: ArenaConfig, backend: MatchBackend | None = None) -> ArenaRunResult:
    backend = backend or SeededScoreMatchBackend()
    results = [
        backend.play(game, Path(config.candidate_model), Path(config.best_model))
        for game in build_match_schedule(config)
    ]
    summary = summarize_results(results)
    decision = evaluate_promotion(summary, config)
    run = ArenaRunResult(config=config, results=results, summary=summary, decision=decision)
    write_arena_metadata(run, config.metadata_path)
    if config.registry_path:
        registry = ModelRegistry(config.registry_path)
        if decision == PromotionDecision.PROMOTED:
            registry.promote(config.candidate_model, summary)
        elif decision == PromotionDecision.REJECTED:
            registry.reject(config.candidate_model, summary)
    return run


def summarize_results(results: list[ArenaGameResult]) -> ArenaSummary:
    wins = sum(result.result == "win" for result in results)
    losses = sum(result.result == "loss" for result in results)
    draws = sum(result.result == "draw" for result in results)
    games = len(results)
    score_rate = (wins + 0.5 * draws) / games if games else 0.0
    return ArenaSummary(
        wins=wins,
        losses=losses,
        draws=draws,
        games=games,
        score_rate=score_rate,
        elo_estimate=_elo_placeholder(score_rate),
        confidence_interval=_confidence_interval_placeholder(score_rate, games),
    )


def evaluate_promotion(summary: ArenaSummary, config: ArenaConfig) -> PromotionDecision:
    if summary.games < config.minimum_games:
        return PromotionDecision.INSUFFICIENT_GAMES
    if config.sprt.enabled:
        return PromotionDecision.SPRT_CONTINUE
    return (
        PromotionDecision.PROMOTED
        if summary.score_rate >= config.minimum_score_rate
        else PromotionDecision.REJECTED
    )


def write_arena_metadata(run: ArenaRunResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_ms": _now_ms(),
        "config": _config_to_dict(run.config),
        "summary": asdict(run.summary),
        "promotion": {"decision": run.decision.value},
        "results": [asdict(result) for result in run.results],
        "boundary": {
            "cpp": "cpp/selfplay supplies schedule and promotion primitives",
            "runner": "Python calls an arena backend; CUDA inference is out of scope",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_arena_config(path: str | Path) -> ArenaConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw.get("sprt"), dict):
        raw["sprt"] = SprtConfig(**raw["sprt"])
    return ArenaConfig(**raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run candidate-vs-best arena gating")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run_arena(load_arena_config(args.config))
    print(
        f"arena complete: games={result.summary.games}, "
        f"score={result.summary.score_rate:.3f}, decision={result.decision.value}"
    )
    return 0


def _registry_entry(path: str | Path, model_id: str | None, status: str) -> dict:
    return {
        "model_id": model_id or Path(path).stem,
        "path": str(path),
        "status": status,
        "created_at_ms": _now_ms(),
    }


def _replace_or_append(entries: list[dict], entry: dict) -> None:
    for index, existing in enumerate(entries):
        if existing.get("path") == entry["path"]:
            entries[index] = entry
            return
    entries.append(entry)


def _config_to_dict(config: ArenaConfig) -> dict:
    data = asdict(config)
    for key in ("candidate_model", "best_model", "metadata_path"):
        data[key] = str(data[key])
    if data["registry_path"] is not None:
        data["registry_path"] = str(data["registry_path"])
    return data


def _elo_placeholder(score_rate: float) -> float:
    clamped = min(0.999, max(0.001, score_rate))
    return 400.0 * math.log10(clamped / (1.0 - clamped))


def _confidence_interval_placeholder(score_rate: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return (0.0, 0.0)
    margin = 1.96 * math.sqrt(max(0.0, score_rate * (1.0 - score_rate)) / games)
    return (max(0.0, score_rate - margin), min(1.0, score_rate + margin))


def _now_ms() -> int:
    return int(time.time() * 1000)


if __name__ == "__main__":
    raise SystemExit(main())
