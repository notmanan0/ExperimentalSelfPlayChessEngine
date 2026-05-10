from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Protocol

import torch

from chessmoe.models.encoding import NUM_MOVE_BUCKETS, move_to_index
from chessmoe.models.tiny_model import scalar_value_from_wdl
from chessmoe.training.data import encode_replay_sample
from replay.index import SCHEMA_SQL
from replay.reader import ReplayReader, ReplaySample


class ReanalysisTargetPolicy(StrEnum):
    ORIGINAL = "original"
    LATEST_REANALYSIS = "latest_reanalysis"
    MIX = "mix"


@dataclass(frozen=True)
class ReanalysisConfig:
    replay_index: str | Path
    output_index: str | Path
    current_model_version: int
    search_budget: int
    reanalysis_timestamp_ms: int | None = None
    source_model_versions: frozenset[int] | None = None
    older_than_timestamp_ms: int | None = None
    minimum_sampling_priority: float = 0.0
    max_chunks: int | None = None
    max_samples: int | None = None


@dataclass(frozen=True)
class ReanalysisTarget:
    chunk_path: Path
    game_id: int
    ply_index: int
    source_model_version: int
    model_version: int
    search_budget: int
    reanalysis_timestamp_ms: int
    root_value: float
    policy: list[dict[str, float | int | str]]


@dataclass(frozen=True)
class ReanalysisSummary:
    chunks_selected: int
    samples_reanalyzed: int
    targets_written: int
    elapsed_ms: float

    @property
    def positions_per_second(self) -> float:
        elapsed_seconds = self.elapsed_ms / 1000.0
        return 0.0 if elapsed_seconds <= 0.0 else self.samples_reanalyzed / elapsed_seconds


class PositionAnalyzer(Protocol):
    def analyze(
        self,
        sample: ReplaySample,
        *,
        search_budget: int,
    ) -> tuple[list[dict[str, float | int | str]], float]:
        ...


class SimpleModelAnalyzer:
    """Deterministic root-only analyzer used for tests and CPU smoke runs."""

    def __init__(
        self,
        *,
        policy_bias: dict[str, float] | None = None,
        root_value: float = 0.0,
    ) -> None:
        self.policy_bias = policy_bias or {}
        self.root_value = root_value

    def analyze(
        self,
        sample: ReplaySample,
        *,
        search_budget: int,
    ) -> tuple[list[dict[str, float | int | str]], float]:
        logits = [float(self.policy_bias.get(move, 0.0)) for move in sample.legal_moves]
        probabilities = _softmax(logits)
        visits = _allocate_visits(probabilities, search_budget)
        policy = [
            {
                "move": move,
                "visit_count": visits[index],
                "probability": probabilities[index],
            }
            for index, move in enumerate(sample.legal_moves)
        ]
        policy.sort(key=lambda entry: (-float(entry["probability"]), str(entry["move"])))
        return policy, max(-1.0, min(1.0, self.root_value))


class TorchModelAnalyzer:
    """Root evaluator for current PyTorch models.

    This is intentionally root-only in Phase 14. The C++ self-play/search path owns
    full MCTS; Python reanalysis stores versioned target overlays without changing
    the original replay chunks.
    """

    def __init__(self, model: torch.nn.Module, *, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def analyze(
        self,
        sample: ReplaySample,
        *,
        search_budget: int,
    ) -> tuple[list[dict[str, float | int | str]], float]:
        features = encode_replay_sample(sample).unsqueeze(0).to(self.device)
        output = self.model(features)
        policy_logits = output.policy_logits[0].detach().cpu()
        legal_logits = torch.tensor(
            [float(policy_logits[move_to_index(move)]) for move in sample.legal_moves],
            dtype=torch.float32,
        )
        probabilities_tensor = torch.softmax(legal_logits, dim=0)
        probabilities = [float(value) for value in probabilities_tensor.tolist()]
        visits = _allocate_visits(probabilities, search_budget)
        root_value = float(scalar_value_from_wdl(output.wdl_logits)[0].detach().cpu())
        policy = [
            {
                "move": move,
                "visit_count": visits[index],
                "probability": probabilities[index],
            }
            for index, move in enumerate(sample.legal_moves)
        ]
        policy.sort(key=lambda entry: (-float(entry["probability"]), str(entry["move"])))
        return policy, max(-1.0, min(1.0, root_value))


def select_replay_chunks(
    db_path: str | Path,
    *,
    model_versions: set[int] | frozenset[int] | None = None,
    older_than_timestamp_ms: int | None = None,
    minimum_sampling_priority: float = 0.0,
    max_chunks: int | None = None,
) -> list[Path]:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        where = ["coalesce(p.sampling_priority, 1.0) >= ?"]
        params: list[object] = [minimum_sampling_priority]
        if model_versions:
            placeholders = ",".join("?" for _ in model_versions)
            where.append(f"c.model_version in ({placeholders})")
            params.extend(sorted(model_versions))
        if older_than_timestamp_ms is not None:
            where.append("c.creation_timestamp_ms < ?")
            params.append(older_than_timestamp_ms)

        sql = f"""
          select c.path
          from chunks c
          left join chunk_priorities p on p.path = c.path
          where {' and '.join(where)}
          order by coalesce(p.sampling_priority, 1.0) desc,
                   c.creation_timestamp_ms asc,
                   c.path asc
        """
        if max_chunks is not None:
            sql += " limit ?"
            params.append(max_chunks)
        rows = connection.execute(sql, params).fetchall()
    return [Path(row[0]) for row in rows]


def reanalyze_index(
    config: ReanalysisConfig,
    *,
    analyzer: PositionAnalyzer,
) -> ReanalysisSummary:
    _validate_config(config)
    started = time.perf_counter()
    timestamp_ms = config.reanalysis_timestamp_ms or int(time.time() * 1000)
    chunks = select_replay_chunks(
        config.replay_index,
        model_versions=config.source_model_versions,
        older_than_timestamp_ms=config.older_than_timestamp_ms,
        minimum_sampling_priority=config.minimum_sampling_priority,
        max_chunks=config.max_chunks,
    )

    samples_reanalyzed = 0
    targets_written = 0
    for chunk_path in chunks:
        chunk = ReplayReader.read_file(chunk_path)
        for sample in chunk.samples:
            if config.max_samples is not None and samples_reanalyzed >= config.max_samples:
                break
            policy, root_value = analyzer.analyze(sample, search_budget=config.search_budget)
            write_reanalysis_target(
                config.output_index,
                ReanalysisTarget(
                    chunk_path=chunk_path.resolve(),
                    game_id=sample.game_id,
                    ply_index=sample.ply_index,
                    source_model_version=chunk.header.model_version,
                    model_version=config.current_model_version,
                    search_budget=config.search_budget,
                    reanalysis_timestamp_ms=timestamp_ms,
                    root_value=root_value,
                    policy=policy,
                ),
            )
            samples_reanalyzed += 1
            targets_written += 1
        if config.max_samples is not None and samples_reanalyzed >= config.max_samples:
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return ReanalysisSummary(
        chunks_selected=len(chunks),
        samples_reanalyzed=samples_reanalyzed,
        targets_written=targets_written,
        elapsed_ms=elapsed_ms,
    )


def write_reanalysis_target(db_path: str | Path, target: ReanalysisTarget) -> None:
    _validate_policy(target.policy)
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute(
            """
            insert into reanalysis_targets (
              chunk_path,
              game_id,
              ply_index,
              source_model_version,
              model_version,
              search_budget,
              reanalysis_timestamp_ms,
              root_value,
              policy_json,
              created_at_ms
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(chunk_path, game_id, ply_index, model_version, search_budget, reanalysis_timestamp_ms)
            do update set
              source_model_version=excluded.source_model_version,
              root_value=excluded.root_value,
              policy_json=excluded.policy_json,
              created_at_ms=excluded.created_at_ms
            """,
            (
                str(target.chunk_path.resolve()),
                target.game_id,
                target.ply_index,
                target.source_model_version,
                target.model_version,
                target.search_budget,
                target.reanalysis_timestamp_ms,
                target.root_value,
                json.dumps(target.policy, sort_keys=True),
                int(time.time() * 1000),
            ),
        )


def load_latest_reanalysis_target(
    db_path: str | Path,
    chunk_path: str | Path,
    *,
    game_id: int,
    ply_index: int,
    model_version: int | None = None,
) -> ReanalysisTarget | None:
    with sqlite3.connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)
        where = ["chunk_path = ?", "game_id = ?", "ply_index = ?"]
        params: list[object] = [str(Path(chunk_path).resolve()), game_id, ply_index]
        if model_version is not None:
            where.append("model_version = ?")
            params.append(model_version)
        row = connection.execute(
            f"""
            select chunk_path, game_id, ply_index, source_model_version,
                   model_version, search_budget, reanalysis_timestamp_ms,
                   root_value, policy_json
            from reanalysis_targets
            where {' and '.join(where)}
            order by reanalysis_timestamp_ms desc, model_version desc, id desc
            limit 1
            """,
            params,
        ).fetchone()
    if row is None:
        return None
    return ReanalysisTarget(
        chunk_path=Path(row[0]),
        game_id=row[1],
        ply_index=row[2],
        source_model_version=row[3],
        model_version=row[4],
        search_budget=row[5],
        reanalysis_timestamp_ms=row[6],
        root_value=row[7],
        policy=json.loads(row[8]),
    )


def reanalysis_policy_target(target: ReanalysisTarget) -> torch.Tensor:
    tensor = torch.zeros(NUM_MOVE_BUCKETS, dtype=torch.float32)
    total_probability = sum(max(0.0, float(entry["probability"])) for entry in target.policy)
    if total_probability <= 0.0:
        total_visits = sum(max(0, int(entry["visit_count"])) for entry in target.policy)
        if total_visits <= 0:
            raise ValueError("reanalysis target has no positive policy mass")
        for entry in target.policy:
            tensor[move_to_index(str(entry["move"]))] = max(0, int(entry["visit_count"])) / total_visits
        return tensor
    for entry in target.policy:
        tensor[move_to_index(str(entry["move"]))] = (
            max(0.0, float(entry["probability"])) / total_probability
        )
    return tensor


def reanalysis_wdl_target(target: ReanalysisTarget) -> int:
    if target.root_value > 0.05:
        return 0
    if target.root_value < -0.05:
        return 2
    return 1


def sample_to_fen(sample: ReplaySample) -> str:
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        empty = 0
        text = ""
        for file_index in range(8):
            piece = sample.board[rank * 8 + file_index]
            if piece is None:
                empty += 1
                continue
            if empty:
                text += str(empty)
                empty = 0
            text += piece
        if empty:
            text += str(empty)
        ranks.append(text)
    side = "w" if sample.side_to_move == "white" else "b"
    castling = "".join(
        flag
        for bit, flag in ((1, "K"), (2, "Q"), (4, "k"), (8, "q"))
        if sample.castling_rights & bit
    )
    if not castling:
        castling = "-"
    ep = sample.en_passant_square or "-"
    return f"{'/'.join(ranks)} {side} {castling} {ep} {sample.halfmove_clock} {sample.fullmove_number}"


def _validate_config(config: ReanalysisConfig) -> None:
    if config.current_model_version <= 0:
        raise ValueError("current_model_version must be positive")
    if config.search_budget <= 0:
        raise ValueError("search_budget must be positive")
    if config.minimum_sampling_priority < 0.0:
        raise ValueError("minimum_sampling_priority cannot be negative")


def _validate_policy(policy: list[dict[str, float | int | str]]) -> None:
    if not policy:
        raise ValueError("reanalysis target policy cannot be empty")
    total = 0.0
    for entry in policy:
        if "move" not in entry or "visit_count" not in entry or "probability" not in entry:
            raise ValueError("policy entries require move, visit_count, and probability")
        total += max(0.0, float(entry["probability"]))
    if total <= 0.0:
        raise ValueError("reanalysis target policy needs positive probability mass")


def _softmax(logits: list[float]) -> list[float]:
    if not logits:
        return []
    values = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=0)
    return [float(value) for value in values.tolist()]


def _allocate_visits(probabilities: list[float], search_budget: int) -> list[int]:
    raw = [probability * search_budget for probability in probabilities]
    visits = [int(value) for value in raw]
    remaining = search_budget - sum(visits)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - visits[index], probabilities[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        visits[index] += 1
    return visits


def main() -> int:
    parser = argparse.ArgumentParser(description="Reanalyze replay chunks into versioned target records")
    parser.add_argument("--replay-index", required=True, type=Path)
    parser.add_argument("--output-index", type=Path)
    parser.add_argument("--current-model-version", required=True, type=int)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--source-model-version", action="append", type=int)
    parser.add_argument("--older-than-timestamp-ms", type=int)
    parser.add_argument("--minimum-sampling-priority", type=float, default=0.0)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--timestamp-ms", type=int)
    args = parser.parse_args()

    summary = reanalyze_index(
        ReanalysisConfig(
            replay_index=args.replay_index,
            output_index=args.output_index or args.replay_index,
            current_model_version=args.current_model_version,
            search_budget=args.search_budget,
            reanalysis_timestamp_ms=args.timestamp_ms,
            source_model_versions=(
                frozenset(args.source_model_version)
                if args.source_model_version
                else None
            ),
            older_than_timestamp_ms=args.older_than_timestamp_ms,
            minimum_sampling_priority=args.minimum_sampling_priority,
            max_chunks=args.max_chunks,
            max_samples=args.max_samples,
        ),
        analyzer=SimpleModelAnalyzer(),
    )
    print(
        json.dumps(
            {
                "chunks_selected": summary.chunks_selected,
                "samples_reanalyzed": summary.samples_reanalyzed,
                "targets_written": summary.targets_written,
                "elapsed_ms": summary.elapsed_ms,
                "positions_per_second": summary.positions_per_second,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
