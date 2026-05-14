from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sqlite3
from typing import Any


def deduplicate_replay_index(db_path: Path) -> int:
    """Remove duplicate chunks from replay index by path. Returns removed count."""
    with sqlite3.connect(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.execute("""
            DELETE FROM chunks WHERE rowid NOT IN (
                SELECT MIN(rowid) FROM chunks GROUP BY path
            )
        """)
        after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return before - after


def split_dataset_by_game(
    db_path: Path,
    train_fraction: float = 0.9,
    seed: int = 1,
) -> tuple[list[str], list[str]]:
    """Split replay chunks into train/validation sets by game, not by position."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT path FROM chunks ORDER BY creation_timestamp_ms, path"
        ).fetchall()

    paths = [r[0] for r in rows]
    import random
    rng = random.Random(seed)
    rng.shuffle(paths)

    split_idx = max(1, int(len(paths) * train_fraction))
    return paths[:split_idx], paths[split_idx:]


def compute_chunk_fingerprint(path: Path) -> str:
    """Compute a content hash for a replay chunk file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            h.update(data)
    return h.hexdigest()[:16]


def detect_duplicate_positions(
    db_path: Path,
    sample_limit: int = 10000,
) -> dict[str, Any]:
    """Detect duplicate FEN positions across replay chunks."""
    from replay.reader import ReplayReader

    with sqlite3.connect(db_path) as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT path FROM chunks ORDER BY creation_timestamp_ms"
        ).fetchall()]

    seen: dict[str, int] = {}
    total = 0
    duplicates = 0

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            chunk = ReplayReader.read_file(path)
            for sample in chunk.samples:
                total += 1
                fen_key = f"{sample.board}|{sample.side_to_move}"
                if fen_key in seen:
                    duplicates += 1
                    seen[fen_key] += 1
                else:
                    seen[fen_key] = 1
                if total >= sample_limit:
                    break
        except Exception:
            continue
        if total >= sample_limit:
            break

    return {
        "total_checked": total,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / max(1, total),
        "unique_positions": len(seen),
    }


def compute_replay_statistics(db_path: Path) -> dict[str, Any]:
    """Compute aggregate statistics across all indexed replay chunks."""
    from replay.reader import ReplayReader

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("""
            SELECT path, sample_count, model_version, generator_version,
                   creation_timestamp_ms
            FROM chunks ORDER BY creation_timestamp_ms
        """).fetchall()

    total_samples = 0
    total_games = len(rows)
    model_versions: set[int] = set()
    generator_versions: set[int] = set()
    earliest_ts = float("inf")
    latest_ts = 0.0

    for row in rows:
        total_samples += row[1]
        model_versions.add(row[2])
        generator_versions.add(row[3])
        earliest_ts = min(earliest_ts, row[4])
        latest_ts = max(latest_ts, row[4])

    return {
        "total_games": total_games,
        "total_samples": total_samples,
        "model_versions": sorted(model_versions),
        "generator_versions": sorted(generator_versions),
        "earliest_timestamp_ms": earliest_ts if earliest_ts != float("inf") else 0,
        "latest_timestamp_ms": latest_ts,
    }


class RollingReplayBuffer:
    """Manages a rolling window of replay data with decay weighting."""

    def __init__(
        self,
        db_path: Path,
        max_chunks: int = 10000,
        decay_rate: float = 0.99,
        min_priority: float = 0.1,
    ) -> None:
        self.db_path = db_path
        self.max_chunks = max_chunks
        self.decay_rate = decay_rate
        self.min_priority = min_priority

    def prune_old_chunks(self) -> int:
        """Remove oldest chunks beyond max_chunks limit. Returns removed count."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count <= self.max_chunks:
                return 0

            to_remove = count - self.max_chunks
            old_paths = conn.execute("""
                SELECT path FROM chunks
                ORDER BY creation_timestamp_ms ASC
                LIMIT ?
            """, (to_remove,)).fetchall()

            for (path,) in old_paths:
                conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
                conn.execute(
                    "DELETE FROM chunk_priorities WHERE path = ?", (path,)
                )

            return len(old_paths)

    def update_priorities(self) -> int:
        """Decay sampling priorities for older chunks. Returns updated count."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO chunk_priorities (path, sampling_priority, updated_at_ms)
                SELECT path, 1.0, 0 FROM chunks
                WHERE path NOT IN (SELECT path FROM chunk_priorities)
            """)
            conn.execute("""
                UPDATE chunk_priorities
                SET sampling_priority = MAX(?, sampling_priority * ?)
            """, (self.min_priority, self.decay_rate))
            updated = conn.execute(
                "SELECT COUNT(*) FROM chunk_priorities WHERE sampling_priority > ?",
                (self.min_priority,)
            ).fetchone()[0]
        return updated

    def get_weighted_paths(self) -> list[tuple[str, float]]:
        """Return chunk paths weighted by sampling priority."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT c.path, COALESCE(p.sampling_priority, 1.0)
                FROM chunks c
                LEFT JOIN chunk_priorities p ON c.path = p.path
                ORDER BY c.creation_timestamp_ms
            """).fetchall()
        return [(r[0], r[1]) for r in rows]

    def maintain(self) -> dict[str, int]:
        """Run all maintenance operations. Returns counts."""
        pruned = self.prune_old_chunks()
        updated = self.update_priorities()
        return {
            "pruned": pruned,
            "priorities_updated": updated,
        }
