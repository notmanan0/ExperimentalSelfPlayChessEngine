from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from replay.index import init_db, index_replay_file  # noqa: E402


@dataclass(frozen=True)
class IndexFailure:
    path: str
    reason: str


@dataclass(frozen=True)
class IndexSummary:
    files_scanned: int
    files_indexed: int
    files_failed: int
    total_samples: int
    updates: int
    elapsed_seconds: float
    failures: tuple[IndexFailure, ...] = field(default_factory=tuple)


def _existing_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    with sqlite3.connect(index_path) as connection:
        try:
            rows = connection.execute("select path from chunks").fetchall()
        except sqlite3.DatabaseError:
            return set()
    return {str(row[0]) for row in rows}


def _sample_count(index_path: Path, chunk_path: Path) -> int:
    with sqlite3.connect(index_path) as connection:
        row = connection.execute(
            "select sample_count from chunks where path = ?",
            (str(chunk_path.resolve()),),
        ).fetchone()
    return int(row[0]) if row is not None else 0


def index_replay_dir(
    replay_dir: str | Path,
    index_path: str | Path,
    *,
    progress_interval: int = 100,
) -> IndexSummary:
    replay_dir = Path(replay_dir)
    index_path = Path(index_path)
    paths = sorted(replay_dir.rglob("*.cmrep"))
    index_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(index_path)
    existing = _existing_paths(index_path)

    indexed = 0
    updates = 0
    total_samples = 0
    failures: list[IndexFailure] = []
    started = time.monotonic()

    for scanned, path in enumerate(paths, start=1):
        if progress_interval > 0 and (scanned == 1 or scanned % progress_interval == 0):
            elapsed = time.monotonic() - started
            files_per_second = scanned / elapsed if elapsed > 0 else 0.0
            print(
                f"index progress: files_scanned={scanned}/{len(paths)} "
                f"files_indexed={indexed} files_failed={len(failures)} "
                f"elapsed={elapsed:.1f}s files/sec={files_per_second:.2f} "
                f"current_file={path}"
            )
        try:
            resolved = str(path.resolve())
            was_update = resolved in existing
            index_replay_file(index_path, path)
            indexed += 1
            updates += 1 if was_update else 0
            total_samples += _sample_count(index_path, path)
        except Exception as exc:  # keep scanning after corrupt chunks
            failure = IndexFailure(str(path), str(exc))
            failures.append(failure)
            print(f"index failure: path={failure.path} reason={failure.reason}")

    elapsed = time.monotonic() - started
    samples_per_second = total_samples / elapsed if elapsed > 0 else 0.0
    files_per_second = len(paths) / elapsed if elapsed > 0 else 0.0
    print(
        f"index summary: files_scanned={len(paths)} files_indexed={indexed} "
        f"files_failed={len(failures)} total_samples={total_samples} "
        f"updates={updates} elapsed={elapsed:.1f}s "
        f"files/sec={files_per_second:.2f} samples/sec={samples_per_second:.2f}"
    )
    if failures:
        print("index failures:")
        for failure in failures:
            print(f"- {failure.path}: {failure.reason}")

    return IndexSummary(
        files_scanned=len(paths),
        files_indexed=indexed,
        files_failed=len(failures),
        total_samples=total_samples,
        updates=updates,
        elapsed_seconds=elapsed,
        failures=tuple(failures),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index a directory of .cmrep chunks.")
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = index_replay_dir(args.replay_dir, args.index)
    return 1 if summary.files_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
