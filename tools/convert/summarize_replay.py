from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from replay.reader import ReplayReader  # noqa: E402


@dataclass(frozen=True)
class ReplaySummary:
    chunk_count: int
    total_samples: int
    min_samples_per_chunk: int
    max_samples_per_chunk: int
    mean_samples_per_chunk: float
    model_versions: tuple[int, ...]
    generator_versions: tuple[int, ...]
    total_size_bytes: int
    shortest_file: str | None
    longest_file: str | None
    short_game_warnings: tuple[str, ...] = field(default_factory=tuple)


def summarize_replay_dir(
    replay_dir: str | Path,
    *,
    short_sample_threshold: int = 4,
    progress_interval: int = 100,
) -> ReplaySummary:
    replay_dir = Path(replay_dir)
    paths = sorted(replay_dir.rglob("*.cmrep"))
    counts: list[int] = []
    model_versions: set[int] = set()
    generator_versions: set[int] = set()
    sizes: dict[Path, int] = {}
    warnings: list[str] = []
    started = time.monotonic()

    for index, path in enumerate(paths, start=1):
        if progress_interval > 0 and (index == 1 or index % progress_interval == 0):
            elapsed = time.monotonic() - started
            print(
                f"summary progress: files_scanned={index}/{len(paths)} "
                f"elapsed={elapsed:.1f}s current_file={path}"
            )
        chunk = ReplayReader.read_file(path)
        sample_count = chunk.header.sample_count
        counts.append(sample_count)
        model_versions.add(chunk.header.model_version)
        generator_versions.add(chunk.header.generator_version)
        sizes[path] = path.stat().st_size
        if sample_count < short_sample_threshold:
            warnings.append(f"{path}: only {sample_count} samples")

    if not counts:
        return ReplaySummary(0, 0, 0, 0, 0.0, (), (), 0, None, None, ())

    shortest = min(sizes, key=sizes.get)
    longest = max(sizes, key=sizes.get)
    return ReplaySummary(
        chunk_count=len(counts),
        total_samples=sum(counts),
        min_samples_per_chunk=min(counts),
        max_samples_per_chunk=max(counts),
        mean_samples_per_chunk=statistics.fmean(counts),
        model_versions=tuple(sorted(model_versions)),
        generator_versions=tuple(sorted(generator_versions)),
        total_size_bytes=sum(sizes.values()),
        shortest_file=str(shortest),
        longest_file=str(longest),
        short_game_warnings=tuple(warnings),
    )


def print_summary(summary: ReplaySummary) -> None:
    print(f"chunks: {summary.chunk_count}")
    print(f"total_samples: {summary.total_samples}")
    print(
        "samples_per_chunk: "
        f"min={summary.min_samples_per_chunk} "
        f"max={summary.max_samples_per_chunk} "
        f"mean={summary.mean_samples_per_chunk:.2f}"
    )
    print(f"model_versions: {list(summary.model_versions)}")
    print(f"generator_versions: {list(summary.generator_versions)}")
    print(f"total_size_bytes: {summary.total_size_bytes}")
    print(f"shortest_file: {summary.shortest_file}")
    print(f"longest_file: {summary.longest_file}")
    if summary.short_game_warnings:
        print("short_game_warnings:")
        for warning in summary.short_game_warnings:
            print(f"- {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize chessmoe replay chunks.")
    parser.add_argument("replay_dir", type=Path)
    parser.add_argument("--short-sample-threshold", type=int, default=4)
    args = parser.parse_args(argv)
    summary = summarize_replay_dir(
        args.replay_dir,
        short_sample_threshold=args.short_sample_threshold,
    )
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
