from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import time


@dataclass(frozen=True)
class StageResult:
    number: int
    name: str
    status: str
    elapsed_seconds: float
    command: tuple[str, ...] = ()


def run_stage(number: int, name: str, command: list[str]) -> StageResult:
    started = time.monotonic()
    print(f"stage {number} start: {name}")
    print("command:", " ".join(command))
    env = os.environ.copy()
    python_path = str(Path("python").resolve())
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else python_path + os.pathsep + env["PYTHONPATH"]
    )
    completed = subprocess.run(command, check=False, env=env)
    elapsed = time.monotonic() - started
    status = "passed" if completed.returncode == 0 else "failed"
    print(f"stage {number} {status}: {name} elapsed={elapsed:.1f}s")
    if completed.returncode != 0:
        raise RuntimeError(
            f"stage failed: {name}; command={' '.join(command)}; exit={completed.returncode}"
        )
    return StageResult(number, name, status, elapsed, tuple(command))


def promote_candidate(candidate: str | Path, version: int, *, weights_dir: str | Path = "weights") -> list[Path]:
    candidate = Path(candidate)
    weights_dir = Path(weights_dir)
    if not candidate.exists():
        raise FileNotFoundError(f"candidate does not exist: {candidate}")
    if version < 1:
        raise ValueError("promotion version must be >= 1")

    history = weights_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    suffix = candidate.suffix
    if suffix not in {".pt", ".onnx", ".engine"}:
        raise ValueError("candidate must be a .pt, .onnx, or .engine artifact")

    best_path = weights_dir / f"best{suffix}"
    copied: list[Path] = []
    if best_path.exists():
        archive = history / f"model_{version - 1:06d}{suffix}"
        if archive.exists():
            raise FileExistsError(
                f"refusing to overwrite existing history artifact: {archive}"
            )
        shutil.copy2(best_path, archive)
        copied.append(archive)
        print(f"previous best archived: {archive}")

    versioned = history / f"model_{version:06d}{suffix}"
    if versioned.exists():
        raise FileExistsError(f"refusing to overwrite candidate history artifact: {versioned}")
    shutil.copy2(candidate, versioned)
    shutil.copy2(candidate, best_path)
    copied.extend([versioned, best_path])
    print(f"candidate promoted: candidate={candidate} version={version}")
    print(f"new best: {best_path}")
    return copied


def _selfplay_exe() -> str:
    return str(Path("build-nmake") / "bin" / "selfplay.exe")


def bootstrap_commands() -> list[tuple[str, list[str]]]:
    return [
        ("material self-play", [_selfplay_exe(), "--config", "configs/selfplay/bootstrap_material.json"]),
        (
            "index replay",
            [
                sys.executable,
                "tools/convert/index_replay_dir.py",
                "data/replay/phase13",
                "--index",
                "data/replay/replay.sqlite",
            ],
        ),
        ("summarize replay", [sys.executable, "tools/convert/summarize_replay.py", "data/replay/phase13"]),
        ("train bootstrap", [sys.executable, "-m", "chessmoe.training.train", "--config", "configs/training/dense_bootstrap.json"]),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run user-facing chessmoe generation pipeline stages.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    generation = sub.add_parser("generation")
    generation.add_argument("--phase", type=int, required=True)
    generation.add_argument("--engine", type=Path, required=True)
    arena = sub.add_parser("arena")
    arena.add_argument("--candidate", required=True)
    arena.add_argument("--baseline", required=True)
    arena.add_argument("--games", type=int, required=True)
    arena.add_argument("--visits", type=int, required=True)
    promote = sub.add_parser("promote")
    promote.add_argument("--candidate", required=True)
    promote.add_argument("--version", type=int, required=True)
    args = parser.parse_args(argv)

    total_started = time.monotonic()
    results: list[StageResult] = []
    try:
        if args.command == "bootstrap":
            for index, (name, command) in enumerate(bootstrap_commands(), start=1):
                results.append(run_stage(index, name, command))
        elif args.command == "generation":
            command = [
                _selfplay_exe(),
                "--config",
                "configs/selfplay/neural_tensorrt.json",
                "--engine",
                str(args.engine),
                "--output-dir",
                f"data/replay/phase{args.phase}",
            ]
            results.append(run_stage(1, "neural self-play", command))
        elif args.command == "arena":
            print(
                "arena refusal: existing arena command is placeholder-seeded and cannot "
                "produce meaningful promotion decisions"
            )
            return 2
        elif args.command == "promote":
            started = time.monotonic()
            copied = promote_candidate(args.candidate, args.version)
            elapsed = time.monotonic() - started
            print("promotion files:")
            for path in copied:
                print(f"- {path}")
            results.append(StageResult(1, "promote", "passed", elapsed))
    except Exception as exc:
        print(f"pipeline failed: {exc}")
        return 1
    finally:
        total_elapsed = time.monotonic() - total_started
        if results:
            print("pipeline summary:")
            for result in results:
                print(f"- stage {result.number} {result.name}: {result.status}")
            print(f"total elapsed: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
