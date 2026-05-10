from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from replay.index import index_replay_file  # noqa: E402
from replay.reader import ReplayReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate chessmoe replay chunks")
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--index", type=Path, help="SQLite metadata index to update")
    args = parser.parse_args()

    for chunk_path in args.chunks:
        chunk = ReplayReader.read_file(chunk_path)
        print(
            f"{chunk_path}: ok, samples={chunk.header.sample_count}, "
            f"model={chunk.header.model_version}, "
            f"generator={chunk.header.generator_version}"
        )
        if args.index is not None:
            index_replay_file(args.index, chunk_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
