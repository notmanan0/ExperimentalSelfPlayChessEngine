from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from replay.reanalysis import ReanalysisConfig, SimpleModelAnalyzer, reanalyze_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark replay reanalysis throughput")
    parser.add_argument("--replay-index", required=True, type=Path)
    parser.add_argument("--output-index", type=Path)
    parser.add_argument("--current-model-version", required=True, type=int)
    parser.add_argument("--search-budget", type=int, default=64)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--source-model-version", action="append", type=int)
    args = parser.parse_args()

    summary = reanalyze_index(
        ReanalysisConfig(
            replay_index=args.replay_index,
            output_index=args.output_index or args.replay_index,
            current_model_version=args.current_model_version,
            search_budget=args.search_budget,
            source_model_versions=(
                frozenset(args.source_model_version)
                if args.source_model_version
                else None
            ),
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
