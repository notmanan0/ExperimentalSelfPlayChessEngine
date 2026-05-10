from __future__ import annotations

import argparse
from pathlib import Path

from chessmoe.training.config import TrainingConfig, load_training_config
from chessmoe.training.train import run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Run chessmoe training with optional DDP")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--local-rank", "--local_rank", dest="local_rank", type=int, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.local_rank is not None:
        config = TrainingConfig(**{**config.__dict__, "local_rank": args.local_rank})

    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
