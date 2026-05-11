from __future__ import annotations

from pathlib import Path
import argparse
import shutil


PROTECTED_TARGETS = {"weights", "checkpoints"}
TARGET_PATHS = {
    "pytest": Path("python-test-output"),
    "replay": Path("data/replay"),
    "metrics": Path("data/metrics"),
    "weights": Path("weights"),
    "checkpoints": Path("data/checkpoints"),
}


def clean_targets(targets: list[str], *, yes: bool = False, dry_run: bool = False) -> list[Path]:
    removed: list[Path] = []
    for target in targets:
        if target not in TARGET_PATHS:
            raise ValueError(f"unknown clean target: {target}")
        if target in PROTECTED_TARGETS and not yes:
            raise PermissionError(f"clean target '{target}' requires --yes")
        path = TARGET_PATHS[target]
        print(f"clean target: {target} path={path} dry_run={dry_run}")
        if path.exists() and not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean generated chessmoe outputs.")
    parser.add_argument("targets", nargs="+", choices=sorted(TARGET_PATHS))
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        clean_targets(args.targets, yes=args.yes, dry_run=args.dry_run)
    except PermissionError as exc:
        print(f"refusing clean: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
