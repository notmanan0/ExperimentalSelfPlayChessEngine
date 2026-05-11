# Git Report

Generated: 2026-05-11

## Repository State

- Branch: `master`
- Working tree: active development for the user-facing AlphaZero generation pipeline.
- Generated metrics such as `*.jsonl` are ignored rather than committed.

## Current Focus

- Preflight determinism cleanup for one-rank DDP training.
- User-facing `selfplay` executable and evaluator selection.
- Replay indexing and summary tools.
- Pipeline cleanup, orchestration, and guarded promotion tooling.

## Verification Commands

```powershell
ctest --test-dir build-nmake --output-on-failure
python -m pytest tests/python --basetemp python-test-output/pytest-current
```

## Notes

- `gpu_selfplay_benchmark` remains a compatibility benchmark.
- `selfplay` is the intended production self-play command.
- TensorRT remains optional at build time; selecting it without compiled support must fail clearly.
