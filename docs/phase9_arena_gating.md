# Phase 9 Arena Gating And Model Promotion

## Objective

Compare a candidate model against the current best model before promotion. The phase adds deterministic arena scheduling, result tracking, threshold-based promotion, a model registry, metadata logging, and a CLI. The current Python arena backend is placeholder-seeded; it is useful for validating scheduling and metadata plumbing, but it must not produce meaningful promotion decisions.

## Files To Create

- `python/chessmoe/analysis/arena.py`
- `python/chessmoe/analysis/__init__.py`
- `cpp/selfplay/include/chessmoe/selfplay/arena_config.h`
- `cpp/selfplay/src/arena_config.cpp`
- `configs/arena/tiny_arena.json`
- `tests/python/test_arena_gating.py`
- `tests/cpp/selfplay/test_arena_config.cpp`
- `data/arena/.gitkeep`

## Core Data Structures

- `ArenaConfig`: candidate path, best path, opening suite, games per opening, equal search budget, seed, promotion thresholds, optional SPRT config, metadata path, and registry path.
- `ArenaGameSpec`: one scheduled game with opening, candidate side, budget, and deterministic seed.
- `ArenaGameResult`: candidate-perspective result for one game.
- `ArenaSummary`: wins, losses, draws, WDL score rate, Elo placeholder, and confidence interval placeholder.
- `ModelRegistry`: JSON registry with `candidate`, `best`, `rejected`, and `promoted` states.
- C++ `ArenaConfig`, `ArenaGameSpec`, `ArenaSummary`, and `PromotionRule`: boundary types for the future evaluator-backed match runner.

## Main Algorithms

1. Validate the opening suite and require an even `games_per_opening` so every opening has side swaps.
2. Build a deterministic schedule from `seed`, opening index, and local game index.
3. Run candidate-vs-best games with equal search budgets through an arena backend.
4. Count candidate-perspective wins, losses, and draws.
5. Compute WDL score rate as:

$$
\text{score} = \frac{\text{wins} + 0.5 \cdot \text{draws}}{\text{games}}
$$

6. Apply the promotion rule: insufficient games blocks promotion, optional SPRT returns a continuation skeleton, otherwise score threshold promotes or rejects.
7. Write metadata JSON with config, results, summary, decision, and C++/Python boundary notes.
8. Update the model registry by moving the candidate to `promoted` and `best`, or to `rejected`.

## Current Production Status

The existing arena flow is not yet a real model-vs-model search match runner. User-facing pipeline commands therefore refuse meaningful promotion from arena results until the backend plays actual candidate-vs-baseline search games with real evaluator outputs.

Controlled promotion is still available for manually accepted artifacts:

```powershell
python tools/run_pipeline.py promote --candidate weights/candidate.engine --version 2
```

Promotion preserves the previous best under `weights/history/model_000001.*` style paths before writing `weights/best.*`.

## Arena Pseudocode

```text
load ArenaConfig
schedule = build_match_schedule(config)
for game in schedule:
    result = backend.play(game, candidate_model, best_model)
summary = summarize(results)
decision = evaluate_promotion(summary, config)
write metadata
if registry is configured:
    promote or reject candidate
```

## C++ / Python Boundary Design

Python owns orchestration, metadata, registry updates, and CLI. C++ owns engine-facing arena primitives: schedule construction, side assignment, search-budget invariants, and promotion-rule parity. A later phase can replace the Python test backend with a subprocess or binding that calls the existing C++ evaluator/search path for real model-backed games. CUDA inference remains out of scope.

## Tests

- Python side-swapping schedule test.
- Python deterministic result test for fixed seeds.
- Python promotion threshold test.
- Python metadata and registry recording test.
- C++ side-swapping and promotion-rule test.

## Commands

```powershell
python -m pytest tests/python/test_arena_gating.py --basetemp python-test-output/pytest
python -m pytest tests/python --basetemp python-test-output/pytest
$env:PYTHONPATH='python'
python -m chessmoe.analysis.arena --config configs/arena/tiny_arena.json
ctest --test-dir build-nmake --output-on-failure
```

## Completion Criteria

- Arena scheduling side-swaps every opening.
- Candidate and best use equal search budgets.
- Runs are deterministic for a fixed seed.
- Results include wins, losses, draws, WDL rate, Elo placeholder, and confidence interval placeholder.
- Promotion thresholds promote or reject correctly.
- Registry and metadata files are written.
- C++ boundary tests pass.
- Placeholder arena status is explicit and cannot silently gate production promotion.

## Common Failure Modes

- Using odd games per opening and silently losing side-swap fairness.
- Reporting WDL from the game winner perspective instead of the candidate perspective.
- Letting candidate and best use different search budgets.
- Updating the best model without preserving rejected/promoted history.
- Treating the Elo and confidence interval placeholders as statistically complete.
- Running arena games with nondeterministic seeds.
- Treating placeholder-seeded arena results as model strength evidence.

## Next Step

Replace the placeholder arena backend with real model-vs-model search games using explicit evaluator backends and equal search budgets before enabling arena-driven promotion.
