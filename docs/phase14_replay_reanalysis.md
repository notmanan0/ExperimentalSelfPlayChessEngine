# Phase 14: Replay Reanalysis

## Objective

Revisit stored replay positions with a newer evaluator and write versioned target overlays that can improve policy and value targets without modifying the original binary replay chunks.

Phase 14 preserves all original `.cmrep` data. Reanalysis results are appended to the SQLite metadata index as separate `reanalysis_targets` records. Training can use original targets, latest reanalysis targets, or a deterministic mixture.

## Files To Create

- `python/replay/reanalysis.py`
- `tools/convert/reanalyze_replay.py`
- `tools/benchmark/benchmark_reanalysis.py`
- `configs/training/reanalysis_phase14.json`
- `cpp/selfplay/include/chessmoe/selfplay/reanalysis_config.h`
- `cpp/selfplay/src/reanalysis_config.cpp`
- `tests/python/test_reanalysis.py`
- `tests/cpp/selfplay/test_reanalysis_config.cpp`
- `docs/phase14_replay_reanalysis.md`

Files updated:

- `python/replay/index.py`
- `python/chessmoe/training/data.py`
- `python/chessmoe/training/config.py`
- `python/chessmoe/training/train.py`
- `cpp/selfplay/CMakeLists.txt`
- `tests/cpp/selfplay/CMakeLists.txt`

## Core Data Structures

### `ReanalysisConfig`

```python
@dataclass(frozen=True)
class ReanalysisConfig:
    replay_index: str | Path
    output_index: str | Path
    current_model_version: int
    search_budget: int
    reanalysis_timestamp_ms: int | None = None
    source_model_versions: frozenset[int] | None = None
    older_than_timestamp_ms: int | None = None
    minimum_sampling_priority: float = 0.0
    max_chunks: int | None = None
    max_samples: int | None = None
```

### SQLite Schema Changes

```sql
create table if not exists chunk_priorities (
  path text primary key,
  sampling_priority real not null default 1.0,
  updated_at_ms integer not null default 0
);

create table if not exists reanalysis_targets (
  id integer primary key autoincrement,
  chunk_path text not null,
  game_id integer not null,
  ply_index integer not null,
  source_model_version integer not null,
  model_version integer not null,
  search_budget integer not null,
  reanalysis_timestamp_ms integer not null,
  root_value real not null,
  policy_json text not null,
  created_at_ms integer not null,
  unique(chunk_path, game_id, ply_index, model_version, search_budget, reanalysis_timestamp_ms)
);
```

### `ReanalysisTarget`

Stores one versioned target overlay for one stored position:

- `chunk_path`
- `game_id`
- `ply_index`
- `source_model_version`
- `model_version`
- `search_budget`
- `reanalysis_timestamp_ms`
- `root_value`
- `policy`: JSON list of `{move, visit_count, probability}`

### Training Target Policy

`ReplayDataset.from_index()` now accepts:

- `target_policy="original"`
- `target_policy="latest_reanalysis"`
- `target_policy="mix"`
- `reanalysis_fraction`
- `reanalysis_seed`

Original features and move-left targets stay tied to the replay chunk. Policy and scalar value-derived WDL targets can come from the latest reanalysis overlay.

## Main Algorithms

### Chunk Selection

```text
select_replay_chunks(index, config):
  start from indexed chunks
  left join chunk_priorities
  filter by source_model_versions when provided
  filter by creation_timestamp_ms < older_than_timestamp_ms when provided
  filter by sampling_priority >= minimum_sampling_priority
  order by sampling_priority descending, age ascending, path
  limit to max_chunks when provided
```

### Reanalysis Pipeline

```text
reanalyze_index(config, analyzer):
  validate current_model_version and search_budget
  timestamp = config timestamp or current time
  chunks = select_replay_chunks(...)
  for each selected chunk:
    read binary chunk with ReplayReader
    for each sample:
      decode stored board, side, rule state, legal moves, and original targets
      run current model root search/analyzer on the stored position
      create updated policy target over stored legal moves
      create updated root value
      append ReanalysisTarget to reanalysis_targets
      stop at max_samples when provided
  return throughput summary
```

Phase 14 includes a root-level `TorchModelAnalyzer` and deterministic `SimpleModelAnalyzer`. Full C++ MCTS reanalysis can be added later once Python/C++ evaluator binding is ready. The storage and training interfaces already support replacement with stronger search outputs.

### Dataset Mixing

```text
ReplayDataset.__getitem__(i):
  sample = original replay sample
  if target_policy == original:
    use original policy and game-result WDL
  if target_policy == latest_reanalysis:
    use latest target for (chunk_path, game_id, ply_index) when present
  if target_policy == mix:
    use deterministic RNG(seed, chunk, game_id, ply) < reanalysis_fraction
  features always come from original replay sample
```

## Tests

- `tests/python/test_reanalysis.py`
  - Select replay chunks by model version, age, and sampling priority.
  - Preserve original chunk bytes after reanalysis.
  - Write and load versioned reanalysis targets from the metadata index.
  - Prefer latest reanalysis target when requested.
  - Mix original and reanalyzed targets deterministically.
  - Keep target JSON schema stable.

- `tests/cpp/selfplay/test_reanalysis_config.cpp`
  - Validate C++ `ReanalysisConfig`.
  - Confirm default storage mode is appended target records.

## Commands

Run reanalysis:

```powershell
python tools/convert/reanalyze_replay.py --replay-index data/replay/replay.sqlite --current-model-version 14 --search-budget 128 --max-chunks 100
```

Benchmark reanalysis throughput:

```powershell
python tools/benchmark/benchmark_reanalysis.py --replay-index data/replay/replay.sqlite --current-model-version 14 --search-budget 128 --max-samples 10000
```

Train with mixed original and reanalyzed targets:

```powershell
python -m chessmoe.training.train --config configs/training/reanalysis_phase14.json
```

Run tests:

```powershell
python -m pytest tests/python/test_reanalysis.py --basetemp python-test-output/pytest-phase14
cmake --build build-nmake --config Debug --target reanalysis_config_tests
ctest --test-dir build-nmake -R reanalysis_config_tests --output-on-failure
```

## Completion Criteria

- Reanalysis selects chunks by model version, age, and sampling priority.
- Stored replay positions decode from existing replay chunks.
- Current model analyzer produces updated policy targets and root values.
- Targets record model version, search budget, and reanalysis timestamp.
- Original replay chunks remain unchanged.
- Reanalysis targets are versioned in the metadata index.
- Training can use original, latest reanalysis, or mixed targets.
- CLI command exists for reanalysis.
- Throughput benchmark emits positions/sec.
- Tests cover preservation, replacement policy, metadata indexing, and dataset mixing.
- No MoE work is included.

## Common Failure Modes

- Rewriting `.cmrep` chunks and losing the original training target.
- Failing to key targets by `chunk_path`, `game_id`, and `ply_index`.
- Mixing targets nondeterministically, causing irreproducible training.
- Selecting only by model version and repeatedly reanalyzing low-priority data.
- Treating reanalysis root values as game outcomes without recording the model version and search budget.
- Letting reanalysis records silently replace newer records with older timestamps.
- Training on reanalysis targets without preserving feature/rule-state decoding from the original sample.

## Next Step

Run a small reanalysis pass over Phase 13 self-play chunks, train with `target_policy="mix"`, and compare the candidate against the current best model through arena gating. Do not start MoE work yet.
