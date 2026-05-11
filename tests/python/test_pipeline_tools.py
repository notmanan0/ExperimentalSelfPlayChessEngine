from __future__ import annotations

from pathlib import Path

import pytest

from tests.python.test_distributed_training import _chunk_bytes, _sample_bytes
from tools.clean import clean_targets
from tools.convert.index_replay_dir import index_replay_dir
from tools.convert.summarize_replay import summarize_replay_dir
from tools.run_pipeline import promote_candidate, run_stage


def _write_chunk(path: Path, sample_count: int = 2) -> None:
    samples = [
        _sample_bytes(game_id=1, ply_index=ply, move_from=12, move_to=28)
        for ply in range(sample_count)
    ]
    path.write_bytes(_chunk_bytes(samples))


def test_index_replay_dir_keeps_scanning_after_invalid_chunk(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_chunk(replay_dir / "valid.cmrep", sample_count=3)
    (replay_dir / "invalid.cmrep").write_bytes(b"not a replay chunk")

    summary = index_replay_dir(
        replay_dir,
        tmp_path / "replay.sqlite",
        progress_interval=0,
    )

    assert summary.files_scanned == 2
    assert summary.files_indexed == 1
    assert summary.files_failed == 1
    assert summary.total_samples == 3
    assert "invalid.cmrep" in summary.failures[0].path


def test_replay_summary_reports_totals_versions_sizes_and_short_warnings(tmp_path: Path):
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    _write_chunk(replay_dir / "short.cmrep", sample_count=2)
    _write_chunk(replay_dir / "long.cmrep", sample_count=5)

    summary = summarize_replay_dir(
        replay_dir,
        short_sample_threshold=3,
        progress_interval=0,
    )

    assert summary.chunk_count == 2
    assert summary.total_samples == 7
    assert summary.min_samples_per_chunk == 2
    assert summary.max_samples_per_chunk == 5
    assert summary.model_versions == (1,)
    assert summary.generator_versions == (1,)
    assert summary.total_size_bytes > 0
    assert summary.shortest_file is not None
    assert summary.longest_file is not None
    assert len(summary.short_game_warnings) == 1


def test_clean_requires_yes_for_weights():
    with pytest.raises(PermissionError, match="requires --yes"):
        clean_targets(["weights"], yes=False, dry_run=True)


def test_pipeline_stage_reports_failed_command():
    with pytest.raises(RuntimeError, match="exit=7"):
        run_stage(1, "fail", ["python", "-c", "import sys; sys.exit(7)"])


def test_promotion_preserves_previous_best_and_records_history(tmp_path: Path):
    weights = tmp_path / "weights"
    weights.mkdir()
    old_best = weights / "best.engine"
    old_best.write_text("old", encoding="utf-8")
    candidate = tmp_path / "candidate.engine"
    candidate.write_text("new", encoding="utf-8")

    copied = promote_candidate(candidate, 2, weights_dir=weights)

    assert (weights / "history" / "model_000001.engine").read_text(encoding="utf-8") == "old"
    assert (weights / "history" / "model_000002.engine").read_text(encoding="utf-8") == "new"
    assert (weights / "best.engine").read_text(encoding="utf-8") == "new"
    assert len(copied) == 3
