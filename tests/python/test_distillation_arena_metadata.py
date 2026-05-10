import json
from pathlib import Path

from chessmoe.analysis.arena import ArenaConfig, SeededScoreMatchBackend, run_arena


OPENINGS = [
    "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
    "8/8/8/8/8/8/4K3/4k3 b - - 0 1",
]


def test_distillation_arena_metadata_includes_candidate_and_best(tmp_path: Path):
    metadata_path = tmp_path / "arena.json"
    config = ArenaConfig(
        candidate_model="student_distilled.pt",
        best_model="teacher.pt",
        openings=OPENINGS,
        games_per_opening=2,
        search_budget=8,
        seed=19,
        minimum_games=4,
        minimum_score_rate=0.5,
        metadata_path=metadata_path,
    )

    run_arena(config, SeededScoreMatchBackend(candidate_bias=0.1))

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["config"]["candidate_model"] == "student_distilled.pt"
    assert payload["config"]["best_model"] == "teacher.pt"
    assert "summary" in payload
    assert "promotion" in payload
