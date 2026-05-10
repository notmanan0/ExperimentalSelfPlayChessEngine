import json
from pathlib import Path

from chessmoe.analysis.arena import (
    ArenaConfig,
    ModelRegistry,
    PromotionDecision,
    SeededScoreMatchBackend,
    build_match_schedule,
    run_arena,
)


OPENINGS = [
    "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
    "8/8/8/8/8/8/4K3/4k3 b - - 0 1",
]


def test_match_schedule_side_swaps_every_opening():
    config = ArenaConfig(
        candidate_model="candidate.pt",
        best_model="best.pt",
        openings=OPENINGS,
        games_per_opening=2,
        search_budget=16,
        seed=7,
    )

    schedule = build_match_schedule(config)

    assert len(schedule) == 4
    assert schedule[0].candidate_color == "white"
    assert schedule[1].candidate_color == "black"
    assert schedule[2].candidate_color == "white"
    assert schedule[3].candidate_color == "black"
    assert {game.opening_fen for game in schedule} == set(OPENINGS)
    assert all(game.search_budget == 16 for game in schedule)


def test_arena_results_are_deterministic_for_fixed_seed(tmp_path: Path):
    config = ArenaConfig(
        candidate_model="candidate.pt",
        best_model="best.pt",
        openings=OPENINGS,
        games_per_opening=2,
        search_budget=8,
        seed=11,
        minimum_games=4,
        minimum_score_rate=0.5,
        metadata_path=tmp_path / "arena.json",
    )
    backend = SeededScoreMatchBackend(candidate_bias=0.2)

    first = run_arena(config, backend)
    second = run_arena(config, backend)

    assert first.results == second.results
    assert first.summary == second.summary


def test_promotion_threshold_accepts_and_rejects_candidates(tmp_path: Path):
    base = dict(
        candidate_model="candidate.pt",
        best_model="best.pt",
        openings=OPENINGS,
        games_per_opening=4,
        search_budget=8,
        minimum_games=8,
        minimum_score_rate=0.55,
        metadata_path=tmp_path / "arena.json",
    )

    promoted = run_arena(
        ArenaConfig(**base, seed=3),
        SeededScoreMatchBackend(candidate_bias=1.0),
    )
    rejected = run_arena(
        ArenaConfig(**base, seed=3),
        SeededScoreMatchBackend(candidate_bias=-1.0),
    )

    assert promoted.decision == PromotionDecision.PROMOTED
    assert rejected.decision == PromotionDecision.REJECTED
    assert promoted.summary.score_rate >= 0.55
    assert rejected.summary.score_rate < 0.55


def test_metadata_and_registry_record_promotion_state(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    metadata_path = tmp_path / "arena.json"
    registry = ModelRegistry(registry_path)
    registry.register_best("best.pt", model_id="best-001")
    registry.register_candidate("candidate.pt", model_id="candidate-001")

    result = run_arena(
        ArenaConfig(
            candidate_model="candidate.pt",
            best_model="best.pt",
            openings=OPENINGS,
            games_per_opening=4,
            search_budget=8,
            seed=5,
            minimum_games=8,
            minimum_score_rate=0.55,
            metadata_path=metadata_path,
            registry_path=registry_path,
        ),
        SeededScoreMatchBackend(candidate_bias=1.0),
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    registry_data = json.loads(registry_path.read_text(encoding="utf-8"))

    assert result.decision == PromotionDecision.PROMOTED
    assert metadata["summary"]["wins"] == result.summary.wins
    assert metadata["promotion"]["decision"] == "promoted"
    assert registry_data["best"]["path"] == "candidate.pt"
    assert registry_data["promoted"][-1]["model_id"] == "candidate-001"
