"""Tests for SPRT (Sequential Probability Ratio Test)."""

from __future__ import annotations

import math
import pytest

from chessmoe.analysis.sprt import (
    SprtConfig,
    SprtDecision,
    compute_sprt,
    elo_to_score,
    score_to_elo,
    sprt_summary,
)


class TestSprtCore:
    def test_elo_to_score_zero(self) -> None:
        assert abs(elo_to_score(0.0) - 0.5) < 1e-10

    def test_elo_to_score_positive(self) -> None:
        assert elo_to_score(100.0) > 0.5

    def test_elo_to_score_negative(self) -> None:
        assert elo_to_score(-100.0) < 0.5

    def test_score_to_elo_roundtrip(self) -> None:
        for elo in [-200, -100, 0, 100, 200]:
            score = elo_to_score(elo)
            recovered = score_to_elo(score)
            assert abs(recovered - elo) < 1.0

    def test_no_games_returns_continue(self) -> None:
        state = compute_sprt(0, 0, 0)
        assert state.decision == SprtDecision.CONTINUE
        assert state.games_played == 0


class TestSprtDecisions:
    def test_dominant_wins_accepts_h1(self) -> None:
        state = compute_sprt(wins=100, losses=10, draws=20)
        assert state.decision == SprtDecision.ACCEPT_H1
        assert state.score_rate > 0.7

    def test_dominant_losses_accepts_h0(self) -> None:
        state = compute_sprt(wins=10, losses=100, draws=20)
        assert state.decision == SprtDecision.ACCEPT_H0
        assert state.score_rate < 0.3

    def test_equal_results_continue(self) -> None:
        state = compute_sprt(wins=20, losses=20, draws=10)
        assert state.decision == SprtDecision.CONTINUE

    def test_custom_elo_bounds(self) -> None:
        config = SprtConfig(elo0=0.0, elo1=10.0, alpha=0.05, beta=0.05)
        # With narrow bounds, fewer games should decide
        state = compute_sprt(wins=60, losses=30, draws=10, config=config)
        # Should be decided or close
        assert state.decision != SprtDecision.CONTINUE or state.llr > 0


class TestSprtState:
    def test_score_rate_calculation(self) -> None:
        state = compute_sprt(wins=50, losses=30, draws=20)
        expected = (50 + 0.5 * 20) / 100
        assert abs(state.score_rate - expected) < 1e-10

    def test_llr_bounds(self) -> None:
        state = compute_sprt(wins=50, losses=30, draws=20)
        assert state.lower_bound < 0
        assert state.upper_bound > 0

    def test_games_played(self) -> None:
        state = compute_sprt(wins=10, losses=20, draws=30)
        assert state.games_played == 60

    def test_summary_format(self) -> None:
        state = compute_sprt(wins=50, losses=30, draws=20)
        summary = sprt_summary(state)
        assert "SPRT:" in summary
        assert "Games:" in summary
        assert "Score:" in summary
        assert "LLR:" in summary


class TestSprtEdgeCases:
    def test_all_wins(self) -> None:
        state = compute_sprt(wins=100, losses=0, draws=0)
        assert state.decision == SprtDecision.ACCEPT_H1
        assert state.score_rate == 1.0

    def test_all_losses(self) -> None:
        state = compute_sprt(wins=0, losses=100, draws=0)
        assert state.decision == SprtDecision.ACCEPT_H0
        assert state.score_rate == 0.0

    def test_all_draws(self) -> None:
        state = compute_sprt(wins=0, losses=0, draws=100)
        assert state.decision == SprtDecision.CONTINUE
        assert state.score_rate == 0.5

    def test_few_games_continue(self) -> None:
        state = compute_sprt(wins=3, losses=2, draws=1)
        assert state.decision == SprtDecision.CONTINUE
