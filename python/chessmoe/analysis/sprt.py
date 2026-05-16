"""Sequential Probability Ratio Test (SPRT) for arena decisions.

Implements the SPRT algorithm for deciding whether a candidate model
is stronger than the current best model, based on game results.

Reference: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class SprtDecision(str, Enum):
    ACCEPT_H0 = "accept_h0"      # Candidate is NOT stronger
    ACCEPT_H1 = "accept_h1"      # Candidate IS stronger
    CONTINUE = "continue"         # Need more games


@dataclass(frozen=True)
class SprtConfig:
    """SPRT configuration.

    H0: candidate's Elo difference <= elo0 (null hypothesis)
    H1: candidate's Elo difference >= elo1 (alternative hypothesis)
    """
    elo0: float = 0.0      # Null hypothesis: no improvement
    elo1: float = 35.0     # Alternative hypothesis: 35 Elo improvement
    alpha: float = 0.05    # False positive rate (accept H1 when H0 is true)
    beta: float = 0.05     # False negative rate (accept H0 when H1 is true)


@dataclass(frozen=True)
class SprtState:
    """Current SPRT state after some games."""
    wins: int
    losses: int
    draws: int
    llr: float              # Log-likelihood ratio
    lower_bound: float      # Lower acceptance bound
    upper_bound: float      # Upper acceptance bound
    decision: SprtDecision
    games_played: int
    score_rate: float


def elo_to_score(elo: float) -> float:
    """Convert Elo difference to expected score (win probability)."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def score_to_elo(score: float) -> float:
    """Convert score rate to Elo difference."""
    if score <= 0.0:
        return -1000.0
    if score >= 1.0:
        return 1000.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def llr_single_game(result: str, s0: float, s1: float) -> float:
    """Compute log-likelihood ratio contribution for a single game.

    result: "win", "loss", or "draw"
    s0: expected score under H0
    s1: expected score under H1
    """
    draw_rate = 0.3  # Assumed draw rate for normalization
    win_rate0 = s0 * (1.0 - draw_rate)
    loss_rate0 = (1.0 - s0) * (1.0 - draw_rate)
    win_rate1 = s1 * (1.0 - draw_rate)
    loss_rate1 = (1.0 - s1) * (1.0 - draw_rate)

    if result == "win":
        if win_rate1 <= 0:
            return 100.0
        if win_rate0 <= 0:
            return 100.0
        return math.log(win_rate1 / win_rate0)
    elif result == "loss":
        if loss_rate1 <= 0:
            return -100.0
        if loss_rate0 <= 0:
            return -100.0
        return math.log(loss_rate1 / loss_rate0)
    else:  # draw
        return 0.0


def compute_sprt(
    wins: int,
    losses: int,
    draws: int,
    config: SprtConfig | None = None,
) -> SprtState:
    """Compute SPRT decision based on game results.

    Returns the current state including LLR and decision.
    """
    if config is None:
        config = SprtConfig()

    games = wins + losses + draws
    if games == 0:
        return SprtState(
            wins=0, losses=0, draws=0,
            llr=0.0,
            lower_bound=math.log(config.beta / (1.0 - config.alpha)),
            upper_bound=math.log((1.0 - config.beta) / config.alpha),
            decision=SprtDecision.CONTINUE,
            games_played=0,
            score_rate=0.0,
        )

    s0 = elo_to_score(config.elo0)
    s1 = elo_to_score(config.elo1)

    llr = 0.0
    for _ in range(wins):
        llr += llr_single_game("win", s0, s1)
    for _ in range(losses):
        llr += llr_single_game("loss", s0, s1)
    for _ in range(draws):
        llr += llr_single_game("draw", s0, s1)

    lower_bound = math.log(config.beta / (1.0 - config.alpha))
    upper_bound = math.log((1.0 - config.beta) / config.alpha)

    if llr >= upper_bound:
        decision = SprtDecision.ACCEPT_H1
    elif llr <= lower_bound:
        decision = SprtDecision.ACCEPT_H0
    else:
        decision = SprtDecision.CONTINUE

    score_rate = (wins + 0.5 * draws) / games if games > 0 else 0.0

    return SprtState(
        wins=wins,
        losses=losses,
        draws=draws,
        llr=llr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        decision=decision,
        games_played=games,
        score_rate=score_rate,
    )


def sprt_summary(state: SprtState) -> str:
    """Format SPRT state as a human-readable string."""
    lines = [
        f"SPRT: {state.decision.value}",
        f"  Games: {state.games_played} (W={state.wins} L={state.losses} D={state.draws})",
        f"  Score: {state.score_rate:.1%}",
        f"  LLR: {state.llr:.3f} [{state.lower_bound:.3f}, {state.upper_bound:.3f}]",
        f"  Elo estimate: {score_to_elo(state.score_rate):.1f}",
    ]
    return "\n".join(lines)
