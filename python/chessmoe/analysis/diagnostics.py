from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Any

from chessmoe.models.encoding import encode_fen, move_to_index, NUM_MOVE_BUCKETS


def _as_flat_sequence(values) -> list[float]:
    import torch
    if isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().cpu().flatten().tolist()]
    if hasattr(values, "flatten") and callable(values.flatten):
        values = values.flatten()
    return [float(v) for v in values]


@dataclass
class DiagnosticsResult:
    metric: str
    value: float
    details: dict[str, Any]


def compute_policy_entropy(policy_target, legal_moves: list[str] | None = None) -> float:
    """Compute Shannon entropy of a policy distribution."""
    probs = _as_flat_sequence(policy_target)

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_topk_accuracy(
    predicted_policy,
    target_policy,
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """Compute top-k accuracy: does the target's best move appear in predicted top-k?"""
    import torch
    if k_values is None:
        k_values = [1, 3, 5, 10]

    pred = _as_flat_sequence(predicted_policy)
    tgt = _as_flat_sequence(target_policy)

    target_move = max(range(len(tgt)), key=tgt.__getitem__)
    pred_order = sorted(range(len(pred)), key=pred.__getitem__, reverse=True)

    results = {}
    for k in k_values:
        results[k] = 1.0 if target_move in pred_order[:k] else 0.0
    return results


def analyze_replay_policy_targets(db_path: Path) -> dict[str, Any]:
    """Analyze policy target quality across replay data."""
    from replay.reader import ReplayReader
    import sqlite3
    import torch

    with sqlite3.connect(db_path) as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT path FROM chunks ORDER BY creation_timestamp_ms"
        ).fetchall()]

    entropies: list[float] = []
    top1_concentrations: list[float] = []
    total_samples = 0

    for path_str in paths[:100]:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            chunk = ReplayReader.read_file(path)
            for sample in chunk.samples:
                total_samples += 1
                target = torch.zeros(NUM_MOVE_BUCKETS, dtype=torch.float32)
                total_visits = sum(max(0, e.visit_count) for e in sample.policy)
                if total_visits > 0:
                    for entry in sample.policy:
                        idx = move_to_index(entry.move)
                        target[idx] = max(0, entry.visit_count) / total_visits

                ent = compute_policy_entropy(target)
                entropies.append(ent)

                if total_visits > 0:
                    top1 = target.max().item()
                    top1_concentrations.append(top1)

        except Exception:
            continue

    avg_entropy = sum(entropies) / max(1, len(entropies))
    avg_top1 = sum(top1_concentrations) / max(1, len(top1_concentrations))

    return {
        "total_samples": total_samples,
        "average_entropy": avg_entropy,
        "average_top1_concentration": avg_top1,
        "entropy_std": _std(entropies),
        "min_entropy": min(entropies) if entropies else 0,
        "max_entropy": max(entropies) if entropies else 0,
        "degenerate_rate": sum(1 for e in entropies if e < 0.1) / max(1, len(entropies)),
    }


def analyze_value_calibration(db_path: Path) -> dict[str, Any]:
    """Analyze how well predicted values match actual outcomes."""
    from replay.reader import ReplayReader
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        paths = [r[0] for r in conn.execute(
            "SELECT path FROM chunks ORDER BY creation_timestamp_ms"
        ).fetchall()]

    value_by_outcome: dict[str, list[float]] = {
        "white_win": [], "draw": [], "black_win": [],
    }
    total = 0

    for path_str in paths[:100]:
        path = Path(path_str)
        if not path.exists():
            continue
        try:
            chunk = ReplayReader.read_file(path)
            for sample in chunk.samples:
                total += 1
                outcome = sample.final_wdl
                value_by_outcome[outcome].append(sample.root_value)
        except Exception:
            continue

    result: dict[str, Any] = {"total_samples": total}
    for outcome, values in value_by_outcome.items():
        if values:
            result[f"{outcome}_count"] = len(values)
            result[f"{outcome}_mean_value"] = sum(values) / len(values)
            result[f"{outcome}_std_value"] = _std(values)
        else:
            result[f"{outcome}_count"] = 0
            result[f"{outcome}_mean_value"] = 0
            result[f"{outcome}_std_value"] = 0

    return result


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
