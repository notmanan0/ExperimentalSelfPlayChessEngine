from __future__ import annotations

import json
from pathlib import Path

from chessmoe.analysis.diagnostics import (
    compute_policy_entropy,
    compute_topk_accuracy,
    analyze_value_calibration,
)


def test_policy_entropy_uniform():
    import torch
    uniform = torch.ones(4192) / 4192
    ent = compute_policy_entropy(uniform)
    assert ent > 10.0


def test_policy_entropy_peaked():
    import torch
    peaked = torch.zeros(4192)
    peaked[0] = 1.0
    ent = compute_policy_entropy(peaked)
    assert ent < 0.01


def test_policy_entropy_list():
    probs = [0.5, 0.3, 0.2]
    ent = compute_policy_entropy(probs)
    assert 0 < ent < 2


def test_topk_accuracy_perfect():
    import torch
    predicted = torch.zeros(4192)
    predicted[42] = 10.0
    target = torch.zeros(4192)
    target[42] = 1.0
    result = compute_topk_accuracy(predicted, target, [1, 3, 5])
    assert result[1] == 1.0
    assert result[3] == 1.0


def test_topk_accuracy_miss():
    import torch
    predicted = torch.zeros(4192)
    predicted[0] = 10.0
    target = torch.zeros(4192)
    target[999] = 1.0
    result = compute_topk_accuracy(predicted, target, [1, 3])
    assert result[1] == 0.0


def test_golden_positions_load():
    golden_path = Path("data/test_positions/golden.json")
    assert golden_path.exists()
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    assert "positions" in data
    assert len(data["positions"]) >= 10


def test_golden_positions_have_required_fields():
    golden_path = Path("data/test_positions/golden.json")
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    for pos in data["positions"]:
        assert "name" in pos
        assert "fen" in pos
        assert "category" in pos


def test_golden_mate_in_1():
    golden_path = Path("data/test_positions/golden.json")
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    mate_positions = [p for p in data["positions"] if p["category"] == "mate_in_1"]
    assert len(mate_positions) >= 2
    for pos in mate_positions:
        assert "best_move" in pos


def test_golden_stalemate():
    golden_path = Path("data/test_positions/golden.json")
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    stalemate = [p for p in data["positions"] if p["category"] == "stalemate"]
    assert len(stalemate) >= 1


def test_golden_positions_legal_moves():
    from chessmoe.analysis.neural_arena import _get_legal_moves_from_fen
    golden_path = Path("data/test_positions/golden.json")
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    for pos in data["positions"]:
        fen = pos["fen"]
        moves = _get_legal_moves_from_fen(fen)
        if "legal_count" in pos:
            assert len(moves) == pos["legal_count"], f"{pos['name']}: expected {pos['legal_count']} got {len(moves)}"
        if "best_move" in pos:
            assert pos["best_move"] in moves, f"{pos['name']}: best_move {pos['best_move']} not in legal moves"
