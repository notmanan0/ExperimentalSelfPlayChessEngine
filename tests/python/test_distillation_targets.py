import math

import torch

from chessmoe.training.distill_losses import teacher_targets_from_output


class DummyOutput:
    def __init__(self, policy_logits: torch.Tensor, wdl_logits: torch.Tensor, moves_left: torch.Tensor):
        self.policy_logits = policy_logits
        self.wdl_logits = wdl_logits
        self.moves_left = moves_left


def test_teacher_targets_use_temperature_and_return_value():
    policy_logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    wdl_logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    moves_left = torch.tensor([12.0], dtype=torch.float32)
    output = DummyOutput(policy_logits, wdl_logits, moves_left)

    targets = teacher_targets_from_output(output, temperature=2.0)

    assert targets.policy.shape == policy_logits.shape
    assert targets.wdl.shape == wdl_logits.shape
    assert targets.moves_left.shape == moves_left.shape
    assert torch.allclose(targets.policy.sum(dim=-1), torch.tensor([1.0]))
    assert torch.allclose(targets.wdl.sum(dim=-1), torch.tensor([1.0]))

    expected_value = targets.wdl[..., 0] - targets.wdl[..., 2]
    assert torch.allclose(targets.value, expected_value)
    assert math.isfinite(float(targets.value.item()))
