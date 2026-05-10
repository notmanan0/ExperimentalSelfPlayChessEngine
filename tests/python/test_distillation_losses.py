import torch

from chessmoe.training.distill_losses import (
    HardTargets,
    compute_distillation_loss,
    teacher_targets_from_output,
)


class DummyOutput:
    def __init__(self, policy_logits: torch.Tensor, wdl_logits: torch.Tensor, moves_left: torch.Tensor):
        self.policy_logits = policy_logits
        self.wdl_logits = wdl_logits
        self.moves_left = moves_left


def test_distillation_loss_is_zero_when_student_matches_teacher():
    torch.manual_seed(7)
    policy_logits = torch.randn(2, 6)
    wdl_logits = torch.randn(2, 3)
    moves_left = torch.tensor([4.0, 9.0], dtype=torch.float32)

    teacher_output = DummyOutput(policy_logits, wdl_logits, moves_left)
    student_output = DummyOutput(policy_logits, wdl_logits, moves_left)

    teacher_targets = teacher_targets_from_output(teacher_output, temperature=1.0)
    hard_targets = HardTargets(
        policy=torch.softmax(policy_logits, dim=-1),
        wdl=torch.tensor([0, 2], dtype=torch.long),
        value=torch.tensor([0.1, -0.2], dtype=torch.float32),
        moves_left=moves_left,
    )

    losses = compute_distillation_loss(
        student_output,
        teacher_targets,
        hard_targets,
        temperature=1.0,
        policy_kl_weight=1.0,
        wdl_kl_weight=1.0,
        value_weight=1.0,
        moves_left_weight=1.0,
        hard_target_weight=0.0,
        hard_value_weight=0.0,
        hard_moves_left_weight=0.0,
    )

    assert torch.allclose(losses.total, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(losses.policy_kl, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(losses.wdl_kl, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(losses.value, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(losses.moves_left, torch.tensor(0.0), atol=1.0e-6)
