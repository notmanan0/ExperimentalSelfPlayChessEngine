from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TeacherTargets:
    policy: torch.Tensor
    wdl: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor

    def to(self, device: torch.device) -> "TeacherTargets":
        return TeacherTargets(
            policy=self.policy.to(device),
            wdl=self.wdl.to(device),
            value=self.value.to(device),
            moves_left=self.moves_left.to(device),
        )


@dataclass(frozen=True)
class HardTargets:
    policy: torch.Tensor
    wdl: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor

    def to(self, device: torch.device) -> "HardTargets":
        return HardTargets(
            policy=self.policy.to(device),
            wdl=self.wdl.to(device),
            value=self.value.to(device),
            moves_left=self.moves_left.to(device),
        )


@dataclass(frozen=True)
class DistillationLossOutput:
    total: torch.Tensor
    policy_kl: torch.Tensor
    wdl_kl: torch.Tensor
    value: torch.Tensor
    moves_left: torch.Tensor


def teacher_targets_from_output(
    output,
    temperature: float,
) -> TeacherTargets:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    policy = torch.softmax(output.policy_logits / temperature, dim=-1)
    wdl = torch.softmax(output.wdl_logits / temperature, dim=-1)
    value = _scalar_value_from_wdl_probs(wdl)
    moves_left = output.moves_left
    return TeacherTargets(policy=policy, wdl=wdl, value=value, moves_left=moves_left)


def compute_distillation_loss(
    student_output,
    teacher: TeacherTargets,
    hard: HardTargets | None,
    *,
    temperature: float,
    policy_kl_weight: float,
    wdl_kl_weight: float,
    value_weight: float,
    moves_left_weight: float,
    hard_target_weight: float = 0.0,
    hard_value_weight: float = 0.0,
    hard_moves_left_weight: float = 0.0,
) -> DistillationLossOutput:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    policy_target = teacher.policy
    wdl_target = teacher.wdl
    value_target = teacher.value
    moves_left_target = teacher.moves_left

    if hard is not None and hard_target_weight > 0.0:
        policy_target = _mix_prob_targets(policy_target, hard.policy, hard_target_weight)
        wdl_target = _mix_prob_targets(
            wdl_target, _one_hot_wdl(hard.wdl), hard_target_weight
        )
    if hard is not None and hard_value_weight > 0.0:
        value_target = (1.0 - hard_value_weight) * value_target + hard_value_weight * hard.value
    if hard is not None and hard_moves_left_weight > 0.0:
        moves_left_target = (
            (1.0 - hard_moves_left_weight) * moves_left_target
            + hard_moves_left_weight * hard.moves_left
        )

    student_policy_log = F.log_softmax(student_output.policy_logits / temperature, dim=-1)
    student_wdl_log = F.log_softmax(student_output.wdl_logits / temperature, dim=-1)

    policy_kl = F.kl_div(student_policy_log, policy_target, reduction="batchmean") * (
        temperature**2
    )
    wdl_kl = F.kl_div(student_wdl_log, wdl_target, reduction="batchmean") * (
        temperature**2
    )

    student_value = _scalar_value_from_wdl_logits(student_output.wdl_logits)
    value_loss = F.mse_loss(student_value, value_target)
    moves_left_loss = F.mse_loss(student_output.moves_left, moves_left_target)

    total = (
        policy_kl_weight * policy_kl
        + wdl_kl_weight * wdl_kl
        + value_weight * value_loss
        + moves_left_weight * moves_left_loss
    )
    return DistillationLossOutput(
        total=total,
        policy_kl=policy_kl,
        wdl_kl=wdl_kl,
        value=value_loss,
        moves_left=moves_left_loss,
    )


def _mix_prob_targets(
    teacher: torch.Tensor,
    hard: torch.Tensor,
    hard_weight: float,
) -> torch.Tensor:
    if hard_weight <= 0.0:
        return teacher
    hard_weight = float(hard_weight)
    mixed = (1.0 - hard_weight) * teacher + hard_weight * hard
    return _normalize_probs(mixed)


def _normalize_probs(values: torch.Tensor) -> torch.Tensor:
    total = values.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return values / total


def _one_hot_wdl(wdl: torch.Tensor) -> torch.Tensor:
    if wdl.ndim != 1:
        raise ValueError("hard wdl targets must be 1D")
    return F.one_hot(wdl.long(), num_classes=3).to(dtype=torch.float32)


def _scalar_value_from_wdl_logits(wdl_logits: torch.Tensor) -> torch.Tensor:
    return _scalar_value_from_wdl_probs(torch.softmax(wdl_logits, dim=-1))


def _scalar_value_from_wdl_probs(wdl_probs: torch.Tensor) -> torch.Tensor:
    return wdl_probs[..., 0] - wdl_probs[..., 2]
