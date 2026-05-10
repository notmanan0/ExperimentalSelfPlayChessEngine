from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from chessmoe.models.tiny_model import TinyModelOutput


@dataclass(frozen=True)
class TinyLossTargets:
    policy: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor | None = None


@dataclass(frozen=True)
class TinyLossOutput:
    total: torch.Tensor
    policy: torch.Tensor
    wdl: torch.Tensor
    moves_left: torch.Tensor


def soft_policy_cross_entropy(policy_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if policy_logits.shape != target.shape:
        raise ValueError("policy logits and target must have identical shape")
    log_probs = F.log_softmax(policy_logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def compute_tiny_loss(
    output: TinyModelOutput,
    targets: TinyLossTargets,
    moves_left_weight: float = 0.01,
) -> TinyLossOutput:
    policy_loss = soft_policy_cross_entropy(output.policy_logits, targets.policy)
    wdl_loss = F.cross_entropy(output.wdl_logits, targets.wdl)

    if targets.moves_left is None:
        moves_left_loss = output.moves_left.new_zeros(())
    else:
        moves_left_loss = F.mse_loss(output.moves_left, targets.moves_left)

    total = policy_loss + wdl_loss + moves_left_weight * moves_left_loss
    return TinyLossOutput(
        total=total,
        policy=policy_loss,
        wdl=wdl_loss,
        moves_left=moves_left_loss,
    )

