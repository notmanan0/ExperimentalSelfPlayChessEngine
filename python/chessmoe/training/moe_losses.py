from __future__ import annotations

from dataclasses import dataclass

import torch

from chessmoe.models.moe_module import MoERouterOutput


@dataclass(frozen=True)
class MoELossComponents:
    load_balance: torch.Tensor
    router_entropy: torch.Tensor
    total_dropped_tokens: torch.Tensor
    expert_usage: torch.Tensor


def compute_moe_auxiliary_loss(
    router_outputs: tuple[MoERouterOutput, ...],
    load_balance_coeff: float = 0.01,
    router_entropy_coeff: float = 0.001,
) -> MoELossComponents:
    if len(router_outputs) == 0:
        zero = torch.tensor(0.0)
        return MoELossComponents(
            load_balance=zero,
            router_entropy=zero,
            total_dropped_tokens=zero,
            expert_usage=zero,
        )

    total_lb = torch.tensor(0.0, device=router_outputs[0].load_balance_loss.device)
    total_entropy = torch.tensor(0.0, device=router_outputs[0].router_entropy_loss.device)
    total_dropped = torch.tensor(0.0, device=router_outputs[0].num_dropped_tokens.device)
    usage_list = []

    for ro in router_outputs:
        total_lb = total_lb + ro.load_balance_loss
        total_entropy = total_entropy + ro.router_entropy_loss
        total_dropped = total_dropped + ro.num_dropped_tokens
        usage_list.append(ro.expert_usage)

    num_layers = len(router_outputs)
    avg_usage = torch.stack(usage_list).mean(dim=0)

    return MoELossComponents(
        load_balance=total_lb * load_balance_coeff / num_layers,
        router_entropy=total_entropy * router_entropy_coeff / num_layers,
        total_dropped_tokens=total_dropped / num_layers,
        expert_usage=avg_usage,
    )


def moe_loss_from_model_output(
    model_output,
    load_balance_coeff: float = 0.01,
    router_entropy_coeff: float = 0.001,
) -> MoELossComponents:
    if not hasattr(model_output, "router_outputs"):
        zero = torch.tensor(0.0)
        return MoELossComponents(
            load_balance=zero,
            router_entropy=zero,
            total_dropped_tokens=zero,
            expert_usage=zero,
        )

    return compute_moe_auxiliary_loss(
        model_output.router_outputs,
        load_balance_coeff=load_balance_coeff,
        router_entropy_coeff=router_entropy_coeff,
    )
