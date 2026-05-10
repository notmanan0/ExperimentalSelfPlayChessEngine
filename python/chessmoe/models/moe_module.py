from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 8
    top_k_training: int = 2
    top_k_inference: int = 1
    capacity_factor: float = 1.25
    load_balance_coeff: float = 0.01
    router_entropy_coeff: float = 0.001
    router_noise_std: float = 0.1
    dense_fallback: bool = False
    expert_dropout: float = 0.0

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class MoERouterOutput(NamedTuple):
    dispatch_weights: torch.Tensor
    expert_indices: torch.Tensor
    router_logits: torch.Tensor
    load_balance_loss: torch.Tensor
    router_entropy_loss: torch.Tensor
    num_dropped_tokens: torch.Tensor
    expert_usage: torch.Tensor


class MoERouter(nn.Module):
    def __init__(self, config: MoEConfig, d_model: int) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(d_model, config.num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        force_top_k: int | None = None,
    ) -> MoERouterOutput:
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.reshape(num_tokens, d_model)

        router_logits = self.gate(x_flat)

        if self.training and self.config.router_noise_std > 0:
            noise = torch.randn_like(router_logits) * self.config.router_noise_std
            router_logits = router_logits + noise

        top_k = force_top_k if force_top_k is not None else (
            self.config.top_k_training if self.training else self.config.top_k_inference
        )

        top_k = min(top_k, self.config.num_experts)
        dispatch_weights, expert_indices = torch.topk(
            F.softmax(router_logits, dim=-1), top_k, dim=-1
        )

        total_capacity = int(num_tokens * top_k * self.config.capacity_factor)
        per_expert_capacity = max(1, total_capacity // self.config.num_experts)

        expert_mask = F.one_hot(expert_indices, self.config.num_experts).sum(dim=1)
        tokens_per_expert = expert_mask.sum(dim=0).float()

        capacity_mask = tokens_per_expert > per_expert_capacity
        num_dropped = capacity_mask.sum().detach()

        if self.training and self.config.expert_dropout > 0:
            drop_mask = torch.rand_like(dispatch_weights) < self.config.expert_dropout
            dispatch_weights = dispatch_weights.masked_fill(drop_mask, 0.0)
            dispatch_weights = dispatch_weights / (dispatch_weights.sum(dim=-1, keepdim=True) + 1e-9)

        probs = F.softmax(router_logits, dim=-1)
        avg_probs = probs.mean(dim=0)
        load_balance_loss = (tokens_per_expert.float() / max(1, num_tokens)) * avg_probs
        load_balance_loss = load_balance_loss.sum() * self.config.num_experts

        log_probs = F.log_softmax(router_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        target_entropy = torch.log(torch.tensor(float(self.config.num_experts)))
        router_entropy_loss = (target_entropy - entropy).clamp(min=0.0)

        expert_usage = tokens_per_expert / max(1, num_tokens)

        return MoERouterOutput(
            dispatch_weights=dispatch_weights,
            expert_indices=expert_indices,
            router_logits=router_logits,
            load_balance_loss=load_balance_loss,
            router_entropy_loss=router_entropy_loss,
            num_dropped_tokens=num_dropped,
            expert_usage=expert_usage,
        )


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


class SparseMoEFFN(nn.Module):
    def __init__(self, config: MoEConfig, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.ffn_dim = ffn_dim

        self.router = MoERouter(config, d_model)
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, ffn_dim, dropout) for _ in range(config.num_experts)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoERouterOutput]:
        residual = x
        x_norm = self.layer_norm(x)

        router_out = self.router(x_norm)
        dispatch_weights = router_out.dispatch_weights
        expert_indices = router_out.expert_indices

        batch_size, seq_len, d_model = x_norm.shape
        num_tokens = batch_size * seq_len
        x_flat = x_norm.reshape(num_tokens, d_model)

        top_k = dispatch_weights.shape[1]
        output = torch.zeros_like(x_flat)

        for e_idx in range(self.config.num_experts):
            expert_mask = (expert_indices == e_idx).any(dim=1)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.nonzero(as_tuple=False).squeeze(-1)
            expert_input = x_flat[token_indices]

            expert_output = self.experts[e_idx](expert_input)

            for k in range(top_k):
                k_mask = expert_indices[token_indices, k] == e_idx
                if k_mask.any():
                    selected = token_indices[k_mask]
                    weights = dispatch_weights[selected, k].unsqueeze(-1)
                    output[selected] += weights * expert_output[k_mask]

        output = output.reshape(batch_size, seq_len, d_model)
        return residual + output, router_out


class DenseFFNFallback(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ffn = ExpertFFN(d_model, ffn_dim, dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        residual = x
        x_norm = self.layer_norm(x)
        return residual + self.ffn(x_norm), None
