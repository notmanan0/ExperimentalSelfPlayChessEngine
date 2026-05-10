from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn

from chessmoe.models.encoding import BOARD_CHANNELS, NUM_MOVE_BUCKETS
from chessmoe.models.moe_module import (
    DenseFFNFallback,
    MoEConfig,
    MoERouterOutput,
    SparseMoEFFN,
)


@dataclass(frozen=True)
class MoETransformerConfig:
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1.0e-5
    uncertainty_head: bool = False
    moe_layers: tuple[int, ...] = (1, 3)
    moe: MoEConfig = MoEConfig()
    dense_fallback_config: bool = False

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["moe"] = self.moe.to_dict()
        result["moe_layers"] = list(self.moe_layers)
        return result


class MoETransformerOutput(NamedTuple):
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor
    uncertainty: torch.Tensor | None
    router_outputs: tuple[MoERouterOutput, ...]


class MoETransformerLayer(nn.Module):
    def __init__(
        self,
        config: MoETransformerConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        d_model = config.d_model
        num_heads = config.num_heads
        ffn_dim = config.ffn_dim
        dropout = config.dropout
        eps = config.layer_norm_eps

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.ffn_layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.attn_dropout = nn.Dropout(dropout)

        use_moe = layer_idx in config.moe_layers and not config.dense_fallback_config
        if use_moe:
            self.ffn = SparseMoEFFN(config.moe, d_model, ffn_dim, dropout)
        else:
            self.ffn = DenseFFNFallback(d_model, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoERouterOutput | None]:
        residual = x
        x_norm = self.attn_layer_norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = residual + self.attn_dropout(attn_out)

        result, router_out = self.ffn(x)
        return result, router_out


class MoETransformerEncoder(nn.Module):
    def __init__(self, config: MoETransformerConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerLayer(config, i) for i in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[MoERouterOutput, ...]]:
        router_outputs = []
        for layer in self.layers:
            x, router_out = layer(x)
            if router_out is not None:
                router_outputs.append(router_out)
        x = self.final_norm(x)
        return x, tuple(router_outputs)


class MoETransformerEvaluator(nn.Module):
    def __init__(self, config: MoETransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or MoETransformerConfig()
        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        d_model = self.config.d_model
        self.piece_embedding = nn.Embedding(7, d_model)
        self.color_embedding = nn.Embedding(3, d_model)
        self.square_embedding = nn.Embedding(64, d_model)
        self.side_to_move_embedding = nn.Embedding(2, d_model)
        self.castling_projection = nn.Linear(4, d_model)
        self.en_passant_projection = nn.Linear(1, d_model)
        self.clock_projection = nn.Linear(2, d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = MoETransformerEncoder(self.config)

        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, NUM_MOVE_BUCKETS),
        )
        self.wdl_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, 3),
        )
        self.moves_left_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, 1),
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, self.config.ffn_dim),
                nn.GELU(),
                nn.Linear(self.config.ffn_dim, 1),
            )
            if self.config.uncertainty_head
            else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.global_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_tokens(
        self,
        board: torch.Tensor,
        halfmove_clock: torch.Tensor | None = None,
        fullmove_number: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if board.ndim != 4 or board.shape[1:] != (BOARD_CHANNELS, 8, 8):
            raise ValueError(f"expected board shape [batch, {BOARD_CHANNELS}, 8, 8]")

        batch_size = board.shape[0]
        device = board.device
        piece_id, color_id = _piece_and_color_ids(board)

        squares = torch.arange(64, device=device).expand(batch_size, 64)
        square_tokens = (
            self.piece_embedding(piece_id)
            + self.color_embedding(color_id)
            + self.square_embedding(squares)
        )

        side_to_move = board[:, 12, 0, 0].round().long().clamp(0, 1)
        castling = board[:, 13:17, 0, 0].to(dtype=board.dtype)
        en_passant = board[:, 17].amax(dim=(1, 2), keepdim=False).unsqueeze(-1)
        halfmove = _normalized_optional_scalar(
            halfmove_clock, batch_size, device, board.dtype, divisor=100.0
        )
        fullmove = _normalized_optional_scalar(
            fullmove_number, batch_size, device, board.dtype, divisor=200.0
        )
        clocks = torch.cat([halfmove, fullmove], dim=-1)

        global_features = (
            self.global_token.expand(batch_size, -1, -1)
            + self.side_to_move_embedding(side_to_move).unsqueeze(1)
            + self.castling_projection(castling).unsqueeze(1)
            + self.en_passant_projection(en_passant).unsqueeze(1)
            + self.clock_projection(clocks).unsqueeze(1)
        )
        return torch.cat([global_features, square_tokens], dim=1)

    def forward(
        self,
        board: torch.Tensor,
        legal_policy_mask: torch.Tensor | None = None,
        halfmove_clock: torch.Tensor | None = None,
        fullmove_number: torch.Tensor | None = None,
    ) -> MoETransformerOutput:
        tokens = self.encode_tokens(board, halfmove_clock, fullmove_number)
        encoded, router_outputs = self.encoder(tokens)
        global_state = encoded[:, 0, :]

        policy_logits = self.policy_head(global_state)
        if legal_policy_mask is not None:
            policy_logits = apply_legal_policy_mask(policy_logits, legal_policy_mask)

        uncertainty = (
            torch.nn.functional.softplus(self.uncertainty_head(global_state).squeeze(-1))
            if self.uncertainty_head is not None
            else None
        )
        return MoETransformerOutput(
            policy_logits=policy_logits,
            wdl_logits=self.wdl_head(global_state),
            moves_left=torch.nn.functional.softplus(
                self.moves_left_head(global_state).squeeze(-1)
            ),
            uncertainty=uncertainty,
            router_outputs=router_outputs,
        )


def scalar_value_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(wdl_logits, dim=-1)
    return probabilities[..., 0] - probabilities[..., 2]


def apply_legal_policy_mask(
    policy_logits: torch.Tensor,
    legal_policy_mask: torch.Tensor,
    masked_value: float = -1.0e9,
) -> torch.Tensor:
    if policy_logits.shape != legal_policy_mask.shape:
        raise ValueError("policy logits and legal mask must have identical shape")
    return policy_logits.masked_fill(~legal_policy_mask.bool(), masked_value)


def parameter_count(model: nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def moe_parameter_count(model: MoETransformerEvaluator) -> dict[str, int]:
    moe_params = 0
    dense_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        if "moe" in name or "experts" in name:
            moe_params += param.numel()
        elif "encoder" in name and "ffn" in name:
            dense_params += param.numel()
        else:
            other_params += param.numel()

    return {
        "moe_params": moe_params,
        "dense_ffn_params": dense_params,
        "other_params": other_params,
        "total": moe_params + dense_params + other_params,
    }


def _piece_and_color_ids(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    white = board[:, 0:6].amax(dim=1)
    black = board[:, 6:12].amax(dim=1)
    occupied = (white + black) > 0

    white_piece = board[:, 0:6].argmax(dim=1) + 1
    black_piece = board[:, 6:12].argmax(dim=1) + 1
    piece_id = torch.where(white.bool(), white_piece, torch.zeros_like(white_piece))
    piece_id = torch.where(black.bool(), black_piece, piece_id)
    piece_id = torch.where(occupied, piece_id, torch.zeros_like(piece_id))
    color_id = torch.where(white.bool(), torch.ones_like(piece_id), torch.zeros_like(piece_id))
    color_id = torch.where(black.bool(), torch.full_like(piece_id, 2), color_id)

    return piece_id.flatten(1).long(), color_id.flatten(1).long()


def _normalized_optional_scalar(
    value: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    divisor: float,
) -> torch.Tensor:
    if value is None:
        return torch.zeros((batch_size, 1), device=device, dtype=dtype)
    return value.to(device=device, dtype=dtype).reshape(batch_size, 1) / divisor
