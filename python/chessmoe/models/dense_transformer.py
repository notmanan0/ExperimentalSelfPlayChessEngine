from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from chessmoe.models.encoding import BOARD_CHANNELS, NUM_MOVE_BUCKETS


@dataclass(frozen=True)
class DenseTransformerConfig:
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 8
    ffn_dim: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1.0e-5
    uncertainty_head: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class DenseTransformerOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor
    uncertainty: torch.Tensor | None = None


class DenseTransformerEvaluator(nn.Module):
    """Dense encoder-only chess transformer.

    Input board tensor shape: `[batch, 18, 8, 8]`.
    Internal token tensor shape: `[batch, 65, d_model]`, consisting of one
    global state token plus 64 square tokens.
    """

    def __init__(self, config: DenseTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or DenseTransformerConfig()
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ffn_dim,
            dropout=self.config.dropout,
            activation="gelu",
            layer_norm_eps=self.config.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(d_model, eps=self.config.layer_norm_eps),
            enable_nested_tensor=False,
        )

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
    ) -> DenseTransformerOutput:
        tokens = self.encode_tokens(board, halfmove_clock, fullmove_number)
        encoded = self.encoder(tokens)
        global_state = encoded[:, 0, :]

        policy_logits = self.policy_head(global_state)
        if legal_policy_mask is not None:
            policy_logits = apply_legal_policy_mask(policy_logits, legal_policy_mask)

        uncertainty = (
            torch.nn.functional.softplus(self.uncertainty_head(global_state).squeeze(-1))
            if self.uncertainty_head is not None
            else None
        )
        return DenseTransformerOutput(
            policy_logits=policy_logits,
            wdl_logits=self.wdl_head(global_state),
            moves_left=torch.nn.functional.softplus(
                self.moves_left_head(global_state).squeeze(-1)
            ),
            uncertainty=uncertainty,
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
