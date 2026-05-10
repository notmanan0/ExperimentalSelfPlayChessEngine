from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from chessmoe.models.encoding import BOARD_CHANNELS, NUM_MOVE_BUCKETS


@dataclass(frozen=True)
class StudentHybridConfig:
    conv_channels: int = 32
    d_model: int = 96
    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 192
    dropout: float = 0.1
    layer_norm_eps: float = 1.0e-5

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class StudentHybridOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class StudentHybridEvaluator(nn.Module):
    """Compact CNN + transformer hybrid for fast inference.

    Input board tensor shape: `[batch, 18, 8, 8]`.
    A lightweight CNN stem builds per-square embeddings, then a small
    transformer encoder aggregates a global token for policy/value heads.
    """

    def __init__(self, config: StudentHybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or StudentHybridConfig()
        if self.config.d_model % self.config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.stem = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, self.config.conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.config.conv_channels,
                self.config.conv_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )
        if self.config.conv_channels == self.config.d_model:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(self.config.conv_channels, self.config.d_model)

        self.square_embedding = nn.Embedding(64, self.config.d_model)
        self.global_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
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
            norm=nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps),
            enable_nested_tensor=False,
        )

        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, NUM_MOVE_BUCKETS),
        )
        self.wdl_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, 3),
        )
        self.moves_left_head = nn.Sequential(
            nn.LayerNorm(self.config.d_model),
            nn.Linear(self.config.d_model, self.config.ffn_dim),
            nn.GELU(),
            nn.Linear(self.config.ffn_dim, 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.global_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, board: torch.Tensor) -> StudentHybridOutput:
        if board.ndim != 4 or board.shape[1:] != (BOARD_CHANNELS, 8, 8):
            raise ValueError(f"expected input shape [batch, {BOARD_CHANNELS}, 8, 8]")

        batch_size = board.shape[0]
        device = board.device
        features = self.stem(board)
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)

        squares = torch.arange(64, device=device).expand(batch_size, 64)
        tokens = tokens + self.square_embedding(squares)

        global_token = self.global_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([global_token, tokens], dim=1))
        global_state = encoded[:, 0, :]

        return StudentHybridOutput(
            policy_logits=self.policy_head(global_state),
            wdl_logits=self.wdl_head(global_state),
            moves_left=torch.nn.functional.softplus(
                self.moves_left_head(global_state).squeeze(-1)
            ),
        )


def scalar_value_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(wdl_logits, dim=-1)
    return probabilities[..., 0] - probabilities[..., 2]
