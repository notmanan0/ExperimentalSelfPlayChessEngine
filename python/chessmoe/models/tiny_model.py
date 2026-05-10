from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chessmoe.models.encoding import BOARD_CHANNELS, NUM_MOVE_BUCKETS


@dataclass(frozen=True)
class TinyModelOutput:
    policy_logits: torch.Tensor
    wdl_logits: torch.Tensor
    moves_left: torch.Tensor


class TinyChessNet(nn.Module):
    """Tiny baseline CNN.

    Input: `x` shaped `[batch, 18, 8, 8]`.
    Outputs:
    - policy logits: `[batch, NUM_MOVE_BUCKETS]`
    - WDL logits: `[batch, 3]` ordered win/draw/loss
    - moves-left estimate: `[batch]`
    """

    def __init__(self, channels: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, hidden),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(hidden, NUM_MOVE_BUCKETS)
        self.wdl_head = nn.Linear(hidden, 3)
        self.moves_left_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> TinyModelOutput:
        if x.ndim != 4 or x.shape[1:] != (BOARD_CHANNELS, 8, 8):
            raise ValueError(f"expected input shape [batch, {BOARD_CHANNELS}, 8, 8]")
        features = self.trunk(x)
        return TinyModelOutput(
            policy_logits=self.policy_head(features),
            wdl_logits=self.wdl_head(features),
            moves_left=self.moves_left_head(features).squeeze(-1),
        )


def scalar_value_from_wdl(wdl_logits: torch.Tensor) -> torch.Tensor:
    """Convert WDL logits to scalar value in [-1, 1] as P(win) - P(loss)."""
    probabilities = torch.softmax(wdl_logits, dim=-1)
    return probabilities[..., 0] - probabilities[..., 2]

