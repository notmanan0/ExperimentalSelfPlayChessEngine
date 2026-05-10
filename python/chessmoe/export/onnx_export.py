from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from chessmoe.models.encoding import BOARD_SHAPE
from chessmoe.models.tiny_model import TinyChessNet


@dataclass(frozen=True)
class ExportResult:
    path: Path
    status: str
    reason: str = ""


class _OnnxTinyWrapper(torch.nn.Module):
    def __init__(self, model: TinyChessNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(board)
        return output.policy_logits, output.wdl_logits, output.moves_left


def export_onnx_skeleton(
    model: TinyChessNet,
    path: str | Path,
    *,
    verify: bool = False,
) -> ExportResult:
    """Export a tiny model to ONNX when optional exporter dependencies exist.

    This is intentionally a skeleton: it fixes input/output names and uses the
    modern `torch.onnx.export(..., dynamo=True)` path, but treats missing ONNX
    exporter dependencies as a skip rather than a hard failure for Phase 5.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _OnnxTinyWrapper(model).eval()
    sample = torch.zeros((1, *BOARD_SHAPE), dtype=torch.float32)

    try:
        torch.onnx.export(
            wrapper,
            (sample,),
            output_path,
            input_names=["board"],
            output_names=["policy_logits", "wdl_logits", "moves_left"],
            dynamo=True,
            verify=verify,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return ExportResult(path=output_path, status="skipped", reason=str(exc))

    return ExportResult(path=output_path, status="exported")

