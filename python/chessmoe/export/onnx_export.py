from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import torch

from chessmoe.models.encoding import BOARD_SHAPE
from torch import nn


@dataclass(frozen=True)
class ExportResult:
    path: Path
    status: str
    reason: str = ""


class _OnnxPolicyValueWrapper(torch.nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(board)
        return output.policy_logits, output.wdl_logits, output.moves_left


def export_policy_value_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    verify: bool = False,
    dynamic_batch: bool = True,
    opset_version: int | None = None,
) -> ExportResult:
    """Export a policy/value network to ONNX with stable tensor names."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_batch = 2 if dynamic_batch else 1
    sample = torch.zeros((sample_batch, *BOARD_SHAPE), dtype=torch.float32)

    dynamic_axes = {
        "board": {0: "batch"},
        "policy_logits": {0: "batch"},
        "wdl_logits": {0: "batch"},
        "moves_left": {0: "batch"},
    } if dynamic_batch else None

    # Try dynamo exporter first
    try:
        wrapper = _OnnxPolicyValueWrapper(model).eval()
        export_kwargs: dict[str, object] = {
            "input_names": ["board"],
            "output_names": ["policy_logits", "wdl_logits", "moves_left"],
            "dynamo": True,
            "verify": verify,
        }
        if dynamic_batch:
            export_kwargs["dynamic_shapes"] = {"board": {0: "batch"}}
        if opset_version is not None:
            export_kwargs["opset_version"] = opset_version
        if "fallback" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["fallback"] = True

        torch.onnx.export(wrapper, (sample,), output_path, **export_kwargs)
        if output_path.exists():
            return ExportResult(path=output_path, status="exported")
    except Exception:
        pass

    # Fallback to legacy TorchScript exporter
    try:
        wrapper = _OnnxPolicyValueWrapper(model).eval()
        legacy_kwargs: dict[str, object] = {
            "input_names": ["board"],
            "output_names": ["policy_logits", "wdl_logits", "moves_left"],
            "opset_version": opset_version or 17,
        }
        if dynamic_batch:
            legacy_kwargs["dynamic_axes"] = {
                "board": {0: "batch"},
                "policy_logits": {0: "batch"},
                "wdl_logits": {0: "batch"},
                "moves_left": {0: "batch"},
            }

        torch.onnx.export(wrapper, (sample,), output_path, **legacy_kwargs)
        if output_path.exists():
            return ExportResult(path=output_path, status="exported")
    except Exception as exc:
        return ExportResult(path=output_path, status="skipped", reason=str(exc))

    return ExportResult(path=output_path, status="skipped", reason="file not created")


def export_tiny_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    verify: bool = False,
    dynamic_batch: bool = True,
    opset_version: int | None = None,
) -> ExportResult:
    """Export the tiny baseline network to ONNX with stable tensor names."""
    return export_policy_value_onnx(
        model,
        path,
        verify=verify,
        dynamic_batch=dynamic_batch,
        opset_version=opset_version,
    )


def export_onnx_skeleton(
    model: TinyChessNet,
    path: str | Path,
    *,
    verify: bool = False,
) -> ExportResult:
    return export_tiny_onnx(model, path, verify=verify, dynamic_batch=True)
