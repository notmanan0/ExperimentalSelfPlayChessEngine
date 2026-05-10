from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chessmoe.models.encoding import BOARD_SHAPE
from chessmoe.models.moe_transformer import MoETransformerEvaluator


TENSORRT_COMPATIBILITY_NOTES = """
TensorRT Compatibility Notes for Sparse MoE Transformer:

1. ROUTING OPERATIONS: The top-k routing and softmax operations are standard ONNX ops
   and are supported by TensorRT. However, the dynamic scatter/gather pattern in
   SparseMoEFFN creates data-dependent control flow that TensorRT cannot optimize
   as well as static graphs.

2. EXPORT STRATEGY: For TensorRT deployment, use one of:
   a) Export with dense_fallback_config=True to replace all MoE layers with dense FFN.
      This gives the best TensorRT compatibility but loses the MoE benefit.
   b) Export the full MoE model. TensorRT will execute the routing but may not
      optimize the sparse computation pattern as well as a native MoE runtime.

3. FP8/FP16: MoE routing logits should remain in FP32 for stable top-k selection.
   Expert FFN layers can use FP16 safely.

4. RECOMMENDED: For match-play inference, benchmark both the dense-fallback export
   and the full MoE export. Use the faster one based on measured latency.

5. CAPACITY FACTOR: During inference (top_k=1), ensure capacity_factor >= 1.0 to
   avoid dropping tokens. Set capacity_factor=1.0 for exact routing without drops.

6. EXPERT PARALLELISM: TensorRT does not natively support expert parallelism.
   All experts execute sequentially on the same device.
"""


@dataclass(frozen=True)
class MoEExportResult:
    path: Path
    status: str
    reason: str = ""
    trt_notes: str = ""


class _MoEPolicyValueWrapper(torch.nn.Module):
    def __init__(self, model: nn.Module, *, dense_fallback: bool = False) -> None:
        super().__init__()
        self.model = model
        self.dense_fallback = dense_fallback

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dense_fallback and isinstance(self.model, MoETransformerEvaluator):
            original_training = self.model.training
            self.model.eval()
            output = self.model(board)
            self.model.train(original_training)
        else:
            output = self.model(board)
        return output.policy_logits, output.wdl_logits, output.moves_left


def export_moe_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    verify: bool = False,
    dynamic_batch: bool = True,
    dense_fallback: bool = False,
    opset_version: int | None = None,
) -> MoEExportResult:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    wrapper = _MoEPolicyValueWrapper(model, dense_fallback=dense_fallback).eval()
    sample_batch = 2 if dynamic_batch else 1
    sample = torch.zeros((sample_batch, *BOARD_SHAPE), dtype=torch.float32)

    import inspect

    export_kwargs: dict[str, object] = {
        "input_names": ["board"],
        "output_names": ["policy_logits", "wdl_logits", "moves_left"],
        "dynamo": True,
        "verify": verify,
        "dynamic_axes": {
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "wdl_logits": {0: "batch"},
            "moves_left": {0: "batch"},
        }
        if dynamic_batch
        else None,
    }
    if dynamic_batch:
        export_kwargs["dynamic_shapes"] = {"board": {0: "batch"}}
    if opset_version is not None:
        export_kwargs["opset_version"] = opset_version
    if "fallback" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["fallback"] = True

    try:
        torch.onnx.export(
            wrapper,
            (sample,),
            output_path,
            **export_kwargs,
        )
    except Exception as exc:
        return MoEExportResult(
            path=output_path,
            status="skipped",
            reason=str(exc),
            trt_notes=TENSORRT_COMPATIBILITY_NOTES,
        )

    return MoEExportResult(
        path=output_path,
        status="exported",
        trt_notes=TENSORRT_COMPATIBILITY_NOTES,
    )
