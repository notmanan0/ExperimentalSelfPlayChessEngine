from pathlib import Path

import pytest
import torch

from chessmoe.export.onnx_export import export_policy_value_onnx
from chessmoe.models.student_hybrid import StudentHybridConfig, StudentHybridEvaluator


OUTPUT_DIR = Path("python-test-output")


def test_export_student_hybrid_onnx_uses_dynamic_batch_and_stable_names():
    torch.manual_seed(123)
    model = StudentHybridEvaluator(
        StudentHybridConfig(
            conv_channels=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            ffn_dim=32,
            dropout=0.0,
        )
    )
    OUTPUT_DIR.mkdir(exist_ok=True)

    result = export_policy_value_onnx(
        model,
        OUTPUT_DIR / "student_hybrid.onnx",
        verify=False,
        dynamic_batch=True,
    )

    assert result.path.name == "student_hybrid.onnx"
    assert result.status in {"exported", "skipped"}
    if result.status == "skipped":
        pytest.skip(result.reason)

    import onnx

    loaded = onnx.load(result.path)
    assert [node.name for node in loaded.graph.input] == ["board"]
    assert [node.name for node in loaded.graph.output] == [
        "policy_logits",
        "wdl_logits",
        "moves_left",
    ]
    batch_dim = loaded.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param == "batch"