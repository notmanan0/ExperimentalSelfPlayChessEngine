from pathlib import Path

import pytest
import torch

from chessmoe.export.onnx_export import export_tiny_onnx
from chessmoe.models.encoding import BOARD_SHAPE, NUM_MOVE_BUCKETS, encode_fen, move_to_index
from chessmoe.models.tiny_model import TinyChessNet


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUTPUT_DIR = Path("python-test-output")


def test_export_tiny_onnx_uses_dynamic_batch_and_stable_names():
    torch.manual_seed(123)
    model = TinyChessNet()
    OUTPUT_DIR.mkdir(exist_ok=True)

    result = export_tiny_onnx(
        model,
        OUTPUT_DIR / "phase10_tiny.onnx",
        verify=False,
        dynamic_batch=True,
    )

    assert result.path.name == "phase10_tiny.onnx"
    assert result.status in {"exported", "skipped"}
    if result.status == "skipped":
        assert result.reason
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


def test_exported_onnx_matches_pytorch_when_runtime_is_available():
    torch.manual_seed(99)
    model = TinyChessNet().eval()
    OUTPUT_DIR.mkdir(exist_ok=True)
    onnx_path = OUTPUT_DIR / "phase10_parity.onnx"

    result = export_tiny_onnx(model, onnx_path, verify=False, dynamic_batch=True)
    if result.status == "skipped":
        pytest.skip(result.reason)

    ort = pytest.importorskip("onnxruntime")
    batch = torch.stack([encode_fen(STARTPOS), torch.zeros(BOARD_SHAPE)])
    with torch.no_grad():
        expected = model(batch)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual_policy, actual_wdl, actual_moves_left = session.run(
        None, {"board": batch.numpy()}
    )

    torch.testing.assert_close(torch.from_numpy(actual_policy), expected.policy_logits)
    torch.testing.assert_close(torch.from_numpy(actual_wdl), expected.wdl_logits)
    torch.testing.assert_close(torch.from_numpy(actual_moves_left), expected.moves_left)


def test_policy_index_contract_for_legal_masking():
    assert move_to_index("e2e4") == 796
    assert move_to_index("e7e8q") == 7484
    assert move_to_index("e7e8n") == 19772
    assert NUM_MOVE_BUCKETS == 20480
