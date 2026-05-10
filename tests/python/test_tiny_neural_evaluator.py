import torch
from pathlib import Path

from chessmoe.export.onnx_export import export_onnx_skeleton
from chessmoe.models.encoding import (
    BOARD_SHAPE,
    NUM_MOVE_BUCKETS,
    encode_fen,
    move_to_index,
)
from chessmoe.models.tiny_model import TinyChessNet, scalar_value_from_wdl
from chessmoe.training.checkpoint import load_checkpoint, save_checkpoint
from chessmoe.training.losses import TinyLossTargets, compute_tiny_loss


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TEST_OUTPUT_DIR = Path("python-test-output")


def test_board_encoding_shape_and_side_plane():
    encoded = encode_fen(STARTPOS)

    assert encoded.shape == BOARD_SHAPE
    assert encoded.dtype == torch.float32
    assert encoded[0].sum().item() == 8
    assert encoded[11].sum().item() == 1
    assert encoded[12].sum().item() == 64


def test_move_indexing_is_stable_and_distinguishes_promotions():
    e2e4 = move_to_index("e2e4")
    e7e8q = move_to_index("e7e8q")
    e7e8n = move_to_index("e7e8n")

    assert 0 <= e2e4 < NUM_MOVE_BUCKETS
    assert 0 <= e7e8q < NUM_MOVE_BUCKETS
    assert 0 <= e7e8n < NUM_MOVE_BUCKETS
    assert e7e8q != e7e8n
    assert e2e4 == move_to_index("e2e4")


def test_tiny_model_output_shapes():
    model = TinyChessNet()
    batch = torch.stack([encode_fen(STARTPOS), encode_fen(STARTPOS)])

    output = model(batch)

    assert output.policy_logits.shape == (2, NUM_MOVE_BUCKETS)
    assert output.wdl_logits.shape == (2, 3)
    assert output.moves_left.shape == (2,)

    scalar = scalar_value_from_wdl(output.wdl_logits)
    assert scalar.shape == (2,)
    assert torch.all(scalar <= 1.0)
    assert torch.all(scalar >= -1.0)


def test_loss_computation_accepts_visit_distribution_and_wdl_targets():
    model = TinyChessNet()
    batch = torch.stack([encode_fen(STARTPOS), encode_fen(STARTPOS)])
    output = model(batch)

    policy_target = torch.zeros((2, NUM_MOVE_BUCKETS), dtype=torch.float32)
    policy_target[0, move_to_index("e2e4")] = 0.75
    policy_target[0, move_to_index("d2d4")] = 0.25
    policy_target[1, move_to_index("g1f3")] = 1.0
    wdl_target = torch.tensor([0, 2], dtype=torch.long)
    moves_left_target = torch.tensor([40.0, 30.0], dtype=torch.float32)

    loss = compute_tiny_loss(
        output,
        TinyLossTargets(
            policy=policy_target,
            wdl=wdl_target,
            moves_left=moves_left_target,
        ),
    )

    assert loss.total.ndim == 0
    assert loss.policy.item() > 0
    assert loss.wdl.item() > 0
    assert loss.moves_left.item() >= 0


def test_model_save_and_load_round_trip():
    torch.manual_seed(7)
    model = TinyChessNet()
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    path = TEST_OUTPUT_DIR / "tiny.pt"
    batch = torch.stack([encode_fen(STARTPOS)])
    before = model(batch)

    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    after = loaded(batch)

    assert torch.allclose(before.policy_logits, after.policy_logits)
    assert torch.allclose(before.wdl_logits, after.wdl_logits)
    assert torch.allclose(before.moves_left, after.moves_left)


def test_export_skeleton_can_skip_or_export():
    model = TinyChessNet()
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = TEST_OUTPUT_DIR / "tiny.onnx"

    result = export_onnx_skeleton(model, output_path, verify=False)

    assert result.path == output_path
    assert result.status in {"exported", "skipped"}
    if result.status == "exported":
        assert output_path.exists()
    else:
        assert result.reason
