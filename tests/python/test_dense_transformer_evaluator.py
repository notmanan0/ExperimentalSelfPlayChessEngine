from pathlib import Path

import torch

from chessmoe.export.onnx_export import export_onnx_skeleton
from chessmoe.models.dense_transformer import (
    DenseTransformerConfig,
    DenseTransformerEvaluator,
    apply_legal_policy_mask,
    parameter_count,
    scalar_value_from_wdl,
)
from chessmoe.models.encoding import BOARD_SHAPE, NUM_MOVE_BUCKETS, encode_fen, move_to_index
from chessmoe.training.checkpoint import load_checkpoint, save_checkpoint
from chessmoe.training.losses import TinyLossTargets, compute_tiny_loss


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUTPUT_DIR = Path("python-test-output")


def tiny_config() -> DenseTransformerConfig:
    return DenseTransformerConfig(
        d_model=32,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        uncertainty_head=True,
    )


def test_dense_transformer_forward_shapes_and_scalar_value():
    model = DenseTransformerEvaluator(tiny_config())
    batch = torch.stack([encode_fen(STARTPOS), torch.zeros(BOARD_SHAPE)])

    output = model(batch)

    assert output.policy_logits.shape == (2, NUM_MOVE_BUCKETS)
    assert output.wdl_logits.shape == (2, 3)
    assert output.moves_left.shape == (2,)
    assert output.uncertainty is not None
    assert output.uncertainty.shape == (2,)

    scalar = scalar_value_from_wdl(output.wdl_logits)
    assert scalar.shape == (2,)
    assert torch.all(scalar <= 1.0)
    assert torch.all(scalar >= -1.0)


def test_dense_transformer_uses_64_square_tokens_and_global_token():
    model = DenseTransformerEvaluator(tiny_config())
    batch = torch.stack([encode_fen(STARTPOS)])

    tokens = model.encode_tokens(batch)

    assert tokens.shape == (1, 65, tiny_config().d_model)


def test_legal_policy_mask_sets_illegal_logits_to_large_negative_value():
    logits = torch.zeros((1, NUM_MOVE_BUCKETS), dtype=torch.float32)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, move_to_index("e2e4")] = True

    masked = apply_legal_policy_mask(logits, legal_mask)

    assert masked[0, move_to_index("e2e4")].item() == 0.0
    assert masked[0, move_to_index("e2e5")].item() < -1.0e8


def test_dense_transformer_loss_and_parameter_count():
    model = DenseTransformerEvaluator(tiny_config())
    batch = torch.stack([encode_fen(STARTPOS), encode_fen(STARTPOS)])
    output = model(batch)

    policy_target = torch.zeros((2, NUM_MOVE_BUCKETS), dtype=torch.float32)
    policy_target[0, move_to_index("e2e4")] = 1.0
    policy_target[1, move_to_index("g1f3")] = 1.0
    targets = TinyLossTargets(
        policy=policy_target,
        wdl=torch.tensor([0, 2], dtype=torch.long),
        moves_left=torch.tensor([40.0, 30.0], dtype=torch.float32),
    )

    losses = compute_tiny_loss(output, targets)

    assert losses.total.ndim == 0
    assert losses.policy.item() > 0.0
    assert parameter_count(model) > 0


def test_dense_transformer_save_and_load_round_trip():
    torch.manual_seed(11)
    model = DenseTransformerEvaluator(tiny_config())
    batch = torch.stack([encode_fen(STARTPOS)])
    before = model(batch)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "dense_transformer.pt"

    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    after = loaded(batch)

    assert isinstance(loaded, DenseTransformerEvaluator)
    torch.testing.assert_close(before.policy_logits, after.policy_logits)
    torch.testing.assert_close(before.wdl_logits, after.wdl_logits)
    torch.testing.assert_close(before.moves_left, after.moves_left)


def test_dense_transformer_export_skeleton_can_skip_or_export():
    model = DenseTransformerEvaluator(tiny_config())
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "dense_transformer.onnx"

    result = export_onnx_skeleton(model, output_path, verify=False)

    assert result.path == output_path
    assert result.status in {"exported", "skipped"}
    if result.status == "exported":
        assert output_path.exists()
    else:
        assert result.reason
