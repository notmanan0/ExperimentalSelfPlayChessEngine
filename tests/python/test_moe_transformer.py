import torch
import pytest
from pathlib import Path

from chessmoe.models.moe_module import MoEConfig
from chessmoe.models.moe_transformer import (
    MoETransformerConfig,
    MoETransformerEvaluator,
    MoETransformerOutput,
    parameter_count,
    moe_parameter_count,
)
from chessmoe.models.encoding import BOARD_SHAPE, NUM_MOVE_BUCKETS, encode_fen, move_to_index
from chessmoe.training.checkpoint import load_checkpoint, save_checkpoint
from chessmoe.training.losses import TinyLossTargets, compute_moe_aware_loss


STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
OUTPUT_DIR = Path("python-test-output")


def tiny_moe_config() -> MoETransformerConfig:
    return MoETransformerConfig(
        d_model=32,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        uncertainty_head=True,
        moe_layers=(1,),
        moe=MoEConfig(
            num_experts=4,
            top_k_training=2,
            top_k_inference=1,
            capacity_factor=1.25,
            load_balance_coeff=0.01,
            router_entropy_coeff=0.001,
            router_noise_std=0.1,
            dense_fallback=False,
            expert_dropout=0.0,
        ),
    )


def test_moe_transformer_forward_shapes():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    batch = torch.stack([encode_fen(STARTPOS), torch.zeros(BOARD_SHAPE)])

    output = model(batch)

    assert output.policy_logits.shape == (2, NUM_MOVE_BUCKETS)
    assert output.wdl_logits.shape == (2, 3)
    assert output.moves_left.shape == (2,)
    assert output.uncertainty is not None
    assert output.uncertainty.shape == (2,)
    assert len(output.router_outputs) > 0


def test_moe_transformer_router_outputs_count():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    batch = torch.stack([encode_fen(STARTPOS)])

    output = model(batch)

    num_moe_layers = len(config.moe_layers)
    assert len(output.router_outputs) == num_moe_layers


def test_moe_transformer_encode_tokens():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    batch = torch.stack([encode_fen(STARTPOS)])

    tokens = model.encode_tokens(batch)

    assert tokens.shape == (1, 65, config.d_model)


def test_moe_transformer_deterministic_eval():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    model.eval()
    batch = torch.stack([encode_fen(STARTPOS)])

    out1 = model(batch)
    out2 = model(batch)

    torch.testing.assert_close(out1.policy_logits, out2.policy_logits)
    torch.testing.assert_close(out1.wdl_logits, out2.wdl_logits)
    torch.testing.assert_close(out1.moves_left, out2.moves_left)


def test_moe_transformer_scalar_value():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    batch = torch.stack([encode_fen(STARTPOS)])

    from chessmoe.models.moe_transformer import scalar_value_from_wdl
    output = model(batch)
    scalar = scalar_value_from_wdl(output.wdl_logits)

    assert scalar.shape == (1,)
    assert scalar.item() >= -1.0
    assert scalar.item() <= 1.0


def test_moe_transformer_loss_with_auxiliary():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    model.train()
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

    losses = compute_moe_aware_loss(output, targets)

    assert losses.total.ndim == 0
    assert losses.policy.item() > 0.0
    assert losses.moe_load_balance.item() >= 0.0
    assert losses.moe_router_entropy.item() >= 0.0


def test_moe_transformer_parameter_count():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)

    total = parameter_count(model)
    assert total > 0

    breakdown = moe_parameter_count(model)
    assert breakdown["total"] == total
    assert breakdown["moe_params"] > 0


def test_moe_transformer_save_load_roundtrip():
    torch.manual_seed(42)
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    model.eval()
    batch = torch.stack([encode_fen(STARTPOS)])
    before = model(batch)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "moe_transformer.pt"

    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    after = loaded(batch)

    assert isinstance(loaded, MoETransformerEvaluator)
    torch.testing.assert_close(before.policy_logits, after.policy_logits)
    torch.testing.assert_close(before.wdl_logits, after.wdl_logits)
    torch.testing.assert_close(before.moves_left, after.moves_left)


def test_moe_transformer_dense_fallback():
    config = tiny_moe_config()
    config_dense = MoETransformerConfig(
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
        uncertainty_head=config.uncertainty_head,
        moe_layers=config.moe_layers,
        moe=config.moe,
        dense_fallback_config=True,
    )
    model = MoETransformerEvaluator(config_dense)
    model.eval()
    batch = torch.stack([encode_fen(STARTPOS), encode_fen(STARTPOS)])

    output = model(batch)

    assert output.policy_logits.shape == (2, NUM_MOVE_BUCKETS)
    assert len(output.router_outputs) == 0


def test_moe_transformer_expert_usage_metrics():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    model.train()
    batch = torch.stack([encode_fen(STARTPOS)] * 4)

    output = model(batch)

    for ro in output.router_outputs:
        assert ro.expert_usage.shape == (config.moe.num_experts,)
        assert torch.all(ro.expert_usage >= 0.0)
        assert ro.num_dropped_tokens.item() >= 0.0


def test_moe_transformer_factory_build():
    from chessmoe.models.factory import build_model

    config = tiny_moe_config()
    model = build_model("moe_transformer", moe_transformer_config=config)
    assert isinstance(model, MoETransformerEvaluator)


def test_moe_transformer_factory_kind():
    from chessmoe.models.factory import model_kind, build_model

    config = tiny_moe_config()
    model = build_model("moe_transformer", moe_transformer_config=config)
    assert model_kind(model) == "moe_transformer"


def test_moe_transformer_legal_policy_mask():
    from chessmoe.models.moe_transformer import apply_legal_policy_mask

    logits = torch.zeros((1, NUM_MOVE_BUCKETS), dtype=torch.float32)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, move_to_index("e2e4")] = True

    masked = apply_legal_policy_mask(logits, legal_mask)

    assert masked[0, move_to_index("e2e4")].item() == 0.0
    assert masked[0, move_to_index("e2e5")].item() < -1.0e8


def test_moe_transformer_config_to_dict():
    config = tiny_moe_config()
    d = config.to_dict()

    assert d["d_model"] == 32
    assert d["num_layers"] == 2
    assert d["moe_layers"] == [1]
    assert isinstance(d["moe"], dict)
    assert d["moe"]["num_experts"] == 4


def test_moe_transformer_checkpoint_config_roundtrip():
    config = tiny_moe_config()
    model = MoETransformerEvaluator(config)
    model.eval()
    batch = torch.stack([encode_fen(STARTPOS)])
    before = model(batch)

    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "moe_transformer_config.pt"

    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    after = loaded(batch)

    assert isinstance(loaded, MoETransformerEvaluator)
    assert loaded.config.moe.num_experts == config.moe.num_experts
    assert loaded.config.moe_layers == config.moe_layers
    torch.testing.assert_close(before.policy_logits, after.policy_logits)
