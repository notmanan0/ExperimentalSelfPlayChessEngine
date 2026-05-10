import torch
import pytest

from chessmoe.models.moe_module import (
    DenseFFNFallback,
    ExpertFFN,
    MoEConfig,
    MoERouter,
    SparseMoEFFN,
)


def _default_config() -> MoEConfig:
    return MoEConfig(
        num_experts=4,
        top_k_training=2,
        top_k_inference=1,
        capacity_factor=1.25,
        load_balance_coeff=0.01,
        router_entropy_coeff=0.001,
        router_noise_std=0.1,
        dense_fallback=False,
        expert_dropout=0.0,
    )


def test_router_output_shapes_training():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.train()
    x = torch.randn(2, 65, d_model)

    out = router(x)

    num_tokens = 2 * 65
    assert out.dispatch_weights.shape == (num_tokens, config.top_k_training)
    assert out.expert_indices.shape == (num_tokens, config.top_k_training)
    assert out.router_logits.shape == (num_tokens, config.num_experts)
    assert out.load_balance_loss.ndim == 0
    assert out.router_entropy_loss.ndim == 0
    assert out.num_dropped_tokens.ndim == 0
    assert out.expert_usage.shape == (config.num_experts,)


def test_router_output_shapes_inference():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.eval()
    x = torch.randn(2, 65, d_model)

    out = router(x)

    num_tokens = 2 * 65
    assert out.dispatch_weights.shape == (num_tokens, config.top_k_inference)
    assert out.expert_indices.shape == (num_tokens, config.top_k_inference)


def test_router_top1_inference():
    config = MoEConfig(num_experts=4, top_k_training=2, top_k_inference=1)
    d_model = 16
    router = MoERouter(config, d_model)
    router.eval()
    x = torch.randn(1, 10, d_model)

    out = router(x)

    assert out.dispatch_weights.shape == (10, 1)
    assert out.expert_indices.shape == (10, 1)
    assert torch.all(out.dispatch_weights >= 0)
    assert torch.all(out.dispatch_weights <= 1)


def test_router_deterministic_eval():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.eval()
    x = torch.randn(2, 65, d_model)

    out1 = router(x)
    out2 = router(x)

    torch.testing.assert_close(out1.dispatch_weights, out2.dispatch_weights)
    torch.testing.assert_close(out1.expert_indices, out2.expert_indices)
    torch.testing.assert_close(out1.router_logits, out2.router_logits)


def test_router_load_balance_loss_nonnegative():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.train()
    x = torch.randn(4, 65, d_model)

    out = router(x)

    assert out.load_balance_loss.item() >= 0.0


def test_router_entropy_loss_nonnegative():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.train()
    x = torch.randn(4, 65, d_model)

    out = router(x)

    assert out.router_entropy_loss.item() >= 0.0


def test_expert_ffn_shapes():
    d_model = 32
    ffn_dim = 64
    ffn = ExpertFFN(d_model, ffn_dim)
    x = torch.randn(10, d_model)
    out = ffn(x)
    assert out.shape == (10, d_model)


def test_sparse_moe_ffn_shapes():
    config = _default_config()
    d_model = 32
    ffn_dim = 64
    moe_ffn = SparseMoEFFN(config, d_model, ffn_dim)
    x = torch.randn(2, 65, d_model)

    out, router_out = moe_ffn(x)

    assert out.shape == (2, 65, d_model)
    assert router_out is not None
    assert router_out.load_balance_loss.ndim == 0


def test_sparse_moe_ffn_residual_connection():
    config = _default_config()
    d_model = 32
    ffn_dim = 64
    moe_ffn = SparseMoEFFN(config, d_model, ffn_dim)
    moe_ffn.eval()
    x = torch.randn(2, 65, d_model)

    out, _ = moe_ffn(x)

    assert out.shape == x.shape


def test_dense_ffn_fallback():
    d_model = 32
    ffn_dim = 64
    ffn = DenseFFNFallback(d_model, ffn_dim)
    x = torch.randn(2, 65, d_model)

    out, router_out = ffn(x)

    assert out.shape == x.shape
    assert router_out is None


def test_expert_usage_sums_to_reasonable_range():
    config = _default_config()
    d_model = 32
    router = MoERouter(config, d_model)
    router.train()
    x = torch.randn(4, 65, d_model)

    out = router(x)

    assert out.expert_usage.sum().item() > 0.0
    assert out.expert_usage.sum().item() <= config.top_k_training + 0.1


def test_router_no_noise_in_eval():
    config = MoEConfig(num_experts=4, top_k_training=2, top_k_inference=1, router_noise_std=0.5)
    d_model = 16
    router = MoERouter(config, d_model)
    router.eval()
    x = torch.randn(1, 10, d_model)

    out1 = router(x)
    out2 = router(x)

    torch.testing.assert_close(out1.router_logits, out2.router_logits)


def test_sparse_moe_ffn_save_load_roundtrip(tmp_path):
    config = _default_config()
    d_model = 32
    ffn_dim = 64
    moe_ffn = SparseMoEFFN(config, d_model, ffn_dim)
    moe_ffn.eval()
    x = torch.randn(2, 65, d_model)
    before, _ = moe_ffn(x)

    path = tmp_path / "moe_ffn.pt"
    torch.save(moe_ffn.state_dict(), path)

    loaded = SparseMoEFFN(config, d_model, ffn_dim)
    loaded.load_state_dict(torch.load(path, weights_only=True))
    loaded.eval()
    after, _ = loaded(x)

    torch.testing.assert_close(before, after)
