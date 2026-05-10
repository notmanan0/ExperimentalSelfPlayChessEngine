# Phase 15: Sparse MoE Transformer Evaluator

## Objective

Add a sparse Mixture-of-Experts (MoE) transformer evaluator to the chessmoe project. The MoE model replaces selected FFN blocks in the dense transformer with sparse MoE FFN layers. The model must prove itself through arena gating before replacing the dense model.

## Architecture

### MoE Layer Design

- **Router**: Linear gate projecting `d_model` to `num_experts`. Produces router logits.
- **Top-k routing**: Top-2 during training, Top-1 during inference.
- **Expert FFN**: Standard 2-layer FFN with GELU activation (same shape as dense FFN).
- **Capacity factor**: Limits tokens per expert. Tokens beyond capacity are dropped.
- **Load-balancing loss**: Encourages uniform expert usage. Coefficient: 0.01.
- **Router entropy loss**: Encourages diverse routing. Coefficient: 0.001.
- **Dense fallback**: Config flag to replace all MoE layers with dense FFN (for export/testing).

### Model Structure

```
Input tokens (65 tokens: 1 global + 64 squares)
  -> Layer 0: Self-Attention + Dense FFN
  -> Layer 1: Self-Attention + Sparse MoE FFN (if in moe_layers)
  -> Layer 2: Self-Attention + Dense FFN
  -> Layer 3: Self-Attention + Sparse MoE FFN (if in moe_layers)
  -> Final LayerNorm
  -> Policy/WDL/Moves-Left heads
```

## Files Created

### Core Implementation
- `python/chessmoe/models/moe_module.py` - MoEConfig, MoERouter, ExpertFFN, SparseMoEFFN, DenseFFNFallback
- `python/chessmoe/models/moe_transformer.py` - MoETransformerConfig, MoETransformerEvaluator, MoETransformerLayer, MoETransformerEncoder

### Training Integration
- `python/chessmoe/training/moe_losses.py` - compute_moe_auxiliary_loss, moe_loss_from_model_output
- Updated `python/chessmoe/training/losses.py` - compute_moe_aware_loss
- Updated `python/chessmoe/training/train.py` - MoE model build and loss integration
- Updated `python/chessmoe/training/config.py` - MoE config fields
- Updated `python/chessmoe/training/checkpoint.py` - MoE save/load

### Export
- `python/chessmoe/export/moe_export.py` - ONNX export with TensorRT compatibility notes

### Config
- `configs/model/moe_transformer_tiny.yaml` - Tiny MoE config for testing

### Tests
- `tests/python/test_moe_module.py` - Router shapes, load-balance loss, deterministic eval, save/load
- `tests/python/test_moe_transformer.py` - Evaluator shapes, loss integration, checkpoint roundtrip

### Benchmark
- `tools/benchmark/moe_inference_benchmark.py` - MoE vs Dense latency comparison

## Core Data Structures

### MoEConfig
```python
@dataclass(frozen=True)
class MoEConfig:
    num_experts: int = 8
    top_k_training: int = 2
    top_k_inference: int = 1
    capacity_factor: float = 1.25
    load_balance_coeff: float = 0.01
    router_entropy_coeff: float = 0.001
    router_noise_std: float = 0.1
    dense_fallback: bool = False
    expert_dropout: float = 0.0
```

### MoERouterOutput
```python
class MoERouterOutput(NamedTuple):
    dispatch_weights: torch.Tensor   # [num_tokens, top_k]
    expert_indices: torch.Tensor     # [num_tokens, top_k]
    router_logits: torch.Tensor      # [num_tokens, num_experts]
    load_balance_loss: torch.Tensor  # scalar
    router_entropy_loss: torch.Tensor  # scalar
    num_dropped_tokens: torch.Tensor  # scalar
    expert_usage: torch.Tensor       # [num_experts]
```

## Main Algorithms

### Router
1. Project tokens to expert logits via linear gate.
2. Add Gaussian noise during training (noise_std=0.1).
3. Apply softmax to get routing probabilities.
4. Select top-k experts per token.
5. Compute load-balance loss: sum(fraction_of_tokens * avg_prob) * num_experts.
6. Compute router entropy loss: clamp(target_entropy - actual_entropy, min=0).

### Sparse MoE FFN
1. Apply layer norm to input.
2. Route tokens to experts via router.
3. For each expert, gather assigned tokens, apply expert FFN, scatter results.
4. Weight expert outputs by dispatch weights.
5. Add residual connection.

### Training Integration
- MoE auxiliary loss = load_balance_coeff * load_balance_loss + router_entropy_coeff * router_entropy_loss.
- Total loss = policy_loss + wdl_loss + moves_left_weight * moves_left_loss + moe_auxiliary_loss.

## Tests

### test_moe_module.py
- Router output shapes (training and inference)
- Top-1 inference produces valid weights
- Deterministic eval routing
- Load-balance loss is non-negative
- Router entropy loss is non-negative
- Expert FFN shapes
- SparseMoEFFN shapes and residual connection
- DenseFFN fallback
- Expert usage sums to reasonable range
- No noise in eval mode
- Save/load roundtrip

### test_moe_transformer.py
- Forward shapes
- Router outputs count matches moe_layers
- Token encoding
- Deterministic eval
- Scalar value from WDL
- Loss with auxiliary components
- Parameter count breakdown
- Save/load roundtrip
- Dense fallback mode
- Expert usage metrics
- Factory build and kind detection
- Legal policy mask
- Config serialization
- Checkpoint config roundtrip

## Completion Criteria

1. All tests in `test_moe_module.py` pass.
2. All tests in `test_moe_transformer.py` pass.
3. MoE model trains without NaN/Inf.
4. Dense fallback produces identical architecture to dense transformer.
5. Save/load roundtrip preserves model outputs exactly.
6. Benchmark script runs and reports MoE overhead.
7. ONNX export succeeds (or gracefully skips with TRT notes).

## Common Failure Modes

1. **NaN in router logits**: Caused by large input magnitudes. Mitigate with proper initialization and layer norm before routing.
2. **All tokens routed to one expert**: Load-balance loss prevents this. Monitor expert_usage metrics.
3. **Dropped tokens at inference**: Set capacity_factor >= 1.0 for inference. Dropped tokens get zero contribution.
4. **Gradient explosion from sparse routing**: Clip gradients. The dispatch_weights are normalized.
5. **ONNX export failure**: MoE dynamic scatter/gather may not export. Use dense_fallback for export.
6. **Memory explosion with many experts**: Each expert has full FFN parameters. Scale ffn_dim down when adding experts.

## TensorRT Compatibility

- Top-k routing and softmax are standard ONNX ops.
- Dynamic scatter/gather in MoE FFN creates data-dependent control flow.
- For best TRT compatibility, export with `dense_fallback=True`.
- FP8 routing logits may cause unstable top-k selection; keep routing in FP32.
- No native expert parallelism in TensorRT.

## Next Step

Phase 16: Distributed multi-GPU training with expert parallelism (NCCL). Requires:
- Expert parallel across GPUs.
- All-to-all communication for token routing.
- Gradient synchronization.
