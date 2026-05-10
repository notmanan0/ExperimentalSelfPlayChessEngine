# Phase 11: Dense Transformer Evaluator

Checked against current official documentation on 2026-05-10.

Sources:

- PyTorch `TransformerEncoderLayer` documents the standard encoder block, `batch_first`, `norm_first`, `activation`, and feedforward options: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- PyTorch `TransformerEncoder` documents stacked encoder layers and nested-tensor behavior: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder
- PyTorch `MultiheadAttention` documents `batch_first=True` shapes and notes that optimized `scaled_dot_product_attention()` is used when possible: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- PyTorch ONNX documentation identifies `torch.onnx.export(..., dynamo=True)` and `dynamic_shapes` as the modern export path: https://docs.pytorch.org/docs/stable/onnx.html
- NVIDIA Transformer Engine 2.13 describes TE as an acceleration library for transformer models on NVIDIA GPUs with PyTorch support; it is not required for this first dense FP32 baseline: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/getting_started/index.html
- NVIDIA Transformer Engine PyTorch API documents `TransformerLayer`, `LayerNormMLP`, and drop-in linear building blocks for later fused GPU variants: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html
- NVIDIA TensorRT 10.16 operator docs include transformer-relevant dense operators such as `MatrixMultiply`, `SoftMax`, `Normalization`, and `Attention`: https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/index.html

## Objective

Replace the tiny CNN as the next training target with a dense encoder-only transformer evaluator while keeping the tiny CNN available for regression tests and speed comparison. This phase does not add MoE, sparse routing, or Gumbel search.

The dense transformer consumes the existing `[batch, 18, 8, 8]` board tensor, builds 64 square tokens plus one global state token, and returns dense policy logits, WDL logits, moves-left prediction, and an optional uncertainty-head skeleton.

## Files To Create

- `python/chessmoe/models/dense_transformer.py`
- `python/chessmoe/models/factory.py`
- `configs/model/dense_transformer_tiny.yaml`
- `configs/training/dense_transformer_tiny.json`
- `tools/benchmark/benchmark_models.py`
- `tests/python/test_dense_transformer_evaluator.py`
- `tests/python/test_dense_transformer_training.py`
- `docs/phase11_dense_transformer_evaluator.md`

Files updated:

- `python/chessmoe/models/__init__.py`
- `python/chessmoe/training/config.py`
- `python/chessmoe/training/train.py`
- `python/chessmoe/training/checkpoint.py`
- `python/chessmoe/export/onnx_export.py`
- `python/export/export_tiny_onnx.py`

## Core Data Structures

- `DenseTransformerConfig`: configurable model size, layers, heads, FFN size, dropout, norm epsilon, and uncertainty-head toggle.
- `DenseTransformerOutput`: `policy_logits`, `wdl_logits`, `moves_left`, and optional `uncertainty`.
- `DenseTransformerEvaluator`: concrete PyTorch model.
- `build_model`: model factory for `tiny_cnn` and `dense_transformer`.
- `parameter_count`: trainable parameter reporting helper.

## Main Algorithms

1. Convert the 18-plane board tensor into square-level piece and color IDs.
2. Embed piece type, color, and square index for each of 64 square tokens.
3. Build a global token from learned state plus side-to-move, castling rights, en passant presence, halfmove, and fullmove features.
4. Concatenate `[global, square_0, ..., square_63]` into `[batch, 65, d_model]`.
5. Run a dense encoder-only transformer with standard multi-head self-attention and dense FFN layers.
6. Feed the global token output into policy, WDL, moves-left, and optional uncertainty heads.
7. Apply legal policy masking through `apply_legal_policy_mask` when a legal mask is provided.
8. Export only the production outputs: policy logits, WDL logits, and moves-left.

Halfmove and fullmove inputs are supported by the model API and default to zeros until the replay/training tensor format is extended to carry those scalar fields beside the existing 18 board planes.

## Tests

- Forward shape checks for policy, WDL, moves-left, and uncertainty.
- Tokenization check for 64 square tokens plus one global token.
- Legal policy masking.
- Loss compatibility with the existing training loss.
- Save/load round trip.
- ONNX export skeleton skip/export behavior.
- Training config loading for transformer fields.
- Training loop checkpoint creation and load for the dense transformer.

## Completion Criteria

- Dense transformer model is configurable and testable with a tiny config.
- Training can instantiate and checkpoint either the transformer or the tiny CNN.
- Export path accepts the dense transformer and preserves stable output names.
- Legal masking exists at the model helper level and remains compatible with evaluator-side masking.
- Parameter count and tiny-CNN comparison benchmark command exist.
- TensorRT compatibility is documented around standard dense operators and FP32-first export.
- No MoE, sparse routing, or Gumbel search is introduced.

## Common Failure Modes

- Applying legal masking after softmax instead of masking logits first.
- Accidentally dropping the global state features and training only on piece placement.
- Making the model depend on Python objects that cannot export to ONNX.
- Enabling Transformer Engine or FP16 before FP32 parity and speed baselines exist.
- Increasing transformer size before replay volume and arena gating can support it.
- Treating uncertainty output as calibrated before it has a loss and validation target.

## Next Step

Train the tiny dense transformer on replay chunks, compare it against the tiny CNN with `tools/benchmark/benchmark_models.py`, export FP32 ONNX, and run arena gating before making it the active model. MoE remains the next architectural phase after dense transformer quality and inference are measured.
