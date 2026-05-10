# Phase 8 Neural Training Loop

## Objective

Train the first TinyChessNet policy/WDL/moves-left baseline from binary replay chunks. This phase reads `.cmrep` chunks through the SQLite replay index and does not implement arena gating.

## Files To Create

- `python/chessmoe/training/config.py`
- `python/chessmoe/training/data.py`
- `python/chessmoe/training/train.py`
- `configs/training/tiny_replay.json`
- `tests/python/test_training_loop.py`

## Core Data Structures

- `TrainingConfig`: paths, optimizer settings, split settings, reproducibility seed, AMP, compile, and model dimensions.
- `ReplayDataset`: map-style PyTorch dataset backed by replay chunks listed in SQLite.
- `TrainingSample`: encoded board tensor, policy target, WDL target, moves-left target.
- `TrainingBatch`: stacked tensors for model input and losses.
- `TrainingResult`: start epoch, completed epoch count, train losses, validation losses.

## Main Algorithms

1. Load chunk paths from the SQLite `chunks` table.
2. Decode replay samples with `ReplayReader`.
3. Encode board state into `[18, 8, 8]` tensors.
4. Build policy targets from MCTS visit counts, falling back to normalized probabilities only when visits are absent.
5. Convert final WDL to side-to-move win/draw/loss targets.
6. Use a simple moves-left target from the last ply observed per game.
7. Split deterministically into train and validation subsets.
8. Train with AdamW, optional AMP, optional `torch.compile`, cosine LR scheduling, gradient clipping, and JSONL metrics.
9. Save resumable checkpoints with model, optimizer, scheduler, scaler, epoch, and config metadata.

## Tests

- Dataset loads samples from a SQLite replay index.
- Batch collation returns expected tensor shapes.
- Loss decreases on a tiny overfit replay dataset.
- Checkpoint resume continues the epoch count and restores optimizer state.

## Commands

```powershell
python -m pytest tests/python/test_training_loop.py --basetemp python-test-output/pytest
python -m pytest tests/python --basetemp python-test-output/pytest
$env:PYTHONPATH='python'
python -m chessmoe.training.train --config configs/training/tiny_replay.json
```

## Completion Criteria

- Replay chunks are trainable through a PyTorch `Dataset` and `DataLoader`.
- Policy, WDL, and moves-left targets are produced from decoded samples.
- Training writes checkpoints and metrics.
- Resume starts from the saved epoch.
- The tiny overfit test proves the loop can optimize on replay data.

## Common Failure Modes

- Using white-perspective WDL labels when the model expects side-to-move labels.
- Building policy targets from probabilities without normalizing when visit counts are missing.
- Changing replay binary layout without updating the Python decoder.
- Resuming only model weights and losing optimizer or scheduler state.
- Enabling AMP on unsupported devices without a guarded autocast path.

## Next Step

Phase 9 should add arena gating. A candidate model should only be promoted after reproducible match results and a documented promotion decision.
