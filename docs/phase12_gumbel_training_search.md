# Phase 12: Gumbel AlphaZero-Style Training Search

Checked against high-quality sources on 2026-05-10.

Sources:

- Danihelka, Guez, Schrittwieser, and Silver, *Policy improvement by planning with Gumbel*, ICLR 2022. The OpenReview abstract says AlphaZero can fail to improve its policy network when not all root actions are visited and proposes sampling actions without replacement for policy improvement: https://openreview.net/forum?id=bERaNdoegnO
- The ICLR 2022 poster page summarizes Gumbel AlphaZero and Gumbel MuZero as replacing heuristic action selection and policy targets with policy-improvement mechanisms, improving few-simulation planning in Go, chess, and Atari: https://iclr.cc/virtual/2022/poster/6418
- Hao et al., *Multiagent Gumbel MuZero*, AAAI 2024, is a later peer-reviewed extension showing the same family of methods can improve low-budget planning in larger action spaces: https://ojs.aaai.org/index.php/AAAI/article/view/29121

## Objective

Add an optional Gumbel AlphaZero-style root search for training self-play while preserving standard PUCT as the default path for UCI and match play. This phase is CPU-only and does not add multithreaded or GPU MCTS.

## Files To Create

- `cpp/search/include/chessmoe/search/search_mode.h`
- `cpp/search/include/chessmoe/search/gumbel_searcher.h`
- `cpp/search/src/gumbel_searcher.cpp`
- `tests/cpp/search/test_gumbel_search.cpp`
- `configs/selfplay/tiny_gumbel_selfplay.yaml`
- `docs/phase12_gumbel_training_search.md`

Files updated:

- `cpp/search/include/chessmoe/search/mcts_searcher.h`
- `cpp/search/src/mcts_searcher.cpp`
- `cpp/search/CMakeLists.txt`
- `cpp/selfplay/include/chessmoe/selfplay/self_play_generator.h`
- `cpp/selfplay/src/self_play_generator.cpp`
- `configs/selfplay/tiny_selfplay.yaml`
- `tests/cpp/search/CMakeLists.txt`
- `tests/cpp/selfplay/test_selfplay.cpp`

## Core Data Structures

- `search::SearchMode`: `Puct` or `Gumbel`.
- `search::GumbelSearchLimits`: simulations, maximum considered root actions, value scale, deterministic test mode, and seed.
- `search::GumbelSearcher`: root-only training search that returns the existing `MctsResult` shape.
- `RootMoveStats::target_probability`: normalized policy-improvement target for replay-compatible self-play samples.
- `SelfPlayConfig::search_mode`: selects PUCT or Gumbel for training self-play.

## Main Algorithms

Root Gumbel search pseudocode:

```text
evaluate root with evaluator
normalize policy over legal moves
for each legal move:
  logit = log(policy_probability)
  g = 0 in deterministic tests, else sample Gumbel(0, 1)
sample top m legal moves by logit + g
remaining = sampled candidates
while simulations remain:
  evaluate each remaining candidate once, from root player's perspective
  if more budget remains and more than one candidate remains:
    keep the best half by logit + g + value_scale * mean_q
best_move = argmax sampled candidate by logit + g + value_scale * mean_q
policy_target = softmax(logit + g + value_scale * mean_q over sampled candidates)
return MctsResult with visit counts and policy_target
```

The current implementation is deliberately root-focused. Non-root traversal still belongs to the existing PUCT search path until there is a measured need to extend Gumbel policy improvement deeper into the tree.

## Tests

- `gumbel_search_tests`
  - `SearchMode::Puct` and `SearchMode::Gumbel` are distinct.
  - Deterministic Gumbel root selection is repeatable.
  - Considered candidates are legal moves only.
  - Policy-improvement targets normalize to one.
- `selfplay_tests`
  - Gumbel self-play emits replay-compatible legal policy entries.
  - Synthetic Gumbel visit counts sum to the configured search budget.
  - Gumbel target probabilities normalize in self-play samples.

## Completion Criteria

- Standard PUCT remains available and default.
- Gumbel search can be selected through self-play configuration.
- Gumbel root search uses legal-only root candidates.
- Sequential-halving-style narrowing spends the configured simulation budget.
- Deterministic test mode is repeatable.
- Replay-compatible visit/probability targets are produced.
- UCI match play remains on PUCT unless a future command/config explicitly opts into Gumbel.
- No GPU self-play pipeline or multithreaded GPU MCTS is added.

## Common Failure Modes

- Sampling illegal actions from dense policy logits.
- Double-counting policy prior by using different Gumbel samples for candidate selection and final scoring.
- Producing a target that does not sum to one.
- Letting Gumbel self-play become the default UCI search path.
- Treating root-only Gumbel search as a full Gumbel MuZero implementation.
- Adding GPU batching before CPU search semantics are stable.

## Next Step

Run short self-play with `configs/selfplay/tiny_gumbel_selfplay.yaml`, write replay chunks, train the dense transformer against those targets, and compare arena results against the standard PUCT-generated replay baseline.
