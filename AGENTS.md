# chessmoe Project Guide

## Project Objective

`chessmoe` is a CUDA-first AlphaZero/Lc0-style self-play chess engine with a future transformer-MoE evaluator. Build it phase by phase, with correctness before speed and clear promotion gates before any model becomes the active engine.

This repository uses C++20, CUDA C++, Python 3.11+, CMake, pytest, and Catch2 or GoogleTest as appropriate.

## Operating Rules

- Do not skip phases.
- Do not produce vague architecture. Produce concrete files, APIs, class structures, algorithms, pseudocode, tests, commands, and completion criteria.
- Before implementation details that depend on modern APIs or tooling, check current official or high-quality sources and cite them when making claims about CUDA, TensorRT, Transformer Engine, NCCL, PyTorch, ONNX Runtime, or Lc0-style implementation details.
- Use metric units only.
- When writing Markdown, use Obsidian-friendly Markdown: headings with `#`, blank lines between paragraphs, `-` lists, fenced code blocks, `$...$` inline math, and `$$...$$` block math.
- When generating matplotlib plots, reason about layout explicitly and avoid overlaps using tools such as `loc='best'`, `bbox_to_anchor`, `tight_layout()`, or constrained layout.

## Source Baseline

Use these as starting points when validating tooling or implementation claims:

- NVIDIA CUDA Toolkit documentation: https://docs.nvidia.com/cuda/
- NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/
- NVIDIA Transformer Engine documentation: https://docs.nvidia.com/deeplearning/transformer-engine/
- NVIDIA NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/
- PyTorch documentation: https://docs.pytorch.org/docs/stable/
- ONNX Runtime CUDA Execution Provider documentation: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
- Lc0 technical explanation: https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/
- Lc0 source repository: https://github.com/LeelaChessZero/lc0

Prefer official documentation over blog posts. For Lc0-style architecture, prefer Lc0 docs and source code, then well-known AlphaZero/MuZero papers when needed.

## Hard Constraints

- Correctness before speed.
- Legal chess core must pass perft before search.
- UCI compliance is mandatory.
- Evaluator interface must stay backend-agnostic.
- CPU PUCT/MCTS first, GPU acceleration later.
- PyTorch dense baseline before transformer.
- Dense transformer before MoE.
- Arena gating before model promotion.
- Use binary replay chunks, not PGN, for training data.
- Separate match-play inference from self-play inference.
- Profile before CUDA optimization.

## Required Phase Output

For every requested phase, provide:

1. Objective.
2. Files to create.
3. Core data structures.
4. Main algorithms.
5. Concrete implementation plan.
6. Tests.
7. Commands to run.
8. Completion criteria.
9. Common failure modes.
10. Next step.

## Phase Roadmap

### Phase 0: Repository Skeleton

Create the build layout, dependency strategy, formatting rules, CI skeleton, and minimal smoke tests. No chess logic yet.

Expected output includes:

- Root `CMakeLists.txt`.
- `cmake/` helper modules.
- `src/`, `include/`, `tests/`, `python/`, `tools/`, `docs/`.
- C++ test framework wiring.
- Python virtual environment instructions.
- Basic CI commands.

Completion requires configuring and building a trivial C++ target, running a trivial C++ test, and running `pytest`.

### Phase 1: Legal Chess Core

Implement a legal chess state, move representation, legal move generation, make/unmake or copy-make semantics, FEN parsing, terminal detection, and perft.

Core APIs should include:

```cpp
namespace chessmoe::chess {

enum class Color : uint8_t;
enum class PieceType : uint8_t;
enum class Square : uint8_t;

struct Move;
struct Board;
struct Position;
struct GameState;

class MoveGenerator;
class Fen;

uint64_t perft(const Position& position, int depth);

}
```

Completion requires known perft test positions to pass before any search code is implemented.

### Phase 2: UCI Shell

Implement a UCI-compliant command loop around the legal chess core. Support `uci`, `isready`, `ucinewgame`, `position`, `go`, `stop`, `quit`, and `setoption` scaffolding.

Completion requires scripted UCI protocol tests and deterministic placeholder search output.

### Phase 3: Evaluator Interface

Define a backend-agnostic evaluator contract that returns policy logits or probabilities plus scalar value without binding search to PyTorch, ONNX Runtime, TensorRT, or CUDA.

Core API shape:

```cpp
namespace chessmoe::eval {

struct EncodedPosition;
struct PolicyValue;
struct BatchRequest;
struct BatchResult;

class IEvaluator {
public:
  virtual ~IEvaluator() = default;
  virtual BatchResult evaluate(const BatchRequest& request) = 0;
};

}
```

Completion requires a deterministic dummy evaluator and tests proving search can depend only on `IEvaluator`.

### Phase 4: CPU PUCT/MCTS

Implement CPU PUCT search with root policy priors, value backup, legal move masking, visit counts, Dirichlet noise only where appropriate, and deterministic test modes.

Completion requires unit tests for selection, expansion, backup, legal masking, terminal handling, and repeatable search with fixed seeds.

### Phase 5: Binary Replay Format

Implement binary self-play chunk writing and reading. Do not use PGN as the training data format.

Replay records should include encoded position features, legal move policy target, game result/value target, side to move, rule state needed for validation, and format versioning.

Completion requires round-trip tests, corruption tests, and documented compatibility rules.

### Phase 6: PyTorch Dense Baseline

Build the first trainable dense policy-value network in Python using PyTorch. Keep this as the correctness baseline before transformer or MoE work.

Completion requires data loading from binary chunks, a minimal training loop, checkpoint save/load, and inference parity tests for exported weights where applicable.

### Phase 7: Arena Gating

Implement match orchestration and statistical promotion criteria. A candidate model must pass arena gating before promotion.

Completion requires reproducible match configs, fixed seeds, stored results, and a clear promotion decision file.

### Phase 8: Match-Play Inference Backend

Add a production-oriented inference backend for match play. Keep it separate from self-play inference so latency and batching policies can diverge.

Potential backends include ONNX Runtime CUDA Execution Provider and TensorRT, after checking current documentation and compatibility.

Completion requires backend selection through configuration and parity tests against the PyTorch baseline within documented tolerances.

### Phase 9: Dense Transformer

Implement a dense transformer evaluator only after the dense baseline and arena flow are stable.

Completion requires training, inference, export, and arena comparisons against the dense baseline.

### Phase 10: Transformer-MoE

Implement MoE only after the dense transformer is correct and measurable. Keep routing, load balancing, expert capacity, and fallback behavior explicit and tested.

Completion requires ablation tests, routing diagnostics, and arena-gated promotion.

### Phase 11: CUDA Acceleration

Accelerate only profiled bottlenecks. Do not write custom CUDA kernels before proving the target is a bottleneck and before simpler library-backed approaches have been evaluated.

Completion requires profiles before and after optimization, correctness tests, and performance reports.

## Repository Layout Target

```text
.
|-- CMakeLists.txt
|-- cmake/
|-- include/chessmoe/
|   |-- chess/
|   |-- eval/
|   |-- search/
|   |-- selfplay/
|   `-- uci/
|-- src/
|   |-- chess/
|   |-- eval/
|   |-- search/
|   |-- selfplay/
|   `-- uci/
|-- tests/
|   |-- cpp/
|   `-- python/
|-- python/chessmoe/
|   |-- data/
|   |-- models/
|   |-- train/
|   `-- arena/
|-- tools/
|-- docs/
`-- weights/
```

## Testing Standards

- Legal chess correctness is gated by perft.
- Search correctness is gated by deterministic unit tests before strength testing.
- Evaluator backends must pass shape, legal-mask, determinism, and parity tests.
- Replay chunks must pass binary round-trip and compatibility tests.
- Arena promotion must be reproducible from stored configuration.
- CUDA changes must include correctness checks and profiling evidence.

## Build And Test Command Pattern

Use commands matching the current phase. Expected command families:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --config Debug
ctest --test-dir build --output-on-failure
python -m pytest tests/python
```

When CUDA is introduced:

```powershell
cmake -S . -B build-cuda -DCHESSMOE_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda --output-on-failure
```

## Common Failure Modes

- Optimizing move generation before perft is correct.
- Coupling MCTS directly to a specific neural backend.
- Treating UCI output as informal text instead of a protocol contract.
- Letting self-play batching assumptions leak into match-play inference.
- Training from PGN instead of versioned binary replay chunks.
- Promoting models without arena evidence.
- Adding CUDA kernels without profiling evidence.
- Relying on stale GPU library API assumptions.

## Definition Of Done

A phase is complete only when:

- The requested files exist.
- Public APIs are documented enough for the next phase.
- Tests for the phase pass.
- Commands and completion criteria are recorded.
- Known limitations and next steps are explicit.
