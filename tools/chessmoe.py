#!/usr/bin/env python3
"""chessmoe - AlphaZero-style self-play chess engine pipeline."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import time


def cmd_probe(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.config import list_hardware_profiles, load_hardware_profile
    print("Available hardware profiles:")
    for name in list_hardware_profiles():
        p = load_hardware_profile(name)
        print(f"  {name}: {p.description}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from chessmoe.analysis.calibration import (
        run_calibration_matrix, save_calibration_result, print_calibration_summary,
    )
    selfplay_exe = Path(args.selfplay_exe) if hasattr(args, "selfplay_exe") and args.selfplay_exe else Path("build-nmake/bin/Debug/selfplay.exe")
    result = run_calibration_matrix(
        selfplay_exe=selfplay_exe,
        hardware_profile=args.hardware_profile,
        quality=args.quality or "debug_smoke",
        concurrent_games_list=[int(x) for x in args.concurrent_games.split(",")] if hasattr(args, "concurrent_games") and args.concurrent_games else None,
        fixed_batch_list=[int(x) for x in args.fixed_batch.split(",")] if hasattr(args, "fixed_batch") and args.fixed_batch else None,
        flush_ms_list=[int(x) for x in args.flush_ms.split(",")] if hasattr(args, "flush_ms") and args.flush_ms else None,
        games_per_point=args.games if hasattr(args, "games") and args.games else 4,
    )
    print_calibration_summary(result)
    save_calibration_result(result, Path(args.output) if hasattr(args, "output") and args.output else Path("data/calibration/result.json"))
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile
    from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
    hw = load_hardware_profile("cpu_bootstrap_debug")
    q = load_quality_profile("fast_bootstrap")
    config = PipelineConfig(phase=0, hardware_profile=hw, quality_profile=q, allow_debug=True)
    runner = PipelineRunner(config)
    runner.run_bootstrap()
    print(runner.summary())
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile
    from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
    hw = load_hardware_profile(args.hardware_profile)
    q = load_quality_profile(args.quality or "balanced_generation")
    config = PipelineConfig(
        phase=args.phase, hardware_profile=hw, quality_profile=q,
        engine_path=Path(args.engine) if args.engine else None,
        allow_debug=args.allow_debug, resume=args.resume,
    )
    runner = PipelineRunner(config)
    runner.stage_neural_selfplay()
    runner.stage_index_replay()
    runner.stage_validate_replay()
    print(runner.summary())
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile
    from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
    hw = load_hardware_profile("cpu_bootstrap_debug")
    q = load_quality_profile("fast_bootstrap")
    config = PipelineConfig(phase=0, hardware_profile=hw, quality_profile=q)
    runner = PipelineRunner(config)
    runner.stage_train(args.config)
    print(runner.summary())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from chessmoe.export.onnx_export import export_policy_value_onnx
    from chessmoe.training.checkpoint import load_checkpoint
    model = load_checkpoint(args.checkpoint)
    result = export_policy_value_onnx(model, args.output)
    print(f"Export: {result.status} -> {result.path}")
    return 0


def cmd_build_engine(args: argparse.Namespace) -> int:
    import subprocess
    command = [sys.executable, "python/export/build_tensorrt_engine.py",
               "--onnx", args.onnx, "--engine", args.engine]
    if args.fp16:
        command.append("--fp16")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def cmd_arena(args: argparse.Namespace) -> int:
    from chessmoe.analysis.arena import run_arena, load_arena_config
    config = load_arena_config(args.config)
    result = run_arena(config)
    print(f"arena: games={result.summary.games} score={result.summary.score_rate:.3f} decision={result.decision.value}")
    return 0


def cmd_arena_neural(args: argparse.Namespace) -> int:
    from chessmoe.analysis.arena import load_arena_config
    from chessmoe.analysis.neural_arena import run_neural_arena, MctsArenaConfig, OnnxModelEvaluator, PytorchModelEvaluator
    config = load_arena_config(args.config)
    if args.candidate.endswith(".onnx"):
        candidate_eval = OnnxModelEvaluator(Path(args.candidate))
    else:
        candidate_eval = PytorchModelEvaluator(Path(args.candidate))
    if args.best.endswith(".onnx"):
        best_eval = OnnxModelEvaluator(Path(args.best))
    else:
        best_eval = PytorchModelEvaluator(Path(args.best))
    mcts_config = MctsArenaConfig(visits=args.visits)
    result = run_neural_arena(config, candidate_eval, best_eval, mcts_config)
    print(f"arena: games={result.summary.games} score={result.summary.score_rate:.3f} decision={result.decision.value}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    from chessmoe.models.registry import promote_candidate, ModelRegistry
    if not args.force:
        registry = ModelRegistry("weights/registry.json")
        entry = registry.get_entry(args.version)
        if entry and entry.arena_result is None:
            print("ERROR: No arena result found for this candidate.")
            print("Run arena first, or use --force to promote without arena evidence.")
            return 1
    copied = promote_candidate(args.candidate, args.version, force=args.force)
    print(f"Promoted: {args.candidate} -> version {args.version}")
    for f in copied:
        print(f"  {f}")
    return 0


def cmd_full_cycle(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.config import load_hardware_profile, load_quality_profile
    from chessmoe.pipeline.runner import PipelineConfig, PipelineRunner
    hw = load_hardware_profile(args.hardware_profile)
    q = load_quality_profile(args.quality or "balanced_generation")
    config = PipelineConfig(
        phase=args.phase, hardware_profile=hw, quality_profile=q,
        engine_path=Path(args.engine) if hasattr(args, "engine") and args.engine else None,
        allow_debug=getattr(args, "allow_debug", False),
    )
    runner = PipelineRunner(config)
    runner.run_full_cycle()
    print(runner.summary())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.report import generate_run_report
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("No runs found.")
        return 0
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.is_dir() and (run_dir / "summary.json").exists():
            import json
            data = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            print(f"{run_dir.name}: games={data.get('games_completed', 0)} "
                  f"evaluator={data.get('evaluator', 'unknown')} "
                  f"health={'ok' if data.get('health_passed') else 'warn'}")
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    from chessmoe.models.registry import ModelRegistry
    registry = ModelRegistry("weights/registry.json")
    print(registry.format_registry())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from chessmoe.pipeline.report import generate_run_report, generate_html_report
    run_dir = Path(args.run_dir)
    if args.format == "html":
        report = generate_html_report(run_dir)
        out_path = Path(args.output) if args.output else run_dir / "report.html"
        out_path.write_text(report, encoding="utf-8")
        print(f"HTML report written to: {out_path}")
    else:
        report = generate_run_report(run_dir)
        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"Markdown report written to: {args.output}")
        else:
            print(report)
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    from chessmoe.analysis.diagnostics import (
        analyze_replay_policy_targets, analyze_value_calibration,
    )
    if args.metric == "policy":
        result = analyze_replay_policy_targets(Path(args.replay_index))
    elif args.metric == "value":
        result = analyze_value_calibration(Path(args.replay_index))
    else:
        print(f"Unknown metric: {args.metric}")
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_replay_buffer(args: argparse.Namespace) -> int:
    from chessmoe.analysis.replay_buffer import (
        RollingReplayBuffer, deduplicate_replay_index,
        detect_duplicate_positions, compute_replay_statistics,
    )
    db = Path(args.replay_index)
    if args.action == "stats":
        stats = compute_replay_statistics(db)
        print(json.dumps(stats, indent=2))
    elif args.action == "dedup":
        removed = deduplicate_replay_index(db)
        print(f"Removed {removed} duplicate entries")
    elif args.action == "duplicates":
        result = detect_duplicate_positions(db)
        print(json.dumps(result, indent=2))
    elif args.action == "maintain":
        buffer = RollingReplayBuffer(db, max_chunks=args.max_chunks)
        result = buffer.maintain()
        print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chessmoe",
        description="AlphaZero-style self-play chess engine pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("probe", help="Detect hardware and recommend profile")

    cal = sub.add_parser("calibrate", help="Run calibration benchmark")
    cal.add_argument("--hardware-profile", required=True)
    cal.add_argument("--quality", default="debug_smoke")
    cal.add_argument("--concurrent-games", default=None, help="comma-separated list")
    cal.add_argument("--fixed-batch", default=None, help="comma-separated list")
    cal.add_argument("--flush-ms", default=None, help="comma-separated list")
    cal.add_argument("--games", type=int, default=4)
    cal.add_argument("--output", default=None)
    cal.add_argument("--selfplay-exe", default=None)

    sub.add_parser("bootstrap", help="Run material bootstrap pipeline")

    gen = sub.add_parser("generate", help="Run neural self-play generation")
    gen.add_argument("--phase", type=int, required=True)
    gen.add_argument("--hardware-profile", required=True)
    gen.add_argument("--quality", default="balanced_generation")
    gen.add_argument("--engine", default=None)
    gen.add_argument("--allow-debug", action="store_true")
    gen.add_argument("--resume", action="store_true")

    trn = sub.add_parser("train", help="Train model from replay data")
    trn.add_argument("--config", required=True)

    exp = sub.add_parser("export", help="Export model to ONNX")
    exp.add_argument("--checkpoint", required=True)
    exp.add_argument("--output", required=True)

    bld = sub.add_parser("build-engine", help="Build TensorRT engine")
    bld.add_argument("--onnx", required=True)
    bld.add_argument("--engine", required=True)
    bld.add_argument("--fp16", action="store_true")

    arn = sub.add_parser("arena", help="Run arena gating (seeded backend)")
    arn.add_argument("--config", required=True)

    arn_n = sub.add_parser("arena-neural", help="Run arena with neural evaluator")
    arn_n.add_argument("--config", required=True)
    arn_n.add_argument("--candidate", required=True, help="Path to candidate .pt or .onnx")
    arn_n.add_argument("--best", required=True, help="Path to best .pt or .onnx")
    arn_n.add_argument("--visits", type=int, default=64)

    pr = sub.add_parser("promote", help="Promote candidate model")
    pr.add_argument("--candidate", required=True)
    pr.add_argument("--version", type=int, required=True)
    pr.add_argument("--force", action="store_true")

    full = sub.add_parser("full-cycle", help="Run full training cycle")
    full.add_argument("--phase", type=int, required=True)
    full.add_argument("--hardware-profile", required=True)
    full.add_argument("--quality", default="balanced_generation")
    full.add_argument("--engine", default=None)
    full.add_argument("--train-config", default=None, help="Training config JSON")
    full.add_argument("--checkpoint", default=None, help="Checkpoint path")
    full.add_argument("--onnx-output", default=None, help="ONNX output path")
    full.add_argument("--engine-output", default=None, help="TensorRT engine output")
    full.add_argument("--arena-config", default=None, help="Arena config JSON")
    full.add_argument("--candidate", default=None, help="Candidate weights path")
    full.add_argument("--best", default=None, help="Best weights path")
    full.add_argument("--skip-engine-build", action="store_true")
    full.add_argument("--skip-promotion", action="store_true")
    full.add_argument("--allow-debug", action="store_true")

    sub.add_parser("status", help="Show run status")
    sub.add_parser("registry", help="Show model registry")

    rpt = sub.add_parser("report", help="Generate run report")
    rpt.add_argument("--run-dir", required=True)
    rpt.add_argument("--format", choices=["md", "html"], default="md")
    rpt.add_argument("--output", default=None)

    diag = sub.add_parser("diagnose", help="Run diagnostics on replay data")
    diag.add_argument("--replay-index", required=True)
    diag.add_argument("--metric", choices=["policy", "value"], required=True)

    rb = sub.add_parser("replay-buffer", help="Replay buffer management")
    rb.add_argument("--replay-index", required=True)
    rb.add_argument("--action", choices=["stats", "dedup", "duplicates", "maintain"], required=True)
    rb.add_argument("--max-chunks", type=int, default=10000)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "probe": cmd_probe,
        "calibrate": cmd_calibrate,
        "bootstrap": cmd_bootstrap,
        "generate": cmd_generate,
        "train": cmd_train,
        "export": cmd_export,
        "build-engine": cmd_build_engine,
        "arena": cmd_arena,
        "arena-neural": cmd_arena_neural,
        "promote": cmd_promote,
        "full-cycle": cmd_full_cycle,
        "status": cmd_status,
        "registry": cmd_registry,
        "report": cmd_report,
        "diagnose": cmd_diagnose,
        "replay-buffer": cmd_replay_buffer,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
