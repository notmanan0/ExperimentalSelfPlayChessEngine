from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys
import time
from typing import Any

from chessmoe.models.registry import ModelRegistry, RegistryEntry, promote_candidate
from chessmoe.pipeline.config import HardwareProfile, QualityProfile


@dataclass
class StageResult:
    name: str
    status: str
    elapsed_seconds: float
    details: dict[str, Any] | None = None


@dataclass
class PipelineConfig:
    phase: int
    hardware_profile: HardwareProfile
    quality_profile: QualityProfile
    selfplay_exe: Path = Path("build-nmake/bin/Debug/selfplay.exe")
    engine_path: Path | None = None
    replay_dir: Path = Path("data/replay")
    weights_dir: Path = Path("weights")
    registry_path: Path = Path("weights/registry.json")
    runs_dir: Path = Path("runs")
    allow_debug: bool = False
    resume: bool = False
    force: bool = False


class PipelineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.results: list[StageResult] = []
        self.registry = ModelRegistry(config.registry_path)

    def run_stage(self, name: str, command: list[str]) -> StageResult:
        started = time.monotonic()
        print(f"[stage] {name}")
        env = __import__("os").environ.copy()
        python_path = str(Path("python").resolve())
        env["PYTHONPATH"] = (
            python_path
            if not env.get("PYTHONPATH")
            else python_path + __import__("os").pathsep + env["PYTHONPATH"]
        )
        completed = subprocess.run(command, check=False, env=env)
        elapsed = time.monotonic() - started
        status = "passed" if completed.returncode == 0 else "failed"
        result = StageResult(name=name, status=status, elapsed_seconds=elapsed)
        self.results.append(result)
        if completed.returncode != 0:
            raise RuntimeError(
                f"stage failed: {name}; exit={completed.returncode}"
            )
        return result

    def stage_neural_selfplay(self) -> StageResult:
        hw = self.config.hardware_profile
        q = self.config.quality_profile
        output_dir = self.config.replay_dir / f"phase{self.config.phase}"
        command = [
            str(self.config.selfplay_exe),
            "--hardware-profile", hw.name,
            "--quality", q.name,
            "--phase", str(self.config.phase),
            "--output-dir", str(output_dir),
        ]
        if self.config.allow_debug:
            command.append("--allow-debug")
        if self.config.resume:
            command.append("--resume")
        if hw.evaluator == "tensorrt" and self.config.engine_path:
            command.extend(["--engine", str(self.config.engine_path)])
        return self.run_stage("neural-selfplay", command)

    def stage_bootstrap(self) -> StageResult:
        command = [
            str(self.config.selfplay_exe),
            "--config", "configs/selfplay/bootstrap_material.json",
            "--allow-debug",
        ]
        return self.run_stage("bootstrap", command)

    def stage_index_replay(self) -> StageResult:
        replay_dir = self.config.replay_dir / f"phase{self.config.phase}"
        index_path = self.config.replay_dir / "replay.sqlite"
        command = [
            sys.executable,
            "tools/convert/index_replay_dir.py",
            str(replay_dir),
            "--index", str(index_path),
        ]
        return self.run_stage("index-replay", command)

    def stage_validate_replay(self) -> StageResult:
        replay_dir = self.config.replay_dir / f"phase{self.config.phase}"
        command = [
            sys.executable,
            "tools/convert/validate_replay.py",
            str(replay_dir),
        ]
        return self.run_stage("validate-replay", command)

    def stage_train(self, training_config: str = "configs/training/dense_bootstrap.json") -> StageResult:
        command = [
            sys.executable,
            "-m", "chessmoe.training.train",
            "--config", training_config,
        ]
        return self.run_stage("train", command)

    def stage_export(self, checkpoint: Path, output: Path) -> StageResult:
        command = [
            sys.executable,
            "python/export/export_tiny_onnx.py",
            "--checkpoint", str(checkpoint),
            "--output", str(output),
        ]
        return self.run_stage("export", command)

    def stage_build_engine(self, onnx: Path, engine: Path, fp16: bool = False) -> StageResult:
        command = [
            sys.executable,
            "python/export/build_tensorrt_engine.py",
            "--onnx", str(onnx),
            "--engine", str(engine),
        ]
        if fp16:
            command.append("--fp16")
        return self.run_stage("build-engine", command)

    def stage_arena(self, candidate: Path, best: Path, arena_config: str) -> StageResult:
        command = [
            sys.executable,
            "-m", "chessmoe.analysis.arena",
            "--config", arena_config,
        ]
        return self.run_stage("arena", command)

    def stage_promote(self, candidate: Path, version: int) -> StageResult:
        started = time.monotonic()
        copied = promote_candidate(
            candidate, version,
            weights_dir=self.config.weights_dir,
            force=self.config.force,
        )
        elapsed = time.monotonic() - started
        return StageResult(
            name="promote",
            status="passed",
            elapsed_seconds=elapsed,
            details={"files": [str(p) for p in copied]},
        )

    def run_bootstrap(self) -> list[StageResult]:
        self.stage_bootstrap()
        self.stage_index_replay()
        self.stage_validate_replay()
        self.stage_train()
        return self.results

    def run_full_cycle(self) -> list[StageResult]:
        self.stage_neural_selfplay()
        self.stage_index_replay()
        self.stage_validate_replay()
        self.stage_train()
        return self.results

    def summary(self) -> str:
        lines = ["=== Pipeline Summary ==="]
        total = 0.0
        for r in self.results:
            lines.append(f"  {r.name}: {r.status} ({r.elapsed_seconds:.1f}s)")
            total += r.elapsed_seconds
        lines.append(f"  Total: {total:.1f}s")
        return "\n".join(lines)
