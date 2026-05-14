from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import json
import os
import shutil
import time
from typing import Any


class PromotionStatus(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass
class RegistryEntry:
    model_version: int
    parent_model_version: int | None = None
    checkpoint_path: str = ""
    onnx_path: str | None = None
    engine_path: str | None = None
    created_at: str = ""
    git_commit: str = ""
    training_data_used: list[str] = field(default_factory=list)
    replay_dirs: list[str] = field(default_factory=list)
    training_config: dict[str, Any] | None = None
    export_config: dict[str, Any] | None = None
    tensorrt_build_config: dict[str, Any] | None = None
    arena_result: dict[str, Any] | None = None
    promotion_status: str = PromotionStatus.CANDIDATE.value
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


class ModelRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.history_dir = self.path.parent / "history"

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"best": None, "entries": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import tempfile
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            shutil.move(tmp_path, str(self.path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def register(self, entry: RegistryEntry) -> None:
        data = self._read()
        entries = data.get("entries", [])
        for i, e in enumerate(entries):
            if e.get("model_version") == entry.model_version:
                entries[i] = entry.to_dict()
                data["entries"] = entries
                self._write(data)
                return
        entries.append(entry.to_dict())
        data["entries"] = entries
        self._write(data)

    def get_entry(self, version: int) -> RegistryEntry | None:
        data = self._read()
        for e in data.get("entries", []):
            if e.get("model_version") == version:
                return RegistryEntry.from_dict(e)
        return None

    def get_best(self) -> RegistryEntry | None:
        data = self._read()
        best_version = data.get("best")
        if best_version is None:
            return None
        return self.get_entry(best_version)

    def get_best_version(self) -> int | None:
        data = self._read()
        return data.get("best")

    def set_best(self, version: int) -> None:
        data = self._read()
        data["best"] = version
        self._write(data)

    def promote(self, version: int, arena_result: dict[str, Any]) -> None:
        entry = self.get_entry(version)
        if entry is None:
            raise ValueError(f"model version {version} not found in registry")
        entry.promotion_status = PromotionStatus.PROMOTED.value
        entry.arena_result = arena_result
        self.register(entry)

        current_best = self.get_best_version()
        if current_best is not None and current_best != version:
            self._archive_best(current_best)

        self.set_best(version)

    def reject(self, version: int, arena_result: dict[str, Any]) -> None:
        entry = self.get_entry(version)
        if entry is None:
            raise ValueError(f"model version {version} not found in registry")
        entry.promotion_status = PromotionStatus.REJECTED.value
        entry.arena_result = arena_result
        self.register(entry)

    def refuse_promotion_without_arena(
        self, version: int, force: bool = False
    ) -> None:
        if force:
            return
        entry = self.get_entry(version)
        if entry is None:
            raise RuntimeError(
                f"Cannot promote version {version}: not registered. "
                "Use --force to override."
            )
        if entry.arena_result is None:
            raise RuntimeError(
                f"Cannot promote version {version}: no arena result. "
                "Run arena first or use --force to override."
            )
        if entry.promotion_status == PromotionStatus.REJECTED.value:
            raise RuntimeError(
                f"Cannot promote version {version}: previously rejected. "
                "Use --force to override."
            )

    def _archive_best(self, version: int) -> None:
        self.history_dir.mkdir(parents=True, exist_ok=True)
        entry = self.get_entry(version)
        if entry is None:
            return
        if entry.checkpoint_path and Path(entry.checkpoint_path).exists():
            dest = self.history_dir / f"model_{version:06d}.pt"
            if not dest.exists():
                shutil.copy2(entry.checkpoint_path, dest)
        if entry.onnx_path and Path(entry.onnx_path).exists():
            dest = self.history_dir / f"model_{version:06d}.onnx"
            if not dest.exists():
                shutil.copy2(entry.onnx_path, dest)
        if entry.engine_path and Path(entry.engine_path).exists():
            dest = self.history_dir / f"model_{version:06d}.engine"
            if not dest.exists():
                shutil.copy2(entry.engine_path, dest)

    def list_entries(self) -> list[RegistryEntry]:
        data = self._read()
        return [RegistryEntry.from_dict(e) for e in data.get("entries", [])]

    def format_registry(self) -> str:
        data = self._read()
        best = data.get("best")
        entries = self.list_entries()
        lines = ["=== Model Registry ==="]
        lines.append(f"Best model: version {best}" if best else "Best model: none")
        lines.append(f"Total entries: {len(entries)}")
        lines.append("")
        for e in sorted(entries, key=lambda x: x.model_version):
            marker = " [BEST]" if e.model_version == best else ""
            lines.append(
                f"  v{e.model_version}: {e.promotion_status}{marker} "
                f"- {e.notes or e.checkpoint_path}"
            )
        return "\n".join(lines)


def promote_candidate(
    candidate_path: str | Path,
    version: int,
    *,
    weights_dir: str | Path = "weights",
    force: bool = False,
) -> list[Path]:
    candidate = Path(candidate_path)
    weights_dir = Path(weights_dir)
    if not candidate.exists():
        raise FileNotFoundError(f"candidate does not exist: {candidate}")
    if version < 1:
        raise ValueError("promotion version must be >= 1")

    history = weights_dir / "history"
    history.mkdir(parents=True, exist_ok=True)
    suffix = candidate.suffix
    if suffix not in {".pt", ".onnx", ".engine"}:
        raise ValueError("candidate must be a .pt, .onnx, or .engine artifact")

    best_path = weights_dir / f"best{suffix}"
    copied: list[Path] = []
    if best_path.exists() and not force:
        archive = history / f"model_{version - 1:06d}{suffix}"
        if archive.exists():
            raise FileExistsError(
                f"refusing to overwrite existing history artifact: {archive}"
            )
        shutil.copy2(best_path, archive)
        copied.append(archive)

    versioned = history / f"model_{version:06d}{suffix}"
    if versioned.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite candidate history artifact: {versioned}"
        )
    shutil.copy2(candidate, versioned)
    shutil.copy2(candidate, best_path)
    copied.extend([versioned, best_path])
    return copied
