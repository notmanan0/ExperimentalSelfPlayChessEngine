import os
import shutil
import tempfile
from pathlib import Path

from _pytest.tmpdir import TempPathFactory


os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


if os.name == "nt":
    _original_getbasetemp = TempPathFactory.getbasetemp
    _original_mktemp = TempPathFactory.mktemp

    def _mkdir_accessible(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        list(path.iterdir())
        return path.resolve()

    def _safe_rmtree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    def _windows_acl_safe_getbasetemp(self: TempPathFactory) -> Path:
        if self._basetemp is not None:
            return self._basetemp

        if self._given_basetemp is None:
            root = Path(os.environ.get("PYTEST_DEBUG_TEMPROOT") or tempfile.gettempdir())
            base = root / "pytest-chessmoe"
            for index in range(1000):
                candidate = base / f"pytest-{index}"
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                    self._basetemp = candidate.resolve()
                    return self._basetemp
                except FileExistsError:
                    continue
            return _original_getbasetemp(self)

        requested = Path(self._given_basetemp)
        try:
            _safe_rmtree(requested)
            self._basetemp = _mkdir_accessible(requested)
        except PermissionError:
            for index in range(1000):
                fallback = requested.with_name(
                    f"{requested.name}-accessible-{os.getpid()}-{index}"
                )
                if not fallback.exists():
                    break
            self._basetemp = _mkdir_accessible(fallback)
        return self._basetemp

    def _windows_acl_safe_mktemp(
        self: TempPathFactory,
        basename: str,
        numbered: bool = True,
    ) -> Path:
        basename = self._ensure_relative_to_basetemp(basename)
        base = self.getbasetemp()
        if not numbered:
            path = base / basename
            path.mkdir()
            return path
        for index in range(1000):
            path = base / f"{basename}{index}"
            try:
                path.mkdir()
                return path
            except FileExistsError:
                continue
        return _original_mktemp(self, basename, numbered)

    TempPathFactory.getbasetemp = _windows_acl_safe_getbasetemp
    TempPathFactory.mktemp = _windows_acl_safe_mktemp
