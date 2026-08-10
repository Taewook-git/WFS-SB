"""Reproducibility manifests for paper experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "PyWavelets",
    "scikit-learn",
    "av",
    "torch",
    "transformers",
    "decord",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repo_root: Path, arguments: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_environment(repo_root: str | Path | None = None) -> dict[str, Any]:
    packages: dict[str, Optional[str]] = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    environment: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_value(root, ["rev-parse", "HEAD"]),
        "git_dirty": bool(_git_value(root, ["status", "--porcelain"])),
    }
    try:
        import torch

        environment["cuda_available"] = bool(torch.cuda.is_available())
        environment["cuda_version"] = torch.version.cuda
        environment["cudnn_version"] = (
            torch.backends.cudnn.version() if torch.cuda.is_available() else None
        )
        environment["gpu_names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, RuntimeError):
        environment["cuda_available"] = False
        environment["cuda_version"] = None
        environment["cudnn_version"] = None
        environment["gpu_names"] = []
    return environment


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_reproducibility_manifests(
    output_dir: str | Path,
    *,
    command: str,
    config: Mapping[str, Any],
    input_paths: Sequence[str | Path] = (),
    extra: Optional[Mapping[str, Any]] = None,
    repo_root: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    manifest_dir = root / "manifest"
    inputs = []
    for value in input_paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"manifest input does not exist: {path}")
        inputs.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    run_manifest: dict[str, Any] = {
        "schema_version": 1,
        "command": str(command),
        "config": dict(config),
        "inputs": inputs,
        "extra": dict(extra or {}),
    }
    run_path = manifest_dir / "run_manifest.json"
    environment_path = manifest_dir / "environment.json"
    _atomic_json(run_path, run_manifest)
    _atomic_json(environment_path, collect_environment(repo_root))
    return run_path, environment_path


__all__ = [
    "TRACKED_PACKAGES",
    "collect_environment",
    "sha256_file",
    "write_reproducibility_manifests",
]
