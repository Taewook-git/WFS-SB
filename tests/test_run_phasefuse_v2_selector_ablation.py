from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phasefuse_v2_selector_ablation.sh"


def _bash() -> str:
    candidates = (
        [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
        if os.name == "nt"
        else [shutil.which("bash")]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("GNU Bash is unavailable")


def test_selector_ablation_runner_has_valid_bash_and_help() -> None:
    syntax = subprocess.run(
        [_bash(), "-n", SCRIPT.resolve().as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [_bash(), SCRIPT.resolve().as_posix(), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "selector-only" in help_result.stdout
    assert "never invokes lmms-eval or Qwen" in help_result.stdout
    assert "v2_selector_ablation" in help_result.stdout


def test_selector_ablation_runner_is_isolated_matched_and_provenance_safe() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'RUN_DIR="${BASE_RUN_DIR}/v2_selector_ablation"' in source
    assert 'CONFIG_RAW="configs/phasefuse_v2_selector_ablation_dev20.yaml"' in source
    assert "phase0_swt_v2_selector dense_swt_v2_selector phasefuse_v2" in source
    assert "--baseline-method phase0_swt_v2_selector" in source
    assert "--treatment-method phasefuse_v2" in source
    assert "--baseline-method dense_swt_v2_selector" in source
    assert "--n-bootstrap" in source
    assert '"n_clusters") != 20' in source
    assert '"num_paired_items") != 60' in source
    assert "validate" not in source or "base marker" in source
    assert "feature artifact failed authentication" in source
    assert "trace array failed authentication" in source
    assert "exact sorted K=16" in source
    assert "phase_marginalization_only_control" in source
    assert "run_mllm_grid.sh" not in source
    assert "lmms_eval" not in source
    assert "export-keyframes" not in source
    assert "rm -rf" not in source
    assert source.count("flock -n") >= 2
    assert "valid_marker" in source


def test_frozen_v2_config_remains_unchanged_and_ablation_config_is_separate() -> None:
    original_path = ROOT / "configs" / "phasefuse_v2_dev20.yaml"
    original_bytes = original_path.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "90caec604e757b0e4bd7877f20b72f0265d35bcdff56247ad86abb40e31e149f"
    )
    original = original_bytes.decode("utf-8")
    ablation = (
        ROOT / "configs" / "phasefuse_v2_selector_ablation_dev20.yaml"
    ).read_text(encoding="utf-8")
    assert "phase0_swt_v2_selector" not in original
    assert "dense_swt_v2_selector" not in original
    assert "phase0_swt_v2_selector" in ablation
    assert "dense_swt_v2_selector" in ablation
    assert "phase_marginalization_only_control_under_matched_global_selector" in ablation
    assert "frozen_selector_only_ablation" in ablation
