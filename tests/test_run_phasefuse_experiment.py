from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phasefuse_experiment.sh"


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


def _script_arg() -> str:
    return SCRIPT.resolve().as_posix()


def test_phasefuse_launcher_has_valid_bash_and_help():
    syntax = subprocess.run(
        [_bash(), "-n", _script_arg()],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(
        [_bash(), _script_arg(), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "Outer origins are" in help_result.stdout
    assert "evaluation perturbations" in help_result.stdout
    assert "--skip-mllm" in help_result.stdout


def test_phasefuse_launcher_wires_all_compute_matched_arms_and_safe_resume():
    source = SCRIPT.read_text(encoding="utf-8")
    config = (ROOT / "configs" / "phasefuse_icassp.yaml").read_text(
        encoding="utf-8"
    )
    methods = (
        "uniform_dense",
        "dense_topk_mmr",
        "single_dwt",
        "single_swt",
        "multiphase_dwt",
        "multiphase_swt_mean",
        "dense_swt",
        "phasefuse",
    )
    for method in methods:
        assert method in config
    assert "preprocess-phasefuse" in source
    assert "run-phasefuse" in source
    assert "export-keyframes" in source
    assert "run_mllm_grid.sh" in source
    assert "evaluate-phasefuse-predictions" in source
    assert "valid_marker" in source
    assert "validate_artifact_bundle" in source
    assert "feature_bundle.jsonl" in source
    assert "source_video_bundle.jsonl" in source
    assert "trace_array_bundle.jsonl" in source
    assert "flock -n" in source
    assert "rm -rf" not in source
    assert "--max-pixels 200704" in source
    assert '--expected-budget "${FRAME_BUDGET}"' in source
    assert "--baseline-method dense_swt" in source
    assert "--treatment-method phasefuse" in source
    assert '--expected-methods "${METHODS[@]}"' in source
    assert "--runtime-signature" in source
    assert "strict full-cohort downstream analysis skipped" in source
    assert "CONFIG_CONTRACT" in source
    assert '--origins "${origins_csv}"' in source
    assert '--max-num-frames "${FRAME_BUDGET}"' in source


def test_phasefuse_launcher_uses_benchmark_specific_default_run_dirs():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'artifacts/phasefuse_${BENCHMARK}_dev20' in source
    assert 'RUN_DIR_RAW=""' in source


def test_qvhighlights_launcher_requires_mllm_skip():
    # The guard is intentionally reached before any filesystem/model work.
    result = subprocess.run(
        [_bash(), _script_arg(), "--benchmark", "qvhighlights"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "pass --skip-mllm" in result.stderr
