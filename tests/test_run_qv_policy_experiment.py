from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from phase_stable.cli import build_parser

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_qv_policy_experiment.sh"


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
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if "GNU bash" in result.stdout:
                return candidate
    pytest.skip("GNU bash is unavailable")


def _script_arg() -> str:
    return SCRIPT.resolve().as_posix()


def test_qv_policy_runner_has_valid_syntax_and_safe_interface() -> None:
    bash = _bash()
    subprocess.run([bash, "-n", _script_arg()], check=True)
    result = subprocess.run(
        [bash, _script_arg(), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    for option in (
        "--manifest",
        "--expected-manifest-sha",
        "--dataset-root",
        "--run-dir",
        "--config",
        "--venv-dir",
        "--feature-batch-size",
        "--resume",
        "--no-resume",
        "--force-step",
        "--rehash-inputs",
    ):
        assert option in result.stdout
    assert "foreground process" in result.stdout
    assert "100 videos x 5 origins" in result.stdout


def test_qv_policy_runner_uses_only_registered_cli_contracts() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")

    expected = {
        "make-benchmark-manifests",
        "preprocess-benchmark",
        "policy-separation",
    }
    assert expected <= set(subparsers.choices)
    assert all(f"-m phase_stable {command}" in source for command in expected)
    assert "--benchmark qvhighlights" in source
    assert '--b-values "${B_VALUES[@]}"' in source
    assert "B_VALUES=(8 15 20)" in source
    assert "NUM_ORIGINS=5" in source
    assert "EXPECTED_VIDEOS=100" in source

    for command in ("make-benchmark-manifests", "preprocess-benchmark"):
        choices = (
            subparsers.choices[command]._option_string_actions["--benchmark"].choices
        )
        assert "qvhighlights" in choices


def test_qv_policy_runner_binds_resume_to_content_and_semantics() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    # Raw inputs, dense feature bundles, and trace arrays all contribute hashes.
    assert "raw_video_sha256_cache.json" in source
    assert '"sha256": sha' in source
    assert "signal-bundle::${SIGNALS_PATH}" in source
    assert "trace-bundle::${POLICY_DIR}/traces.jsonl" in source
    assert "output checksum mismatch" in source

    # A marker is written only after the stage-specific scientific audit.
    command_index = source.index('"$@" 2>&1 | tee "${log}"')
    validation_index = source.index('validate_step "${step}"', command_index)
    marker_index = source.index("write_marker", validation_index)
    assert command_index < validation_index < marker_index
    assert "fixed-B cardinality mismatch" in source
    assert "nested fixed-B sets are not strict supersets" in source
    assert "signal origin coverage mismatch" in source
    assert "manifest SHA-256 is not the locked val100 protocol input" in source
    assert "all_nested_b_feasible" in source
    assert "find wfs -type f -name '*.py'" in source
    assert "snapshot_download" in source
    assert '"script=${SCRIPT_HASH}"' in source

    # The runner never launches or records a PID itself; detachment belongs to
    # the caller, which keeps its exit status and foreground failure semantics.
    executable = source.split("usage()", maxsplit=1)[1]
    executable = executable.split("EOF", maxsplit=2)[-1]
    assert "disown" not in executable
    assert "setsid" not in executable
    assert "PID_FILE" not in source
