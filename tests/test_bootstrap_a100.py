from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "bootstrap_a100.sh"


def _find_native_bash() -> str | None:
    candidates = [
        shutil.which("bash") if os.name != "nt" else None,
        r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else None,
        r"C:\Program Files\Git\usr\bin\bash.exe" if os.name == "nt" else None,
    ]
    return next((path for path in candidates if path and Path(path).is_file()), None)


BASH = _find_native_bash()


def test_bootstrap_has_strict_shell_and_pinned_lmms_commit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -Eeuo pipefail" in text
    assert "bb1ebe76e7a942386c25c4664f902e0e59e8a401" in text
    assert "apply --reverse --check" in text
    assert "git patch-id --stable" in text
    assert "merge --ff-only" in text
    assert "status --porcelain" in text
    assert "git check-ref-format --branch" in text
    assert "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}" in text
    assert "check-ignore --quiet --no-index" in text


def test_bootstrap_never_passes_or_prints_hf_token() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '--token' not in text
    assert 'echo "${HF_TOKEN' not in text
    assert 'printf "${HF_TOKEN' not in text
    assert '"${hf_cli}" auth login' in text
    assert 'os.environ.get("HF_TOKEN")' in text
    assert "login(token=token, add_to_git_credential=False)" in text
    assert 'os.environ.pop("HF_TOKEN", None)' in text


def test_bootstrap_exposes_consistent_repository_interface() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--repo-url URL" in text
    assert "--branch BRANCH" in text
    assert "--repo-dir PATH" in text
    assert "--branch|--repo-branch" in text
    assert "WFS_REPO_URL" in text
    assert "WFS_REPO_BRANCH" in text
    assert "WFS_REPO_DIR" in text
    assert 'DETECTED_REPO_URL="$(' in text
    assert 'DETECTED_REPO_BRANCH="${detected_branch}"' in text


def test_paths_mode_permits_download_after_bootstrap() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '[[ "${DATASET_CHECK}" == "paths" ]]' in text
    assert 'mkdir -p -- "${raw_dir}"' in text
    assert 'if [[ "${DATASET_CHECK}" == "full" ]]; then' in text


def test_bootstrap_contains_required_a100_and_dataset_checks() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "Python 3.10 is required",
        "not the selected virtualenv",
        "requirements-phase-stable.txt",
        "nvidia-smi",
        '"A100" in name.upper()',
        "videomme_json_file.json",
        "mlvu_dev.json",
        "lvb_val.json",
        "pip check",
    ):
        assert required in text


@pytest.mark.skipif(BASH is None, reason="requires a native Bash runtime")
def test_bootstrap_bash_syntax_and_help_are_side_effect_free() -> None:
    syntax = subprocess.run(
        [BASH, "-n", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = subprocess.run(
        [BASH, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": "must-not-appear"},
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "Usage: bootstrap_a100.sh" in help_result.stdout
    assert "must-not-appear" not in help_result.stdout
    assert "must-not-appear" not in help_result.stderr
