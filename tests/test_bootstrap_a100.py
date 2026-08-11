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


def test_bootstrap_narrowly_handles_decord_metadata_and_smokes_pyav_keyframes() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"${VENV_PYTHON}" -m pip check' in text
    assert '"${check_output}" == "decord 0.6.0 is not supported on this platform"' in text
    assert "_read_video_pyav_keyframe" in text
    assert 'qwen_model_class.__module__ != "lmms_eval.models.chat.qwen2_5_vl"' in text
    assert 'metadata.get("video_backend") != "pyav"' in text
    assert 'metadata.get("frames_indices") != [0, 3]' in text
    assert "torch.equal(video.cpu(), expected)" in text
    assert '"decord",' not in text
    assert "installed Python dependencies are inconsistent" in text


def test_bootstrap_only_migrates_the_exact_legacy_lmms_patch() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'LEGACY_LMMS_PATCH_IDS=(' in text
    assert '"23eb590a95c58f878849e6d58e332a2728d4699a"' in text
    assert '"fc590b52df6e2503459240599de737716865ab30"' in text
    assert "snapshot_lmms_worktree_patch" in text
    assert "--untracked-files=all" in text
    assert 'git patch-id --stable <"${legacy_diff}"' in text
    assert '"${actual_id}" == "${legacy_id}"' in text
    assert 'diff --cached --quiet --' in text
    assert "index contains staged changes; refusing legacy migration" in text
    assert 'apply --reverse --check "${legacy_diff}"' in text
    assert 'apply --reverse "${legacy_diff}"' in text
    assert "was not clean after reversing the exact legacy patch" in text
    assert "does not apply after legacy migration" in text
    assert "git reset --hard" not in text
    assert "git clean -fd" not in text
    assert 'rm -rf -- "${LMMS_DIR}"' not in text


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
