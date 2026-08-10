from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_a100_experiment.sh"
FETCHER = REPO_ROOT / "scripts" / "fetch_videomme.sh"


def _bash() -> str:
    candidates: list[str | None]
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    else:
        candidates = [shutil.which("bash")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("GNU Bash is unavailable")


@pytest.mark.parametrize("script", [LAUNCHER, FETCHER])
def test_a100_entry_scripts_have_valid_bash_syntax(script: Path) -> None:
    subprocess.run([_bash(), "-n", script.resolve().as_posix()], check=True)


def test_launcher_documents_and_chains_the_complete_default_run() -> None:
    result = subprocess.run(
        [_bash(), LAUNCHER.resolve().as_posix(), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "prompt for Hugging Face login" in result.stdout
    assert "DWT/SWT" in result.stdout
    assert "official" in result.stdout and "parser" in result.stdout
    assert "resume" in result.stdout

    source = LAUNCHER.read_text(encoding="utf-8")
    ordered_calls = [
        "bootstrap_a100.sh",
        "fetch_videomme.sh",
        "run_stage0.sh",
        "run_mllm_grid.sh",
        "evaluate-predictions",
    ]
    positions = [source.index(value) for value in ordered_calls]
    assert positions == sorted(positions)
    assert 'payload["experiment"]["frame_budget"] = budget' in source
    assert "HF_TOKEN" not in source or "printf 'HF_TOKEN" not in source


def test_fetcher_is_resumable_and_validates_requested_video_ids() -> None:
    result = subprocess.run(
        [_bash(), FETCHER.resolve().as_posix(), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "stops as soon as every requested MP4 exists" in result.stdout
    source = FETCHER.read_text(encoding="utf-8")
    assert '[[ -s "$data_dir/$video_id.mp4" ]]' in source
    assert 'hf download "$repo_id" "$archive_name"' in source
    assert "for chunk_number in $(seq -w 1 20)" in source
    assert "import zipfile" in source
    assert "python hf unzip" not in source
