from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parents[1]
SCRIPT = REPOSITORY / "scripts" / "run_mllm_grid.sh"


def _bash_executable() -> str:
    if os.name == "nt":
        candidates = (
            Path("C:/Program Files/Git/bin/bash.exe"),
            Path("C:/Program Files/Git/usr/bin/bash.exe"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    executable = shutil.which("bash")
    if executable:
        return executable
    pytest.skip("bash is not installed")


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    resolved = path.resolve().as_posix()
    drive, remainder = resolved.split(":", 1)
    return f"/{drive.lower()}{remainder}"


def _run(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    command = [_bash_executable(), _bash_path(SCRIPT), *arguments]
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _keyframes(directory: Path, methods=("dwt", "swt"), origins=(0,)) -> None:
    directory.mkdir(parents=True)
    for method in methods:
        for origin in origins:
            (directory / f"videomme_{method}_origin{origin:02d}.json").write_text(
                "[]\n", encoding="utf-8"
            )


def _fake_python(path: Path, *, make_artifacts: bool) -> None:
    artifact_block = r'''
sleep 0.05
data_name="${data_files##*/}"
data_name="${data_name%%.json*}"
model_dir="${output_path}/Qwen__Qwen2.5-VL-7B-Instruct"
mkdir -p "${model_dir}"
printf '{"ok":true}\n' > "${model_dir}/${data_name}+now_results.json"
printf '{"doc_id":0,"videomme_perception_score":{"question_id":"q","pred_answer":"A","answer":"A"}}\n' > "${model_dir}/${data_name}+now_samples_${task}.jsonl"
''' if make_artifacts else ""
    path.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail
printf 'call\n' >> "${FAKE_CALL_LOG}"
printf 'HF_TOKEN=%s\n' "${HF_TOKEN-unset}"
output_path=""
data_files=""
task=""
while (($#)); do
  case "$1" in
    --output_path) output_path="$2"; shift 2 ;;
    --data_files) data_files="$2"; shift 2 ;;
    --tasks) task="$2"; shift 2 ;;
    *) shift ;;
  esac
done
'''
        + artifact_block,
        encoding="utf-8",
        newline="\n",
    )
    subprocess.run(
        [_bash_executable(), "-lc", f"chmod +x {shlex.quote(_bash_path(path))}"],
        check=True,
    )


def test_help_and_dry_run_show_benchmark_mapping_without_writes(tmp_path: Path):
    help_result = _run("--help")
    assert help_result.returncode == 0
    assert "longvideobench_val_v" in help_result.stdout
    assert "strict 7-field predictions JSONL" in help_result.stdout
    assert "never reparsed" in help_result.stdout

    keyframes = tmp_path / "keyframes"
    output = tmp_path / "output"
    _keyframes(keyframes)
    result = _run(
        "--benchmark",
        "videomme",
        "--keyframe-dir",
        _bash_path(keyframes),
        "--output-root",
        _bash_path(output),
        "--origins",
        "0",
        "--dry-run",
    )
    assert result.returncode == 0, result.stderr
    assert "--tasks videomme" in result.stdout
    assert "use_keyframe=True" in result.stdout
    assert "max_pixels=200704" in result.stdout
    assert 'test' in result.stdout
    assert "env -u HF_TOKEN" in result.stdout
    assert not output.exists()


def test_success_requires_logs_then_marker_and_valid_marker_resumes(tmp_path: Path):
    keyframes = tmp_path / "keyframes"
    output = tmp_path / "output"
    fake_python = tmp_path / "fake-python"
    call_log = tmp_path / "calls.txt"
    _keyframes(keyframes, methods=("dwt",), origins=(0,))
    _fake_python(fake_python, make_artifacts=True)
    environment = os.environ.copy()
    environment["HF_TOKEN"] = "DO_NOT_PRINT_THIS_TOKEN"
    environment["FAKE_CALL_LOG"] = _bash_path(call_log)
    arguments = (
        "--benchmark",
        "videomme",
        "--keyframe-dir",
        _bash_path(keyframes),
        "--output-root",
        _bash_path(output),
        "--methods",
        "dwt",
        "--origins",
        "0",
        "--python-bin",
        _bash_path(fake_python),
        "--no-convert",
    )

    first = _run(*arguments, env=environment)
    assert first.returncode == 0, first.stderr
    marker = output / "videomme" / "dwt" / "origin00" / ".complete"
    assert marker.is_file()
    assert "results_json=" in marker.read_text(encoding="utf-8")
    combined_output = first.stdout + first.stderr
    assert "DO_NOT_PRINT_THIS_TOKEN" not in combined_output
    assert "HF_TOKEN=unset" in first.stdout
    assert call_log.read_text(encoding="utf-8").splitlines() == ["call"]

    second = _run(*arguments, env=environment)
    assert second.returncode == 0, second.stderr
    assert "SKIP:" in second.stdout
    assert call_log.read_text(encoding="utf-8").splitlines() == ["call"]

    # A marker from a previous annotation must not hide changed selections.
    (keyframes / "videomme_dwt_origin00.json").write_text(
        '[{"changed":true}]\n', encoding="utf-8"
    )
    third = _run(*arguments, env=environment)
    assert third.returncode == 0, third.stderr
    assert "stale completion marker" in third.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == ["call", "call"]
    marker_text = marker.read_text(encoding="utf-8")
    assert "keyframe_sha256=" in marker_text
    assert "config_fingerprint=" in marker_text
    assert "results_sha256=" in marker_text
    assert "samples_sha256=" in marker_text

    changed_config = _run(*arguments, "--attention", "eager", env=environment)
    assert changed_config.returncode == 0, changed_config.stderr
    assert "stale completion marker" in changed_config.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "call",
        "call",
        "call",
    ]

    # Artifact mutation also invalidates a marker, even when its byte size is
    # unchanged (the stored SHA-256 is part of the resume contract).
    marker_values = dict(
        line.split("=", 1)
        for line in marker.read_text(encoding="utf-8").splitlines()
    )
    samples = Path(marker_values["samples_jsonl"])
    original_samples = samples.read_text(encoding="utf-8")
    samples.write_text(original_samples.replace('"pred_answer":"A"', '"pred_answer":"B"'), encoding="utf-8")
    changed_artifact = _run(*arguments, "--attention", "eager", env=environment)
    assert changed_artifact.returncode == 0, changed_artifact.stderr
    assert "stale completion marker" in changed_artifact.stderr
    assert call_log.read_text(encoding="utf-8").splitlines() == [
        "call",
        "call",
        "call",
        "call",
    ]


def test_zero_exit_without_expected_lmms_logs_fails_without_marker(tmp_path: Path):
    keyframes = tmp_path / "keyframes"
    output = tmp_path / "output"
    fake_python = tmp_path / "fake-python"
    call_log = tmp_path / "calls.txt"
    _keyframes(keyframes, methods=("dwt",), origins=(0,))
    _fake_python(fake_python, make_artifacts=False)
    environment = os.environ.copy()
    environment["FAKE_CALL_LOG"] = _bash_path(call_log)

    result = _run(
        "--benchmark",
        "videomme",
        "--keyframe-dir",
        _bash_path(keyframes),
        "--output-root",
        _bash_path(output),
        "--methods",
        "dwt",
        "--origins",
        "0",
        "--python-bin",
        _bash_path(fake_python),
        "--no-convert",
        env=environment,
    )
    assert result.returncode != 0
    assert "expected new result/sample logs are missing" in result.stderr
    assert not (output / "videomme" / "dwt" / "origin00" / ".complete").exists()


def test_completed_grid_is_merged_to_exact_prediction_rows(tmp_path: Path):
    keyframes = tmp_path / "keyframes"
    output = tmp_path / "output"
    predictions = tmp_path / "predictions.jsonl"
    fake_python = tmp_path / "fake-python"
    call_log = tmp_path / "calls.txt"
    _keyframes(keyframes, methods=("dwt", "swt"), origins=(0,))
    annotation = [
        {
            "video_id": "v",
            "question_id": "q",
            "answer": "A",
            "keyframe_indices": [1, 3],
        }
    ]
    for method in ("dwt", "swt"):
        (keyframes / f"videomme_{method}_origin00.json").write_text(
            json.dumps(annotation), encoding="utf-8"
        )
    _fake_python(fake_python, make_artifacts=True)
    environment = os.environ.copy()
    environment["FAKE_CALL_LOG"] = _bash_path(call_log)

    result = _run(
        "--benchmark",
        "videomme",
        "--keyframe-dir",
        _bash_path(keyframes),
        "--output-root",
        _bash_path(output),
        "--methods",
        "dwt,swt",
        "--origins",
        "0",
        "--python-bin",
        _bash_path(fake_python),
        "--converter-python",
        _bash_path(Path(sys.executable)),
        "--predictions-output",
        _bash_path(predictions),
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"dwt", "swt"}
    assert all(set(row) == {
        "dataset",
        "video_id",
        "question_id",
        "origin_id",
        "method",
        "prediction",
        "gold",
    } for row in rows)
