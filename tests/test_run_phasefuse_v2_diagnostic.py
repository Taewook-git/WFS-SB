from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_phasefuse_v2_diagnostic.sh"


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


def _strict_merge_source() -> str:
    source = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"# STRICT_MERGE_PY_BEGIN\n(.*?)\n# STRICT_MERGE_PY_END",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _synthetic_prediction_inputs(tmp_path: Path) -> tuple[Path, ...]:
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
    base_rows: list[dict[str, object]] = []
    verified_rows: list[dict[str, object]] = []
    new_rows: list[dict[str, object]] = []
    for method in methods:
        for video_index in range(20):
            for question_index in range(3):
                gold = "ABCDE"[(video_index + question_index) % 5]
                for origin_id in range(5):
                    row: dict[str, object] = {
                        "dataset": "videomme",
                        "video_id": f"video-{video_index:02d}",
                        "question_id": f"question-{question_index}",
                        "origin_id": origin_id,
                        "method": method,
                        "prediction": gold if origin_id % 2 == 0 else "",
                        "gold": gold,
                    }
                    base_rows.append(row)
                    if method == "dense_swt":
                        verified_rows.append(dict(row))
    for row in verified_rows:
        replacement = dict(row)
        replacement["method"] = "phasefuse_v2"
        replacement["prediction"] = replacement["gold"]
        new_rows.append(replacement)

    base = tmp_path / "base.jsonl"
    verified = tmp_path / "verified.jsonl"
    new = tmp_path / "new.jsonl"
    summary = tmp_path / "summary.json"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(base, base_rows)
    _write_jsonl(verified, verified_rows)
    _write_jsonl(new, new_rows)
    summary.write_text(
        json.dumps(
            {
                "num_prediction_rows": 2400,
                "predictions_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return base, verified, new, summary, output


def test_phasefuse_v2_diagnostic_has_valid_bash_and_help() -> None:
    syntax = subprocess.run(
        [_bash(), "-n", _script_arg()],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(
        [_bash(), _script_arg(), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "BASE_RUN_DIR/v2_diagnostic" in result.stdout
    assert "--skip-mllm" in result.stdout
    assert "Only phasefuse_v2 is sent to Qwen" in result.stdout


def test_phasefuse_v2_diagnostic_wires_reuse_resume_and_exact_protocol() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'BASE_RUN_DIR_RAW="artifacts/phasefuse_videomme_dev20"' in source
    assert 'RUN_DIR="${BASE_RUN_DIR}/v2_diagnostic"' in source
    assert 'CONFIG_RAW="configs/phasefuse_v2_dev20.yaml"' in source
    assert "-m phase_stable preprocess-phasefuse" not in source
    assert 'BASE_SIGNALS="${BASE_RUN_DIR}/preprocess/dense_signals.jsonl"' in source
    assert "validate_base_preprocess" in source
    assert "validate_artifact_bundle" in source
    assert source.count("flock -n") >= 2
    assert "valid_marker" in source
    assert "rm -rf" not in source
    assert "--methods dense_swt phasefuse_v2" in source
    assert "--treatment-method phasefuse_v2" in source
    assert "--expected-budget 16" in source
    assert "--allow-annotation-subset" in source
    assert "--methods phasefuse_v2 --origins 0,1,2,3,4" in source
    assert "--max-num-frames 16 --max-pixels 200704" in source
    assert "snapshot_download" not in source
    assert "inherited_base_provenance_sha256" in source
    assert "--runtime-signature" in source
    assert "base_arm != verified_arm" in source
    assert "len(merged) != 600" in source
    assert "--expected-methods dense_swt phasefuse_v2" in source


def test_strict_merge_accepts_exact_300_plus_300_grid(tmp_path: Path) -> None:
    inputs = _synthetic_prediction_inputs(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", _strict_merge_source(), *(str(path) for path in inputs)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output_rows = [
        json.loads(line) for line in inputs[-1].read_text(encoding="utf-8").splitlines()
    ]
    assert len(output_rows) == 600
    assert [row["method"] for row in output_rows[:300]] == ["dense_swt"] * 300
    assert [row["method"] for row in output_rows[300:]] == ["phasefuse_v2"] * 300
    assert {
        (row["dataset"], row["video_id"], row["question_id"], row["origin_id"])
        for row in output_rows[:300]
    } == {
        (row["dataset"], row["video_id"], row["question_id"], row["origin_id"])
        for row in output_rows[300:]
    }


def test_strict_merge_hard_fails_on_cross_arm_gold_mismatch(tmp_path: Path) -> None:
    base, verified, new, summary, output = _synthetic_prediction_inputs(tmp_path)
    rows = [json.loads(line) for line in new.read_text(encoding="utf-8").splitlines()]
    rows[0]["gold"] = "E" if rows[0]["gold"] != "E" else "D"
    _write_jsonl(new, rows)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _strict_merge_source(),
            str(base),
            str(verified),
            str(new),
            str(summary),
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "gold" in result.stderr.lower()
