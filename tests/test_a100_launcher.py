from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_a100_experiment.sh"
FETCHER = REPO_ROOT / "scripts" / "fetch_videomme.sh"


def _provenance_checker_source() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    block = source.split("# MLLM_PROVENANCE_CHECK_BEGIN", 1)[1].split(
        "# MLLM_PROVENANCE_CHECK_END", 1
    )[0]
    return block.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _write_mllm_provenance_grid(
    root: Path,
    methods: tuple[str, str],
    signature: dict[str, object],
    *,
    benchmark: str = "videomme",
) -> None:
    for method in methods:
        for origin in range(5):
            cell = root / method / f"origin{origin:02d}"
            cell.mkdir(parents=True)
            result_path = cell / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        **signature,
                        "git_hash": f"ignored-{method}-{origin}",
                        "date": f"ignored-{origin}",
                        "path": str(cell),
                    }
                ),
                encoding="utf-8",
            )
            (cell / ".complete").write_text(
                "\n".join(
                    (
                        f"benchmark={benchmark}",
                        "task=videomme",
                        f"method={method}",
                        f"origin_id={origin}",
                        f"results_json={result_path}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )


def _mllm_signature(*, max_pixels: int = 200704) -> dict[str, object]:
    return {
        "config": {
            "model": "qwen2_5_vl",
            "model_args": f"pretrained=Qwen/Qwen2.5-VL,max_pixels={max_pixels}",
            "batch_size": "1",
        },
        "task_hashes": {"videomme": "shared-task-hash"},
        "versions": {"videomme": 1},
        "model_name": "Qwen/Qwen2.5-VL",
        "model_source": "qwen2_5_vl",
        "chat_template_sha": None,
        "system_instruction_sha": None,
    }


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
    assert "--matched-only" in result.stdout
    assert "10 new Qwen cells" in result.stdout

    source = LAUNCHER.read_text(encoding="utf-8")
    positions = [
        source.index("bootstrap_a100.sh"),
        source.index("fetch_videomme.sh"),
        source.index('bash "${SCRIPT_DIR}/run_stage0.sh"'),
        source.rindex('bash "${SCRIPT_DIR}/run_mllm_grid.sh"'),
        source.rindex("evaluate-predictions"),
    ]
    assert positions == sorted(positions)
    assert 'payload["experiment"]["frame_budget"] = budget' in source
    assert "--qwen-max-pixels N" in result.stdout
    assert "QWEN_MAX_PIXELS=200704" in source
    assert '--max-pixels "$QWEN_MAX_PIXELS"' in source
    assert "matched-selection" in source
    assert "--methods dwt_matched,swt_matched" in source
    assert '${RUN_DIR}/matched_cardinality/${matched_token}' in source
    assert "--baseline-method dwt_matched" in source
    assert "--treatment-method swt_matched" in source
    assert "if ((MATCHED_ONLY)); then" in source
    assert ".stage0_state/preprocess.done.json" in source
    assert "origin signal/features no longer match" in source
    assert "annotation subset differs" in source
    assert "cannot rewrite the completed Stage-0 config" in source
    assert "CONFIG_EXPLICIT=1" in source
    assert "completed Stage-0 analysis config is missing" in source
    assert "HF_TOKEN" not in source or "printf 'HF_TOKEN" not in source


def test_matched_launcher_checks_provenance_then_runs_joint_interaction() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    matched_evaluation = source.index(
        '"$matched_predictions" "$matched_summary"',
    )
    adaptive_required = source.index(
        '[[ -s "$adaptive_predictions" ]]', matched_evaluation
    )
    provenance_check = source.index(
        "# MLLM_PROVENANCE_CHECK_BEGIN", adaptive_required
    )
    interaction = source.index(
        '"$PYTHON_BIN" -m phase_stable evaluate-prediction-interaction',
        provenance_check,
    )
    assert matched_evaluation < adaptive_required < provenance_check < interaction
    assert '${matched_root}/adaptive_matched_interaction_summary.json' in source
    assert '"$adaptive_predictions" "$matched_predictions" "$interaction_summary"' in source
    assert "--adaptive-baseline-method dwt" in source
    assert "--adaptive-treatment-method swt" in source
    assert "--matched-baseline-method dwt_matched" in source
    assert "--matched-treatment-method swt_matched" in source
    interaction_block = source[interaction : source.index("printf '\\nMatched", interaction)]
    assert 'interaction_bootstrap_repetitions=50000' in source
    assert '--n-bootstrap "$interaction_bootstrap_repetitions"' in interaction_block
    assert '--seed "$SEED"' in interaction_block


def test_mllm_provenance_preflight_compares_all_expected_cells(tmp_path: Path) -> None:
    adaptive_root = tmp_path / "adaptive"
    matched_root = tmp_path / "matched"
    signature = _mllm_signature()
    _write_mllm_provenance_grid(adaptive_root, ("dwt", "swt"), signature)
    _write_mllm_provenance_grid(
        matched_root, ("dwt_matched", "swt_matched"), signature
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _provenance_checker_source(),
            str(adaptive_root),
            str(matched_root),
            "videomme",
            "5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "10 adaptive and 10 matched cells share one signature" in result.stdout


def test_mllm_provenance_preflight_rejects_cross_regime_drift(
    tmp_path: Path,
) -> None:
    adaptive_root = tmp_path / "adaptive"
    matched_root = tmp_path / "matched"
    _write_mllm_provenance_grid(
        adaptive_root, ("dwt", "swt"), _mllm_signature()
    )
    _write_mllm_provenance_grid(
        matched_root,
        ("dwt_matched", "swt_matched"),
        _mllm_signature(max_pixels=602112),
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _provenance_checker_source(),
            str(adaptive_root),
            str(matched_root),
            "videomme",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "adaptive and matched MLLM provenance do not match" in result.stderr
    assert "config" in result.stderr


def test_mllm_provenance_preflight_rejects_drift_within_one_grid(
    tmp_path: Path,
) -> None:
    adaptive_root = tmp_path / "adaptive"
    matched_root = tmp_path / "matched"
    signature = _mllm_signature()
    _write_mllm_provenance_grid(adaptive_root, ("dwt", "swt"), signature)
    _write_mllm_provenance_grid(
        matched_root, ("dwt_matched", "swt_matched"), signature
    )
    changed_result = adaptive_root / "swt" / "origin04" / "result.json"
    changed_payload = json.loads(changed_result.read_text(encoding="utf-8"))
    changed_payload["task_hashes"] = {"videomme": "different-task-hash"}
    changed_result.write_text(json.dumps(changed_payload), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _provenance_checker_source(),
            str(adaptive_root),
            str(matched_root),
            "videomme",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "adaptive MLLM provenance differs across cells" in result.stderr
    assert "task_hashes" in result.stderr


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
