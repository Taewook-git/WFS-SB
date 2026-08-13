from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evaluate_rc12_gate import evaluate_gate
from scripts.evaluate_canonical_gate import evaluate_gate as evaluate_canonical_gate

ROOT = Path(__file__).parents[1]


def test_qwen_launcher_is_approval_gated_locked_and_runs_ten_cells():
    runner = (ROOT / "scripts/run_rc12_exact_diagnostic.sh").read_text()
    launcher = (ROOT / "scripts/launch_rc12_qwen_approved.sh").read_text()
    assert "run_mllm_grid.sh" not in runner
    assert '[[ "${APPROVED}" == "RC12_DEV20_QWEN" ]]' in launcher
    assert ".rc12_qwen.lock" in launcher and "flock -n 9" in launcher
    assert '--evaluation-root) EVALUATION_ROOT=' in launcher
    assert 'EVALUATION_DATA_ROOT="${EVALUATION_ROOT}/datasets/videomme/data"' in launcher
    assert "source member is not bound to evaluation dataset root" in launcher
    assert '--repo-root "${EVALUATION_ROOT}"' in launcher
    assert '"evaluation_root":evaluation_root' in launcher
    assert "--methods canonical_uniform,phasefuse_rc12 --origins 0,1,2,3,4" in launcher
    assert '"independent_cells":10' in launcher
    assert '"uniform_inference_reused":False' in launcher
    for token in (
        "WFS_SOURCE_SHA",
        "LMMS_SOURCE_SHA",
        "PACKAGE_SIGNATURE",
        "QWEN_SIGNATURE",
        "VALIDATION_SHA",
        "RUNTIME_SIGNATURE",
        "keyframe content drift",
        "source member drift",
    ):
        assert token in launcher


def test_rc14_launcher_is_five_cell_reuse_gated_and_frozen():
    launcher = (ROOT / "scripts/launch_rc14_qwen_approved.sh").read_text()
    assert '[[ "${APPROVED}" == "RC14_DEV20_QWEN" ]]' in launcher
    assert ".rc14_qwen.lock" in launcher and "flock -n 9" in launcher
    assert "--methods phasefuse_rc14 --origins 0,1,2,3,4" in launcher
    assert "--repo-root \"${EVALUATION_ROOT}\"" in launcher
    assert "canonical_uniform_inference_reused" in launcher
    assert "uniform prediction reuse drift" in launcher
    assert "logical_selected_hash_comparisons']==4800" in launcher


def test_generic_canonical_gate_supports_rc14_and_strict_boundaries():
    def summary(low=-0.029, high=0.029):
        return {
            "stability": {
                "baseline_method": "canonical_uniform",
                "treatment_method": "phasefuse_rc14",
                "comparison": {
                    "effect_definition": "treatment - baseline",
                    "effect_order": [
                        "delta_mean_accuracy",
                        "delta_pairwise_answer_disagreement",
                    ],
                    "estimate": [0.0, 0.0],
                    "ci_low": [low, -0.01],
                    "ci_high": [0.02, high],
                },
            }
        }

    assert evaluate_canonical_gate(summary(), "phasefuse_rc14")["status"] == "pass"
    assert evaluate_canonical_gate(summary(low=-0.03), "phasefuse_rc14")["status"] == "fail"
    assert evaluate_canonical_gate(summary(high=0.03), "phasefuse_rc14")["status"] == "fail"


def _prediction(method: str, origin: int, video: str, question: str):
    return {
        "dataset": "videomme",
        "video_id": video,
        "question_id": question,
        "origin_id": origin,
        "method": method,
        "prediction": "A",
        "gold": "A",
    }


def test_strict_merger_validates_ten_cell_primary_and_context(tmp_path: Path):
    primary = tmp_path / "primary.jsonl"
    base = tmp_path / "base.jsonl"
    output = tmp_path / "output.jsonl"
    context = tmp_path / "context.jsonl"
    primary_rows = []
    base_rows = []
    for item in range(60):
        video = f"v{item // 3:02d}"
        question = f"q{item:02d}"
        for origin in range(5):
            for method in ("canonical_uniform", "phasefuse_rc12"):
                primary_rows.append(_prediction(method, origin, video, question))
            for method in ("uniform_dense", "dense_swt", "phasefuse_v2"):
                base_rows.append(_prediction(method, origin, video, question))

    def write(path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    write(primary, primary_rows)
    write(base, base_rows)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/merge_rc12_predictions.py"),
            "--primary",
            str(primary),
            "--context-arm",
            f"uniform_dense=uniform_dense@{base}",
            "--context-arm",
            f"dense_swt_v2_selector=dense_swt@{base}",
            "--context-arm",
            f"phasefuse_v2=phasefuse_v2@{base}",
            "--output",
            str(output),
            "--context-output",
            str(context),
        ],
        check=True,
    )
    assert len(output.read_text().splitlines()) == 600
    contextual = [json.loads(line) for line in context.read_text().splitlines()]
    assert len(contextual) == 1500
    assert {row["method"] for row in contextual} == {
        "canonical_uniform",
        "phasefuse_rc12",
        "uniform_dense",
        "dense_swt_v2_selector",
        "phasefuse_v2",
    }


def _summary(accuracy_low: float, pad_high: float) -> dict:
    return {
        "stability": {
            "baseline_method": "canonical_uniform",
            "treatment_method": "phasefuse_rc12",
            "comparison": {
                "effect_definition": "treatment - baseline",
                "effect_order": [
                    "delta_pairwise_answer_disagreement",
                    "delta_mean_accuracy",
                ],
                "estimate": [0.0, 0.0],
                "ci_low": [-0.1, accuracy_low],
                "ci_high": [pad_high, 0.1],
            },
        }
    }


def test_frozen_gate_uses_strict_bounds_and_correct_direction():
    assert evaluate_gate(_summary(-0.029, 0.029))["status"] == "pass"
    assert evaluate_gate(_summary(-0.03, 0.029))["status"] == "fail"
    assert evaluate_gate(_summary(-0.029, 0.03))["status"] == "fail"
    reversed_summary = _summary(-0.029, 0.029)
    reversed_summary["stability"]["baseline_method"] = "phasefuse_rc12"
    with pytest.raises(ValueError, match="direction"):
        evaluate_gate(reversed_summary)
