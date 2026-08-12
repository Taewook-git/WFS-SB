from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phase_stable.artifacts import iter_jsonl
from phase_stable.cli import build_parser
from phase_stable.multiphase import (
    DenseOuterRecordPayload,
    build_dense_outer_record,
    build_multiphase_manifest,
)
from phase_stable.phasefuse_analysis import evaluate_phasefuse_analysis
from phase_stable.phasefuse_experiment import (
    PHASEFUSE_METHODS,
    PhaseFuseExperimentConfig,
    config_from_mapping,
    run_phasefuse_experiment,
    run_phasefuse_method,
)


def _records(tmp_path: Path):
    manifest = build_multiphase_manifest(
        "video-1",
        34.0,
        num_phases=4,
        num_outer_origins=2,
        outer_origins_sec=(0.05, 0.55),
    )
    records = []
    for origin in manifest.outer_origins:
        timestamps = np.asarray(origin.dense_target_timestamps_sec)
        source_indices = (
            np.arange(timestamps.size, dtype=int) + origin.outer_origin_id * 1000
        )
        relevance = (
            0.5 + 0.25 * np.sin(timestamps * 0.7) + 0.20 * np.cos(timestamps * 0.23)
        )
        features = np.column_stack(
            (
                np.sin(timestamps),
                np.cos(timestamps),
                timestamps / max(timestamps[-1], 1.0),
            )
        ).astype(np.float32)
        feature_path = tmp_path / f"features_o{origin.outer_origin_id}.npy"
        np.save(feature_path, features, allow_pickle=False)
        records.append(
            build_dense_outer_record(
                manifest,
                dataset="videomme",
                question_id="q1",
                payload=DenseOuterRecordPayload(
                    outer_origin_id=origin.outer_origin_id,
                    actual_pts_sec=tuple(float(value) for value in timestamps),
                    source_frame_indices=tuple(int(value) for value in source_indices),
                    relevance_scores=tuple(float(value) for value in relevance),
                    pixel_hashes=tuple(f"{value:064x}" for value in source_indices),
                    visual_features_path=str(feature_path.resolve()),
                    metadata={"query": "what happens?", "query_metadata": {}},
                ),
            )
        )
    return records


def test_all_phasefuse_arms_share_candidates_and_emit_exact_source_budget(
    tmp_path: Path,
):
    records = _records(tmp_path)
    config = PhaseFuseExperimentConfig(frame_budget=8, min_frames_per_segment=2)
    rows = run_phasefuse_experiment(records, tmp_path / "run", config=config)

    assert len(rows) == len(records) * len(PHASEFUSE_METHODS)
    assert list(iter_jsonl(tmp_path / "run" / "traces.jsonl")) == rows
    for row in rows:
        assert len(row["selected_indices"]) == 8
        assert len(set(row["selected_indices"])) == 8
        assert len(set(row["selected_source_frame_indices"])) == 8
        assert row["selected_source_frame_indices"] == sorted(
            row["selected_source_frame_indices"]
        )
        assert Path(row["array_path"]).is_file()
        global_support = row["record_metadata"]["multiphase"][
            "manifest_common_valid_support_sec"
        ]
        assert min(row["selected_timestamps_sec"]) >= global_support[0] - 1e-12
        assert max(row["selected_timestamps_sec"]) <= global_support[1] + 1e-12
    for origin_id in range(2):
        arm_rows = [row for row in rows if row["origin_id"] == origin_id]
        assert {row["method"] for row in arm_rows} == set(PHASEFUSE_METHODS)
        assert len({tuple(row["source_frame_indices"]) for row in arm_rows}) == 1
        assert len({tuple(row["timestamps_sec"]) for row in arm_rows}) == 1

    full = [row for row in rows if row["method"] == "phasefuse"]
    assert all(row["max_boundary_count"] == 3 for row in full)
    assert all(row["zero_allocation_segments"] == 0 for row in full)
    assert all("selected_phase_uncertainty_mean" in row for row in full)
    for row in full:
        with np.load(row["array_path"], allow_pickle=False) as arrays:
            expected = float(
                np.mean(arrays["uncertainty"][arrays["fusion_domain_mask"]])
            )
        assert row["phase_uncertainty_mean"] == pytest.approx(expected)

    comparison = evaluate_phasefuse_analysis(
        rows,
        baseline_method="dense_swt",
        treatment_method="phasefuse",
        n_bootstrap=25,
        seed=4,
    )
    assert comparison["num_paired_items"] == 1
    assert comparison["comparison"]["n_clusters"] == 1

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["protocol"]["outer_origins_never_fused"] is True
    assert summary["frame_budget"] == 8


def test_method_subset_and_config_validation(tmp_path: Path):
    records = _records(tmp_path)
    config = PhaseFuseExperimentConfig(
        methods=("dense_topk_mmr", "phasefuse"),
        frame_budget=8,
    )
    rows = run_phasefuse_experiment(
        records,
        tmp_path / "subset",
        config=config,
        methods=("phasefuse",),
    )
    assert {row["method"] for row in rows} == {"phasefuse"}
    with pytest.raises(ValueError, match="enabled"):
        run_phasefuse_experiment(
            records,
            tmp_path / "bad",
            config=config,
            methods=("uniform_dense",),
        )
    loaded = config_from_mapping(
        {
            "methods": ["dense_topk_mmr", "phasefuse"],
            "frame_budget": 8,
            "min_frames_per_segment": 2,
            "legacy_selection": {"lambda_param": 0.4},
        }
    )
    assert loaded.methods == ("dense_topk_mmr", "phasefuse")
    assert loaded.legacy_selection.lambda_param == pytest.approx(0.4)
    with pytest.raises(ValueError, match="unsupported"):
        config_from_mapping({"methods": ["unknown"]})


def test_dense_feature_provenance_rejects_post_record_tampering(tmp_path: Path):
    record = _records(tmp_path)[0]
    feature_path = Path(record.visual_features_path)
    features = np.load(feature_path, allow_pickle=False)
    features[0, 0] += 1.0
    np.save(feature_path, features, allow_pickle=False)

    with pytest.raises(ValueError, match="provenance mismatch"):
        run_phasefuse_method(
            record,
            "phasefuse",
            PhaseFuseExperimentConfig(frame_budget=8),
        )


def test_phasefuse_cli_commands_are_exposed():
    parser = build_parser()
    preprocess = parser.parse_args(
        [
            "preprocess-phasefuse",
            "--benchmark",
            "videomme",
            "--questions-file",
            "questions.json",
            "--dataset-root",
            "dataset",
            "--output-dir",
            "out",
            "--signal-jsonl",
            "signals.jsonl",
            "--config",
            "config.yaml",
        ]
    )
    assert preprocess.command == "preprocess-phasefuse"
    run = parser.parse_args(
        [
            "run-phasefuse",
            "signals.jsonl",
            "run",
            "--config",
            "config.yaml",
            "--methods",
            "dense_swt",
            "phasefuse",
        ]
    )
    assert run.methods == ["dense_swt", "phasefuse"]
    downstream = parser.parse_args(
        [
            "evaluate-phasefuse-predictions",
            "--predictions",
            "predictions.jsonl",
            "--traces",
            "traces.jsonl",
            "--output",
            "summary.json",
        ]
    )
    assert downstream.baseline_method == "dense_swt"
