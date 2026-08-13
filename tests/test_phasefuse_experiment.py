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
    PHASEFUSE_ANALYSIS_METHODS,
    PHASEFUSE_DEFAULT_METHODS,
    PHASEFUSE_METHODS,
    PhaseFuseExperimentConfig,
    config_from_mapping,
    load_phasefuse_config,
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

    assert "phasefuse_v2" in PHASEFUSE_METHODS
    assert config.methods == PHASEFUSE_DEFAULT_METHODS
    assert "phasefuse_v2" not in config.methods
    assert "canonical_uniform" not in PHASEFUSE_METHODS
    assert "phasefuse_rc12" not in PHASEFUSE_METHODS
    assert "phasefuse_rc14" not in PHASEFUSE_METHODS
    assert "phasefuse_nested_r2" not in PHASEFUSE_METHODS
    assert "canonical_uniform" in PHASEFUSE_ANALYSIS_METHODS
    assert "phasefuse_rc12" in PHASEFUSE_ANALYSIS_METHODS
    assert "phasefuse_rc14" in PHASEFUSE_ANALYSIS_METHODS
    assert "phasefuse_nested_r2" in PHASEFUSE_ANALYSIS_METHODS
    assert len(rows) == len(records) * len(config.methods)
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
        assert {row["method"] for row in arm_rows} == set(config.methods)
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
    upgraded = config_from_mapping(
        {
            "methods": ["dense_swt", "phasefuse_v2"],
            "frame_budget": 16,
            "num_phases": 4,
            "selection_strategy": "global_coverage",
            "uniform_reserve": 8,
            "selection_event_weight": 0.25,
            "component_scaling": "percentile",
            "min_selection_distance_sec": 0.5,
        }
    )
    assert upgraded.methods == ("dense_swt", "phasefuse_v2")
    assert upgraded.selection_strategy == "global_coverage"
    assert upgraded.uniform_reserve == 8
    assert upgraded.selection_event_weight == pytest.approx(0.25)
    assert upgraded.component_scaling == "percentile"
    assert upgraded.min_selection_distance_sec == pytest.approx(0.5)
    assert PhaseFuseExperimentConfig(frame_budget=4).uniform_reserve == 8
    with pytest.raises(ValueError, match="cannot exceed"):
        PhaseFuseExperimentConfig(
            methods=("phasefuse_v2",),
            frame_budget=4,
        )
    with pytest.raises(ValueError, match="unsupported"):
        config_from_mapping({"methods": ["unknown"]})

    file_config = load_phasefuse_config(
        Path(__file__).parents[1] / "configs" / "phasefuse_v2_dev20.yaml"
    )
    assert file_config.experiment.methods == (
        "dense_swt",
        "phasefuse",
        "phasefuse_v2",
    )
    assert file_config.experiment.uniform_reserve == 8
    assert file_config.metadata["main_method"] == "phasefuse_v2"


def test_phasefuse_v2_is_global_four_phase_median_and_keeps_v1_unchanged(
    tmp_path: Path,
):
    records = _records(tmp_path)
    config = PhaseFuseExperimentConfig(
        methods=("phasefuse", "phasefuse_v2"),
        frame_budget=8,
        min_frames_per_segment=2,
        uniform_reserve=4,
    )
    rows = run_phasefuse_experiment(records, tmp_path / "v2", config=config)
    by_method = {row["method"]: row for row in rows if row["origin_id"] == 0}

    v1_config = by_method["phasefuse"]["method_metadata"]["phasefuse"]["config"]
    v2 = by_method["phasefuse_v2"]
    v2_metadata = v2["method_metadata"]
    v2_config = v2_metadata["phasefuse"]["config"]

    assert v1_config["selection_strategy"] == "segmented"
    assert v1_config["uniform_reserve"] == 0
    assert v1_config["uncertainty_penalty"] == pytest.approx(0.25)
    assert v1_config["relevance_weight"] == pytest.approx(0.5)
    assert v1_config["phase_vote_weight"] == pytest.approx(0.25)

    assert v2_metadata["selector_kind"] == "multiphase_fusion"
    assert v2_metadata["selection_strategy"] == "global_coverage"
    assert v2["selection_strategy"] == "global_coverage"
    assert v2_config["num_phases"] == 4
    assert v2_config["consensus"] == "median"
    assert v2_config["selection_strategy"] == "global_coverage"
    assert v2_config["uniform_reserve"] == 4
    assert v2_config["selection_event_weight"] == pytest.approx(0.25)
    assert v2_config["component_scaling"] == "percentile"
    assert v2_config["min_selection_distance_sec"] == pytest.approx(0.5)
    assert v2_config["uncertainty_penalty"] == pytest.approx(0.0)
    assert v2_config["relevance_weight"] == pytest.approx(0.0)
    assert v2_config["phase_vote_weight"] == pytest.approx(0.0)
    assert len(v2_metadata["phasefuse"]["anchor_indices"]) == 4
    assert v2["peaks"] == []
    assert v2["allocation"] == {"0": 8}

    summary = json.loads((tmp_path / "v2" / "summary.json").read_text())
    assert summary["config"]["selection_strategy"] == "global_coverage"
    assert summary["config"]["uniform_reserve"] == 4

    with pytest.raises(ValueError, match="exactly four"):
        PhaseFuseExperimentConfig(
            methods=("phasefuse_v2",),
            num_phases=2,
        )


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
            "phasefuse_v2",
        ]
    )
    assert run.methods == ["dense_swt", "phasefuse", "phasefuse_v2"]
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
    rc12_downstream = parser.parse_args(
        [
            "evaluate-phasefuse-predictions",
            "--predictions",
            "predictions.jsonl",
            "--traces",
            "traces.jsonl",
            "--output",
            "summary.json",
            "--baseline-method",
            "canonical_uniform",
            "--treatment-method",
            "phasefuse_rc12",
            "--expected-methods",
            "canonical_uniform",
            "phasefuse_rc12",
        ]
    )
    assert rc12_downstream.expected_methods == [
        "canonical_uniform",
        "phasefuse_rc12",
    ]
