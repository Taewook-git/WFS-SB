from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phase_stable.multiphase import (
    DenseOuterRecordPayload,
    build_dense_outer_record,
    build_multiphase_manifest,
)
from phase_stable.rc12_experiment import build_decision_specs, load_video_contracts
from scripts.decode_rc12_exact import build_request_pairs


def _records(tmp_path: Path):
    manifest = build_multiphase_manifest(
        "v1",
        40.0,
        num_phases=4,
        num_outer_origins=5,
        outer_origins_sec=(0.05, 0.15, 0.25, 0.35, 0.45),
    )
    records = []
    for origin in manifest.outer_origins:
        ts = np.asarray(origin.dense_target_timestamps_sec)
        records.append(
            build_dense_outer_record(
                manifest,
                dataset="videomme",
                question_id="q1",
                payload=DenseOuterRecordPayload(
                    outer_origin_id=origin.outer_origin_id,
                    actual_pts_sec=tuple(ts),
                    source_frame_indices=tuple(range(len(ts))),
                    relevance_scores=tuple((np.sin(ts / 4) + 1).tolist()),
                    metadata={"query": "q", "query_metadata": {}},
                ),
            )
        )
    return manifest, records


def test_decision_specs_are_targets_only_and_rectangular(tmp_path: Path):
    manifest, records = _records(tmp_path)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"placeholder")
    contract = {
        "v1": {
            "video_path": str(video),
            "duration_sec": 40.0,
            "support_start_sec": manifest.common_valid_support_sec[0],
            "support_stop_sec": manifest.common_valid_support_sec[1],
        }
    }
    rows = build_decision_specs(records, contract)
    assert len(rows) == 10
    assert {r["method"] for r in rows} == {"canonical_uniform", "phasefuse_rc12"}
    assert {r["origin_id"] for r in rows} == set(range(5))
    for row in rows:
        assert len(row["target_timestamps_sec"]) == 16
        assert row["target_timestamps_sec"] == sorted(row["target_timestamps_sec"])
        assert "actual_pts_sec" not in row
        assert "source_frame_indices" not in row
        assert row["origin_sec"] == records[row["origin_id"]].origin_sec
        assert row["record_metadata"]["query"] == "q"
        serialized_metadata = json.dumps(row["record_metadata"])
        assert "actual_pts" not in serialized_metadata
        assert "source_frame" not in serialized_metadata
        assert "dense_to_unique" not in serialized_metadata
        assert row["decoder_contract"]["fresh_source_decode"] is True
        assert row["decoder_contract"]["prohibit_scout_actual_pts_remap"] is True
    uniforms = [
        r["target_timestamps_sec"] for r in rows if r["method"] == "canonical_uniform"
    ]
    assert all(value == uniforms[0] for value in uniforms)


def test_load_video_contracts_requires_aligned_video_grid(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    catalog = tmp_path / "catalog.jsonl"
    manifest = tmp_path / "manifest.jsonl"
    catalog.write_text(json.dumps({"video_id": "v1", "video_path": str(video)}) + "\n")
    manifest.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "duration_sec": 4.0,
                "common_valid_support_sec": [0.5, 3.5],
            }
        )
        + "\n"
    )
    result = load_video_contracts(catalog, manifest)
    assert result["v1"]["support_start_sec"] == 0.5


def test_decode_adapter_requires_full_video_batch_and_one_path_contract(tmp_path: Path):
    manifest, records = _records(tmp_path)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"placeholder")
    contract = {
        "v1": {
            "video_path": str(video),
            "duration_sec": 40.0,
            "support_start_sec": manifest.common_valid_support_sec[0],
            "support_stop_sec": manifest.common_valid_support_sec[1],
        }
    }
    one_question = build_decision_specs(records, contract)
    with pytest.raises(ValueError, match="exactly 15 pairs"):
        build_request_pairs(one_question)

    full = []
    for question in ("q1", "q2", "q3"):
        full.extend({**row, "question_id": question} for row in one_question)
    pairs, _ = build_request_pairs(full)
    assert len(pairs[str(video.resolve())]) == 15

    corrupted = list(full)
    corrupted[0] = {**corrupted[0], "video_id": "aliased-video"}
    corrupted[1] = {**corrupted[1], "video_id": "aliased-video"}
    with pytest.raises(ValueError, match="multiple video IDs"):
        build_request_pairs(corrupted)
