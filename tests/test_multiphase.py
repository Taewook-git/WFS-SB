from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from phase_stable.artifacts import OriginSignalRecord
from phase_stable.multiphase import (
    DenseOuterRecordPayload,
    build_dense_decode_plan,
    build_dense_outer_record,
    build_dense_outer_records,
    build_multiphase_manifest,
    common_valid_support,
    deduplicate_source_pts,
    interpolate_no_extrapolation,
    split_dense_record,
    validate_dense_outer_record,
)


def _manifest(*, phase_order=(0, 1, 2, 3)):
    return build_multiphase_manifest(
        "video-1",
        8.0,
        base_sample_fps=1.0,
        num_phases=4,
        master_seed=17,
        num_outer_origins=2,
        outer_origins_sec=(0.1, 0.6),
        phase_order=phase_order,
    )


def _payload(manifest, origin_id: int) -> DenseOuterRecordPayload:
    targets = np.asarray(
        manifest.origin(origin_id).dense_target_timestamps_sec, dtype=float
    )
    # An irregular/off-grid target lattice may decode multiple neighboring
    # targets to one physical source frame.  Scores remain target-aligned.
    actual_pts = np.round(targets * 2.0) / 2.0
    source_indices = np.rint(actual_pts * 2.0).astype(int)
    return DenseOuterRecordPayload(
        outer_origin_id=origin_id,
        actual_pts_sec=tuple(actual_pts),
        source_frame_indices=tuple(source_indices),
        relevance_scores=tuple(np.sin(targets) + 1.0),
        pixel_hashes=tuple(f"frame-{index}" for index in source_indices),
        metadata={"query_metadata": {"gold": "A"}},
    )


def test_manifest_builds_one_physical_dense_lattice_and_interleaved_phases() -> None:
    manifest = _manifest()
    first = manifest.origin(0)

    assert manifest.dense_sample_fps == pytest.approx(4.0)
    assert manifest.dense_period_sec == pytest.approx(0.25)
    assert manifest.candidate_count_per_phase == 7
    assert manifest.dense_candidate_count == 28
    np.testing.assert_allclose(
        first.dense_target_timestamps_sec[:8],
        [0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6, 1.85],
    )
    assert first.phase_ids[:8] == (0, 1, 2, 3, 0, 1, 2, 3)
    np.testing.assert_allclose(first.phase_timestamps(2), np.arange(0.6, 7.0, 1.0))
    assert first.phase_dense_indices(2) == tuple(range(2, 28, 4))
    assert first.common_valid_support_sec == pytest.approx((0.85, 6.1))
    assert manifest.common_valid_support_sec == pytest.approx((1.35, 6.1))
    assert all(
        timestamp <= manifest.duration_sec - manifest.epsilon_sec
        for origin in manifest.outer_origins
        for timestamp in origin.dense_target_timestamps_sec
    )


def test_phase_label_permutation_does_not_change_physical_lattice() -> None:
    canonical = _manifest(phase_order=(0, 1, 2, 3))
    permuted = _manifest(phase_order=(2, 0, 3, 1))

    for canonical_origin, permuted_origin in zip(
        canonical.outer_origins, permuted.outer_origins
    ):
        assert (
            canonical_origin.dense_target_timestamps_sec
            == permuted_origin.dense_target_timestamps_sec
        )
        canonical_streams = sorted(
            canonical_origin.phase_timestamps(phase_id)
            for phase_id in range(canonical.num_phases)
        )
        permuted_streams = sorted(
            permuted_origin.phase_timestamps(phase_id)
            for phase_id in range(permuted.num_phases)
        )
        assert canonical_streams == permuted_streams
        assert canonical_origin.phase_timestamps(0) == permuted_origin.phase_timestamps(2)


def test_generated_outer_origins_are_deterministic_and_use_base_period() -> None:
    first = build_multiphase_manifest(
        "deterministic", 12.0, base_sample_fps=2.0, num_phases=3, master_seed=9
    )
    second = build_multiphase_manifest(
        "deterministic", 12.0, base_sample_fps=2.0, num_phases=3, master_seed=9
    )

    assert first == second
    assert all(0 <= origin.outer_origin_sec < 0.5 for origin in first.outer_origins)
    assert first.dense_sample_fps == pytest.approx(6.0)
    assert all(
        len(origin.phase_timestamps(phase_id)) == first.candidate_count_per_phase
        for origin in first.outer_origins
        for phase_id in range(first.num_phases)
    )


def test_dense_decode_plan_flattens_one_video_once_and_splits_in_outer_order() -> None:
    manifest = _manifest()
    plan = build_dense_decode_plan(manifest)

    assert plan.outer_origin_ids == (0, 1)
    assert plan.outer_slices == ((0, 28), (28, 56))
    assert plan.target_timestamps_sec == tuple(
        timestamp
        for origin in manifest.outer_origins
        for timestamp in origin.dense_target_timestamps_sec
    )
    synthetic_matches = tuple(f"match-{index}" for index in range(56))
    split = plan.split_aligned(synthetic_matches)
    assert split[0] == synthetic_matches[:28]
    assert split[1] == synthetic_matches[28:]
    with pytest.raises(ValueError, match="every decode-plan target"):
        plan.split_aligned(synthetic_matches[:-1])


def test_source_pts_dedup_and_scatter_are_lossless() -> None:
    scatter = deduplicate_source_pts(
        (0.0, 0.5, 0.5, 1.0, 1.0, 1.5),
        (0, 1, 1, 2, 2, 3),
    )

    assert scatter.unique_dense_indices == (0, 1, 3, 5)
    assert scatter.unique_actual_pts_sec == (0.0, 0.5, 1.0, 1.5)
    assert scatter.unique_source_frame_indices == (0, 1, 2, 3)
    assert scatter.dense_to_unique == (0, 1, 1, 2, 2, 3)
    unique_features = np.asarray([[0.0], [10.0], [20.0], [30.0]])
    dense_features = scatter.scatter_unique(unique_features)
    np.testing.assert_array_equal(
        dense_features[:, 0], [0.0, 10.0, 10.0, 20.0, 20.0, 30.0]
    )
    np.testing.assert_array_equal(scatter.gather_unique(dense_features), unique_features)

    inconsistent = dense_features.copy()
    inconsistent[2, 0] = 11.0
    with pytest.raises(ValueError, match="duplicate-frame values"):
        scatter.gather_unique(inconsistent)


@pytest.mark.parametrize(
    ("pts", "frames", "message"),
    [
        ((0.0, 0.0), (0, 1), "PTS maps to multiple"),
        ((0.0, 0.5), (0, 0), "frame index maps to multiple"),
        ((0.5, 0.0), (1, 0), "non-decreasing"),
    ],
)
def test_source_pts_dedup_rejects_ambiguous_physical_identity(
    pts, frames, message
) -> None:
    with pytest.raises(ValueError, match=message):
        deduplicate_source_pts(pts, frames)


def test_dense_record_contract_and_split_preserve_absolute_time() -> None:
    manifest = _manifest(phase_order=(2, 0, 3, 1))
    payload = _payload(manifest, 0)
    record = build_dense_outer_record(
        manifest,
        dataset="qvhighlights",
        question_id="query-1",
        payload=payload,
        metadata={"experiment": "synthetic"},
    )

    contract = record.metadata["multiphase"]
    assert record.metadata["sample_fps"] == pytest.approx(4.0)
    assert contract["phase_ids"] == list(manifest.origin(0).phase_ids)
    assert contract["source_pts_dense_to_unique"] == list(
        validate_dense_outer_record(record, manifest).dense_to_unique
    )
    assert contract["no_extrapolation"] is True
    assert contract["visual_features_layout"] == "dense_target_aligned"
    assert contract["visual_features_rows"] == manifest.dense_candidate_count

    streams = split_dense_record(record, manifest)
    assert [stream.phase_id for stream in streams] == [0, 1, 2, 3]
    assert [stream.phase_slot for stream in streams] == [1, 3, 0, 2]
    for stream in streams:
        origin = manifest.origin(0)
        assert stream.dense_indices == origin.phase_dense_indices(stream.phase_id)
        assert stream.timestamps_sec == origin.phase_timestamps(stream.phase_id)
        np.testing.assert_allclose(
            stream.relevance_scores,
            np.asarray(record.relevance_scores)[list(stream.dense_indices)],
        )
        assert stream.valid_dense_indices
        assert all(
            stream.common_valid_support_sec[0] - 1e-12
            <= record.timestamps_sec[index]
            <= stream.common_valid_support_sec[1] + 1e-12
            for index in stream.valid_dense_indices
        )


def test_phase_permutation_split_reconstructs_same_dense_physical_signal() -> None:
    reconstructed = []
    for order in ((0, 1, 2, 3), (2, 0, 3, 1)):
        manifest = _manifest(phase_order=order)
        record = build_dense_outer_record(
            manifest,
            dataset="demo",
            question_id="q",
            payload=_payload(manifest, 0),
        )
        restored = np.empty(len(record.relevance_scores), dtype=float)
        for stream in split_dense_record(record, manifest):
            restored[list(stream.dense_indices)] = stream.relevance_scores
        reconstructed.append(restored)

    np.testing.assert_allclose(reconstructed[0], reconstructed[1])


def test_common_support_and_off_grid_interpolation_never_extrapolate() -> None:
    first = (0.13, 1.13, 2.13)
    second = (0.38, 1.38, 2.38)
    support = common_valid_support((first, second))

    assert support == pytest.approx((0.38, 2.13))
    result = interpolate_no_extrapolation(
        first,
        np.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]]),
        (0.38, 1.38, 2.13),
    )
    np.testing.assert_allclose(
        result,
        [[0.25, 0.5], [1.25, 2.5], [2.0, 4.0]],
    )
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_no_extrapolation(first, (0.0, 1.0, 2.0), (0.12, 1.0))
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_no_extrapolation(first, (0.0, 1.0, 2.0), (1.0, 2.14))


def test_dense_record_validation_rejects_contract_tampering() -> None:
    manifest = _manifest()
    record = build_dense_outer_record(
        manifest,
        dataset="demo",
        question_id="q",
        payload=_payload(manifest, 0),
    )
    tampered_metadata = dict(record.metadata)
    tampered_contract = dict(tampered_metadata["multiphase"])
    tampered_contract["phase_ids"] = list(reversed(tampered_contract["phase_ids"]))
    tampered_metadata["multiphase"] = tampered_contract

    with pytest.raises(ValueError, match="phase_ids.*mismatch"):
        split_dense_record(replace(record, metadata=tampered_metadata), manifest)

    bad_rate_metadata = dict(record.metadata)
    bad_rate_metadata["sample_fps"] = 1.0
    with pytest.raises(ValueError, match="sample_fps mismatch"):
        validate_dense_outer_record(
            replace(record, metadata=bad_rate_metadata), manifest
        )


def test_dense_record_validation_rejects_grid_and_duration_extrapolation() -> None:
    manifest = _manifest()
    record = build_dense_outer_record(
        manifest,
        dataset="demo",
        question_id="q",
        payload=_payload(manifest, 0),
    )
    shifted_targets = tuple(value + 0.01 for value in record.timestamps_sec)
    with pytest.raises(ValueError, match="physical dense lattice"):
        validate_dense_outer_record(
            replace(record, timestamps_sec=shifted_targets), manifest
        )

    bad_payload = replace(
        _payload(manifest, 0),
        actual_pts_sec=tuple(
            list(_payload(manifest, 0).actual_pts_sec[:-1])
            + [manifest.duration_sec + 0.5]
        ),
        source_frame_indices=tuple(
            list(_payload(manifest, 0).source_frame_indices[:-1])
            + [_payload(manifest, 0).source_frame_indices[-1] + 1]
        ),
    )
    with pytest.raises(ValueError, match="exceeds the manifest duration"):
        build_dense_outer_record(
            manifest,
            dataset="demo",
            question_id="q",
            payload=bad_payload,
        )


def test_plural_builder_requires_exact_outer_origin_coverage() -> None:
    manifest = _manifest()
    records = build_dense_outer_records(
        manifest,
        dataset="demo",
        question_id="q",
        payloads=(_payload(manifest, 1), _payload(manifest, 0)),
    )

    assert [record.origin_id for record in records] == [0, 1]
    assert all(record.metadata["sample_fps"] == 4.0 for record in records)
    with pytest.raises(ValueError, match="cover every manifest outer origin"):
        build_dense_outer_records(
            manifest,
            dataset="demo",
            question_id="q",
            payloads=(_payload(manifest, 0),),
        )
    with pytest.raises(ValueError, match="duplicate dense payload"):
        build_dense_outer_records(
            manifest,
            dataset="demo",
            question_id="q",
            payloads=(_payload(manifest, 0), _payload(manifest, 0)),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_phases": 1},
        {"base_sample_fps": 0.0},
        {"outer_origins_sec": (0.2, 0.1), "num_outer_origins": 2},
        {"outer_origins_sec": (0.1, 1.0), "num_outer_origins": 2},
        {"phase_order": (0, 0, 1, 2)},
    ],
)
def test_manifest_rejects_invalid_phase_and_outer_origin_contract(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_multiphase_manifest("v", 8.0, **kwargs)


def test_manifest_rejects_too_short_video_instead_of_extrapolating() -> None:
    with pytest.raises(ValueError, match="fewer than two samples per phase"):
        build_multiphase_manifest(
            "short",
            1.2,
            num_phases=4,
            num_outer_origins=2,
            outer_origins_sec=(0.1, 0.6),
        )


def test_split_requires_origin_record_not_uncontracted_dense_arrays() -> None:
    manifest = _manifest()
    origin = manifest.origin(0)
    targets = origin.dense_target_timestamps_sec
    raw = OriginSignalRecord(
        dataset="demo",
        video_id=manifest.video_id,
        question_id="q",
        origin_id=0,
        origin_sec=origin.outer_origin_sec,
        timestamps_sec=targets,
        actual_pts_sec=targets,
        source_frame_indices=tuple(range(len(targets))),
        relevance_scores=tuple(np.ones(len(targets))),
    )

    with pytest.raises(ValueError, match="sample_fps mismatch"):
        split_dense_record(raw, manifest)
