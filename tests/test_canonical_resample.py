from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from phase_stable.canonical_resample import (
    CanonicalArmRequest,
    CanonicalDecodeExhaustedError,
    CanonicalRequestPair,
    decode_canonical_request_batch,
    decode_canonical_targets,
)
from phase_stable.phasefuse_analysis import _validate_trace_row
from phase_stable.sampling import DecodedFrameMatch, nearest_pts_indices


class SyntheticVFRDecoder:
    def __init__(self, decoded_pts_sec: Sequence[float]):
        self.decoded_pts_sec = tuple(float(value) for value in decoded_pts_sec)
        self.calls: list[tuple[str, tuple[float, ...], int]] = []

    def __call__(self, video_path, targets, *, stream_index=0):
        requested = tuple(float(value) for value in targets)
        self.calls.append((str(video_path), requested, stream_index))
        indices = nearest_pts_indices(self.decoded_pts_sec, requested)
        return [
            DecodedFrameMatch(
                target_timestamp_sec=target,
                actual_pts_sec=self.decoded_pts_sec[index],
                decode_error_ms=1000.0 * abs(self.decoded_pts_sec[index] - target),
                pixel_hash=f"{index:064x}",
                rgb=np.full((2, 2, 3), index, dtype=np.uint8),
                decoded_frame_index=index,
            )
            for target, index in zip(requested, indices)
        ]


def _arm(
    role: str,
    primary,
    lattice,
    *,
    scores=None,
    method=None,
):
    lattice = tuple(float(value) for value in lattice)
    if scores is None:
        scores = np.zeros(len(lattice), dtype=float)
    return CanonicalArmRequest(
        method=method or role,
        role=role,
        primary_target_timestamps_sec=tuple(float(value) for value in primary),
        lattice_timestamps_sec=lattice,
        candidate_scores=tuple(float(value) for value in scores),
        primary_sources=tuple(f"{role}_primary_{index}" for index in range(16)),
        primary_ranks=tuple(range(16)),
        candidate_sources=tuple(
            f"{role}_candidate_{index}" for index in range(len(lattice))
        ),
        candidate_ranks=tuple(range(len(lattice))),
    )


def test_fresh_vfr_decode_uses_absolute_targets_and_earlier_midpoint_without_scout_remap():
    lattice = (0.0, 1.2, *map(float, range(2, 18)))
    rc12_primary = (1.2, *map(float, range(2, 17)))
    uniform_primary = (0.0, *map(float, range(2, 17)))
    decoder = SyntheticVFRDecoder((0.0, 0.4, *map(float, range(2, 18))))

    result = decode_canonical_targets(
        "source-vfr.mp4",
        rc12=_arm("rc12", rc12_primary, lattice),
        canonical_uniform=_arm("canonical_uniform", uniform_primary, lattice),
        decoder=decoder,
    )

    assert len(decoder.calls) == 1
    assert decoder.calls[0][1] == tuple(
        sorted(set(rc12_primary) | set(uniform_primary))
    )
    midpoint = next(
        item for item in result.arm("rc12").selected if item.target_sec == 1.2
    )
    assert midpoint.actual_pts_sec == pytest.approx(0.4)
    assert midpoint.abs_error_sec == pytest.approx(0.8)
    assert midpoint.decoded_frame_index == 1
    assert result.midpoint_tie_policy == "earlier_pts"

    rows = [
        result.trace_row(
            method,
            dataset="videomme",
            video_id="vfr",
            question_id="q0",
            origin_id=0,
            origin_sec=0.137,
        )
        for method in ("rc12", "canonical_uniform")
    ]
    assert rows[0]["timestamps_sec"] == rows[1]["timestamps_sec"]
    assert rows[0]["actual_pts_sec"] == rows[1]["actual_pts_sec"]
    assert rows[0]["source_frame_indices"] == rows[1]["source_frame_indices"]
    assert all(row["canonical_decode"]["fresh_source_decode"] for row in rows)
    assert all(row["canonical_decode"]["prohibit_scout_frame_remap"] for row in rows)
    for row in rows:
        _validate_trace_row(row)


def test_duplicate_winner_and_dynamic_backup_repair_restore_exact_k16():
    lattice = tuple(map(float, range(21)))
    primary = tuple(map(float, range(16)))
    scores = np.zeros(len(lattice), dtype=float)
    scores[17], scores[18], scores[19], scores[20] = 1.0, 0.9, 0.8, 0.7
    # Targets 0 and 1 tie at frame PTS .5.  The earlier target must survive.
    # Target 17 maps back to frame 15 and is rejected; target 18 maps to 20.
    decoder = SyntheticVFRDecoder((0.5, *map(float, range(2, 16)), 20.0))

    result = decode_canonical_targets(
        "duplicates.mp4",
        rc12=_arm("rc12", primary, lattice, scores=scores),
        canonical_uniform=_arm("canonical_uniform", primary, lattice),
        decoder=decoder,
    )
    rc12 = result.arm("rc12")

    assert len(rc12.selected) == 16
    assert len({item.decoded_frame_index for item in rc12.selected}) == 16
    assert rc12.initial_duplicate_rejection_count == 1
    assert rc12.repair_duplicate_rejection_count == 1
    assert rc12.repair_attempt_count == 2
    assert [item.target_sec for item in rc12.attempts if item.stage == "repair"] == [
        17.0,
        18.0,
    ]
    loser = next(item for item in rc12.attempts if item.target_sec == 1.0)
    assert loser.status == "rejected_duplicate"
    assert loser.duplicate_winner_target_sec == 0.0
    assert rc12.attempts[-1].repair_source == "dynamic_frozen_residual_priority"
    assert rc12.attempts[-1].candidate_score == pytest.approx(0.9)
    assert [item.target_sec for item in rc12.selected][-1] == 18.0


def test_repair_distance_relaxes_only_when_no_two_second_candidate_exists():
    lattice = tuple(np.arange(0.0, 8.0 + 0.5, 0.5))
    primary = lattice[:16]
    decoder = SyntheticVFRDecoder((0.25, *np.arange(1.0, 8.0 + 0.5, 0.5)))

    result = decode_canonical_targets(
        "relax.mp4",
        rc12=_arm("rc12", primary, lattice),
        canonical_uniform=_arm("canonical_uniform", primary, lattice),
        decoder=decoder,
    )

    for method in ("rc12", "canonical_uniform"):
        arm = result.arm(method)
        assert len(arm.selected) == 16
        assert arm.distance_relaxation_count == 1
        repair = next(item for item in arm.attempts if item.stage == "repair")
        assert repair.target_sec == 8.0
        assert repair.distance_relaxed
        assert repair.priority_nearest_distance_sec == pytest.approx(0.5)


def test_lattice_exhaustion_aborts_with_serializable_provenance():
    lattice = tuple(np.arange(0.0, 8.0, 0.5))
    decoder = SyntheticVFRDecoder((0.25, *np.arange(1.0, 8.0, 0.5)))

    with pytest.raises(CanonicalDecodeExhaustedError) as captured:
        decode_canonical_targets(
            "exhaust.mp4",
            rc12=_arm("rc12", lattice, lattice),
            canonical_uniform=_arm("canonical_uniform", lattice, lattice),
            decoder=decoder,
        )

    provenance = captured.value.provenance
    assert provenance["accepted_unique_frames"] == 15
    assert provenance["attempted_canonical_indices"] == list(range(16))
    assert provenance["prohibit_scout_frame_remap"]
    assert len(provenance["attempts"]) == 16


def test_video_batch_decodes_full_canonical_union_once_and_discards_rgb_cache():
    lattice = tuple(map(float, range(20)))
    first_primary = lattice[:16]
    second_primary = lattice[4:20]
    decoder = SyntheticVFRDecoder(lattice)
    pairs = (
        CanonicalRequestPair(
            "q0/o0",
            _arm("rc12", first_primary, lattice, method="rc12"),
            _arm(
                "canonical_uniform",
                second_primary,
                lattice,
                method="canonical_uniform",
            ),
        ),
        CanonicalRequestPair(
            "q1/o4",
            _arm("rc12", second_primary, lattice, method="rc12"),
            _arm(
                "canonical_uniform",
                first_primary,
                lattice,
                method="canonical_uniform",
            ),
        ),
    )

    batch = decode_canonical_request_batch("one-video.mp4", pairs, decoder=decoder)

    assert len(decoder.calls) == 1
    assert decoder.calls[0][1] == lattice
    assert tuple(batch.request_results) == ("q0/o0", "q1/o4")
    assert len(batch.decode_passes) == 1
    for request_id in ("q0/o0", "q1/o4"):
        result = batch.request(request_id)
        assert len(result.arm("rc12").selected) == 16
        assert len(result.arm("canonical_uniform").selected) == 16
        assert all(candidate.rgb.size == 0 for candidate in result.candidate_union)
        assert all(
            attempt.rgb.size == 0
            for arm in result.arms.values()
            for attempt in arm.attempts
        )
