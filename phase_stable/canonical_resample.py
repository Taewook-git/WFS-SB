"""Fresh canonical second-stage decoding for the frozen RC12 comparison.

The scout pass is deliberately absent from this module's API.  Callers provide
physical-time targets on the shared canonical lattice, and this module decodes
the source video again with the sequential nearest-PTS decoder.  This prevents
an accidental remap to already-decoded, origin-dependent scout frames.

The two confirmatory arms are decoded against a common union of targets.  Any
targets added while repairing decoded-frame collisions are also added to that
union, so both returned trace rows expose exactly the same candidate arrays.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .sampling import DecodedFrameMatch, iter_nearest_frames_pyav_indexed

CANONICAL_RESAMPLE_SCHEMA_VERSION = 1
FRAME_BUDGET = 16
REPAIR_MIN_DISTANCE_SEC = 2.0
REPAIR_COVERAGE_CAP_SEC = 8.0
MIDPOINT_TIE_POLICY = "earlier_pts"
DEFAULT_DECODER_BACKEND = "pyav_sequential_nearest_pts"
_TIME_TOLERANCE_SEC = 1e-9
_DUPLICATE_LOSS_STATUSES = frozenset({"rejected_duplicate", "replaced_by_lower_error"})


def _finite_tuple(name: str, values: Sequence[Real]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of real numbers") from exc
    result: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name}[{index}] must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _text_tuple(
    name: str,
    values: Sequence[str],
    *,
    expected: int,
    default: str,
) -> tuple[str, ...]:
    if not values:
        return (default,) * expected
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of strings") from exc
    if len(result) != expected:
        raise ValueError(f"{name} must align with its target array")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


def _rank_tuple(
    name: str,
    values: Sequence[int],
    *,
    expected: int,
) -> tuple[int, ...]:
    if not values:
        return tuple(range(expected))
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of integers") from exc
    if len(raw) != expected:
        raise ValueError(f"{name} must align with its target array")
    result: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name}[{index}] must be an integer")
        if int(value) < 0:
            raise ValueError(f"{name}[{index}] must be non-negative")
        result.append(int(value))
    return tuple(result)


def _strictly_increasing(name: str, values: Sequence[float]) -> None:
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing")


def _canonical_indices(
    targets: Sequence[float], lattice: Sequence[float]
) -> tuple[int, ...]:
    lattice_array = np.asarray(lattice, dtype=np.float64)
    result: list[int] = []
    for target in targets:
        insertion = int(np.searchsorted(lattice_array, target, side="left"))
        candidates = [
            index
            for index in (insertion - 1, insertion)
            if 0 <= index < lattice_array.size
            and abs(float(lattice_array[index]) - target) <= _TIME_TOLERANCE_SEC
        ]
        if not candidates:
            raise ValueError(
                f"primary target {target!r} is not a canonical lattice tick"
            )
        result.append(min(candidates, key=lambda index: abs(lattice[index] - target)))
    if len(set(result)) != len(result):
        raise ValueError("primary targets must identify distinct canonical ticks")
    return tuple(result)


@dataclass(frozen=True)
class CanonicalArmRequest:
    """One frozen arm's primary decision and complete repair lattice.

    ``candidate_scores`` are the frozen quantized RC12 scores.  The canonical
    uniform control must supply zeros, which makes the shared repair rule use
    coverage and then earlier time only.
    """

    method: str
    role: str
    primary_target_timestamps_sec: tuple[float, ...]
    lattice_timestamps_sec: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    primary_sources: tuple[str, ...] = ()
    primary_ranks: tuple[int, ...] = ()
    candidate_sources: tuple[str, ...] = ()
    candidate_ranks: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if self.role not in {
            "rc12",
            "rc14",
            "nested_r2",
            "canonical_uniform",
        }:
            raise ValueError(
                "role must be 'rc12', 'rc14', 'nested_r2', or 'canonical_uniform'"
            )
        primary = _finite_tuple(
            "primary_target_timestamps_sec", self.primary_target_timestamps_sec
        )
        lattice = _finite_tuple("lattice_timestamps_sec", self.lattice_timestamps_sec)
        scores = _finite_tuple("candidate_scores", self.candidate_scores)
        if len(primary) != FRAME_BUDGET:
            raise ValueError(f"every canonical arm must have exactly K={FRAME_BUDGET}")
        if len(lattice) < FRAME_BUDGET:
            raise ValueError(
                "canonical lattice cannot be smaller than the frame budget"
            )
        if len(scores) != len(lattice):
            raise ValueError("candidate_scores must align with lattice_timestamps_sec")
        if primary[0] < 0.0 or lattice[0] < 0.0:
            raise ValueError("canonical timestamps must be non-negative")
        _strictly_increasing("primary_target_timestamps_sec", primary)
        _strictly_increasing("lattice_timestamps_sec", lattice)
        _canonical_indices(primary, lattice)
        if self.role == "canonical_uniform" and any(score != 0.0 for score in scores):
            raise ValueError("canonical_uniform candidate_scores must all be zero")

        primary_sources = _text_tuple(
            "primary_sources",
            self.primary_sources,
            expected=FRAME_BUDGET,
            default=f"{self.role}_primary",
        )
        primary_ranks = _rank_tuple(
            "primary_ranks", self.primary_ranks, expected=FRAME_BUDGET
        )
        candidate_sources = _text_tuple(
            "candidate_sources",
            self.candidate_sources,
            expected=len(lattice),
            default=f"{self.role}_canonical_candidate",
        )
        candidate_ranks = _rank_tuple(
            "candidate_ranks", self.candidate_ranks, expected=len(lattice)
        )
        object.__setattr__(self, "primary_target_timestamps_sec", primary)
        object.__setattr__(self, "lattice_timestamps_sec", lattice)
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "primary_sources", primary_sources)
        object.__setattr__(self, "primary_ranks", primary_ranks)
        object.__setattr__(self, "candidate_sources", candidate_sources)
        object.__setattr__(self, "candidate_ranks", candidate_ranks)

    @property
    def primary_canonical_indices(self) -> tuple[int, ...]:
        return _canonical_indices(
            self.primary_target_timestamps_sec, self.lattice_timestamps_sec
        )


@dataclass(frozen=True)
class DecodePass:
    pass_index: int
    target_timestamps_sec: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "target_timestamps_sec": list(self.target_timestamps_sec),
        }


@dataclass(frozen=True)
class CanonicalTargetAttempt:
    """One primary or repair target with complete fresh-decode provenance."""

    method: str
    role: str
    attempt_index: int
    stage: str
    repair_round: int
    repair_source: str | None
    target_source: str
    canonical_index: int
    candidate_rank: int
    candidate_score: float
    target_sec: float
    actual_pts_sec: float
    abs_error_sec: float
    decoded_frame_index: int
    pixel_hash: str
    decode_pass_index: int
    distance_relaxed: bool
    priority_nearest_distance_sec: float | None
    priority_capped_distance_sec: float | None
    status: str
    duplicate_winner_target_sec: float | None = None
    duplicate_resolution_stage: str | None = None
    rgb: np.ndarray = field(
        repr=False, compare=False, default_factory=lambda: np.empty(0)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "role": self.role,
            "attempt_index": self.attempt_index,
            "stage": self.stage,
            "repair_round": self.repair_round,
            "repair_source": self.repair_source,
            "target_source": self.target_source,
            "canonical_index": self.canonical_index,
            "candidate_rank": self.candidate_rank,
            "candidate_score": self.candidate_score,
            "target_sec": self.target_sec,
            "actual_pts_sec": self.actual_pts_sec,
            "abs_error_sec": self.abs_error_sec,
            "decode_error_ms": 1000.0 * self.abs_error_sec,
            "decoded_frame_index": self.decoded_frame_index,
            "pixel_hash": self.pixel_hash,
            "decode_pass_index": self.decode_pass_index,
            "midpoint_tie_policy": MIDPOINT_TIE_POLICY,
            "distance_relaxed": self.distance_relaxed,
            "priority_nearest_distance_sec": self.priority_nearest_distance_sec,
            "priority_capped_distance_sec": self.priority_capped_distance_sec,
            "status": self.status,
            "duplicate_winner_target_sec": self.duplicate_winner_target_sec,
            "duplicate_resolution_stage": self.duplicate_resolution_stage,
        }


@dataclass(frozen=True)
class CanonicalArmResult:
    method: str
    role: str
    attempts: tuple[CanonicalTargetAttempt, ...]
    selected: tuple[CanonicalTargetAttempt, ...]
    repair_attempt_count: int
    duplicate_rejection_count: int
    initial_duplicate_rejection_count: int
    repair_duplicate_rejection_count: int
    distance_relaxation_count: int
    duplicate_replacement_count: int = 0

    def __post_init__(self) -> None:
        if len(self.selected) != FRAME_BUDGET:
            raise ValueError(
                f"canonical arm result must contain exactly K={FRAME_BUDGET}"
            )
        frames = tuple(item.decoded_frame_index for item in self.selected)
        if len(set(frames)) != FRAME_BUDGET:
            raise ValueError("canonical arm result contains duplicate decoded frames")
        if any(
            right.target_sec <= left.target_sec
            for left, right in zip(self.selected, self.selected[1:])
        ):
            raise ValueError("selected canonical targets must be strictly increasing")
        if any(item.status != "selected" for item in self.selected):
            raise ValueError("every final selected attempt must have status='selected'")
        selected_attempts = tuple(
            sorted(
                (item for item in self.attempts if item.status == "selected"),
                key=lambda item: item.target_sec,
            )
        )
        if selected_attempts != self.selected:
            raise ValueError(
                "final selected attempts must exactly match selected-status provenance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "role": self.role,
            "frame_budget": FRAME_BUDGET,
            "repair_attempt_count": self.repair_attempt_count,
            "duplicate_rejection_count": self.duplicate_rejection_count,
            "initial_duplicate_rejection_count": self.initial_duplicate_rejection_count,
            "repair_duplicate_rejection_count": self.repair_duplicate_rejection_count,
            "duplicate_replacement_count": self.duplicate_replacement_count,
            "distance_relaxation_count": self.distance_relaxation_count,
            "selected_target_timestamps_sec": [
                item.target_sec for item in self.selected
            ],
            "selected_actual_pts_sec": [item.actual_pts_sec for item in self.selected],
            "selected_source_frame_indices": [
                item.decoded_frame_index for item in self.selected
            ],
            "attempts": [item.to_dict() for item in self.attempts],
        }


@dataclass(frozen=True)
class CandidateUnionEntry:
    target_sec: float
    actual_pts_sec: float
    abs_error_sec: float
    decoded_frame_index: int
    pixel_hash: str
    decode_pass_index: int
    attempted_by: tuple[str, ...]
    rgb: np.ndarray = field(
        repr=False, compare=False, default_factory=lambda: np.empty(0)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_sec": self.target_sec,
            "actual_pts_sec": self.actual_pts_sec,
            "abs_error_sec": self.abs_error_sec,
            "decode_error_ms": 1000.0 * self.abs_error_sec,
            "decoded_frame_index": self.decoded_frame_index,
            "pixel_hash": self.pixel_hash,
            "decode_pass_index": self.decode_pass_index,
            "attempted_by": list(self.attempted_by),
            "midpoint_tie_policy": MIDPOINT_TIE_POLICY,
        }


class CanonicalDecodeExhaustedError(RuntimeError):
    """Raised when the canonical lattice cannot yield K unique source frames."""

    def __init__(self, method: str, provenance: Mapping[str, Any]):
        self.method = method
        self.provenance = dict(provenance)
        super().__init__(
            f"canonical decode exhausted the lattice before K={FRAME_BUDGET} "
            f"unique frames for method={method!r}"
        )


@dataclass(frozen=True)
class CanonicalDecodeResult:
    video_path: str
    decoder_backend: str
    midpoint_tie_policy: str
    decode_passes: tuple[DecodePass, ...]
    candidate_union: tuple[CandidateUnionEntry, ...]
    arms: Mapping[str, CanonicalArmResult]

    def arm(self, method: str) -> CanonicalArmResult:
        try:
            return self.arms[method]
        except KeyError as exc:
            raise KeyError(f"unknown canonical arm {method!r}") from exc

    def trace_row(
        self,
        method: str,
        *,
        dataset: str,
        video_id: str,
        question_id: str,
        origin_id: int,
        origin_sec: float,
        record_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one strict-analysis/export row on the shared candidate union."""

        arm = self.arm(method)
        candidate_index = {
            candidate.target_sec: index
            for index, candidate in enumerate(self.candidate_union)
        }
        selected_indices = sorted(
            candidate_index[item.target_sec] for item in arm.selected
        )
        selected = [self.candidate_union[index] for index in selected_indices]
        all_attempts = [
            attempt
            for candidate in self.candidate_union
            for other_arm in self.arms.values()
            for attempt in other_arm.attempts
            if attempt.target_sec == candidate.target_sec
        ]
        attempts_by_target: dict[float, list[dict[str, Any]]] = {
            candidate.target_sec: [] for candidate in self.candidate_union
        }
        for attempt in all_attempts:
            attempts_by_target[attempt.target_sec].append(attempt.to_dict())
        candidate_provenance: list[dict[str, Any]] = []
        for candidate in self.candidate_union:
            payload = candidate.to_dict()
            payload["arm_attempts"] = attempts_by_target[candidate.target_sec]
            candidate_provenance.append(payload)

        return {
            "schema_version": CANONICAL_RESAMPLE_SCHEMA_VERSION,
            "dataset": dataset,
            "video_id": video_id,
            "question_id": question_id,
            "origin_id": origin_id,
            "origin_sec": float(origin_sec),
            "method": method,
            "timestamps_sec": [item.target_sec for item in self.candidate_union],
            "actual_pts_sec": [item.actual_pts_sec for item in self.candidate_union],
            "source_frame_indices": [
                item.decoded_frame_index for item in self.candidate_union
            ],
            "pixel_hashes": [item.pixel_hash for item in self.candidate_union],
            "selected_indices": selected_indices,
            "selected_timestamps_sec": [item.target_sec for item in selected],
            "selected_actual_pts_sec": [item.actual_pts_sec for item in selected],
            "selected_source_frame_indices": [
                item.decoded_frame_index for item in selected
            ],
            "selected_pixel_hashes": [item.pixel_hash for item in selected],
            "record_metadata": dict(record_metadata or {}),
            "candidate_provenance": candidate_provenance,
            "canonical_decode": {
                "schema_version": CANONICAL_RESAMPLE_SCHEMA_VERSION,
                "fresh_source_decode": True,
                "prohibit_scout_frame_remap": True,
                "decoder_backend": self.decoder_backend,
                "midpoint_tie_policy": self.midpoint_tie_policy,
                "frame_budget": FRAME_BUDGET,
                "candidate_union_size": len(self.candidate_union),
                "decode_pass_count": len(self.decode_passes),
                "decode_passes": [item.to_dict() for item in self.decode_passes],
                "repair_attempt_count": arm.repair_attempt_count,
                "duplicate_rejection_count": arm.duplicate_rejection_count,
                "initial_duplicate_rejection_count": (
                    arm.initial_duplicate_rejection_count
                ),
                "repair_duplicate_rejection_count": (
                    arm.repair_duplicate_rejection_count
                ),
                "duplicate_replacement_count": arm.duplicate_replacement_count,
                "distance_relaxation_count": arm.distance_relaxation_count,
                "attempts": [item.to_dict() for item in arm.attempts],
            },
            "used_fallback": bool(arm.repair_attempt_count),
            "fallback_reason": (
                "decoded_frame_duplicate_repair" if arm.repair_attempt_count else None
            ),
        }


@dataclass(frozen=True)
class CanonicalRequestPair:
    """One item/origin comparison to be served by a video-level decode pass."""

    request_id: str
    rc12: CanonicalArmRequest
    canonical_uniform: CanonicalArmRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.rc12, CanonicalArmRequest) or not isinstance(
            self.canonical_uniform, CanonicalArmRequest
        ):
            raise TypeError(
                "canonical request pairs require CanonicalArmRequest values"
            )
        if (
            self.rc12.role not in {"rc12", "rc14", "nested_r2"}
            or self.canonical_uniform.role != "canonical_uniform"
        ):
            raise ValueError(
                "request-pair roles must be a canonical residual and canonical_uniform"
            )
        if self.rc12.method == self.canonical_uniform.method:
            raise ValueError("request-pair method names must be distinct")
        if (
            self.rc12.lattice_timestamps_sec
            != self.canonical_uniform.lattice_timestamps_sec
        ):
            raise ValueError("paired arms must use the exact same canonical lattice")


@dataclass(frozen=True)
class CanonicalDecodeBatchResult:
    """Video-level results sharing primary-union and optional repair passes."""

    video_path: str
    decoder_backend: str
    midpoint_tie_policy: str
    decode_passes: tuple[DecodePass, ...]
    request_results: Mapping[str, CanonicalDecodeResult]

    def request(self, request_id: str) -> CanonicalDecodeResult:
        try:
            return self.request_results[request_id]
        except KeyError as exc:
            raise KeyError(f"unknown canonical request {request_id!r}") from exc


Decoder = Callable[..., Sequence[DecodedFrameMatch]]


@dataclass
class _ArmState:
    request: CanonicalArmRequest
    attempts: list[CanonicalTargetAttempt] = field(default_factory=list)
    accepted_by_frame: dict[int, CanonicalTargetAttempt] = field(default_factory=dict)
    attempted_canonical_indices: set[int] = field(default_factory=set)
    repair_round: int = 0
    distance_relaxation_count: int = 0


def _decode_batch(
    video_path: str | Path,
    targets: Sequence[float],
    *,
    stream_index: int,
    decoder: Decoder | None,
    pass_index: int,
) -> tuple[dict[float, tuple[DecodedFrameMatch, int]], DecodePass]:
    ordered = tuple(sorted({float(value) for value in targets}))
    if not ordered:
        raise ValueError("internal error: a decode batch cannot be empty")
    if decoder is None:
        decoded_slots: list[DecodedFrameMatch | None] = [None] * len(ordered)
        for target_index, match in iter_nearest_frames_pyav_indexed(
            video_path, ordered, stream_index=stream_index
        ):
            if target_index < 0 or target_index >= len(ordered):
                raise RuntimeError("decoder returned an invalid target index")
            if decoded_slots[target_index] is not None:
                raise RuntimeError("decoder returned one target more than once")
            # Compact immediately: the iterator materializes one RGB frame for
            # its pixel hash, and holding that array until a long-video pass
            # completes would defeat the video-level batching contract.
            decoded_slots[target_index] = DecodedFrameMatch(
                target_timestamp_sec=float(match.target_timestamp_sec),
                actual_pts_sec=float(match.actual_pts_sec),
                decode_error_ms=float(match.decode_error_ms),
                pixel_hash=str(match.pixel_hash),
                rgb=np.empty((0,), dtype=np.uint8),
                decoded_frame_index=int(match.decoded_frame_index),
            )
        if any(match is None for match in decoded_slots):
            raise RuntimeError("decoder did not return every requested target")
        decoded = tuple(match for match in decoded_slots if match is not None)
    else:
        decoded = tuple(decoder(video_path, ordered, stream_index=stream_index))
        if len(decoded) != len(ordered):
            raise RuntimeError(
                f"decoder returned {len(decoded)} matches for {len(ordered)} targets"
            )
    result: dict[float, tuple[DecodedFrameMatch, int]] = {}
    prior_actual: float | None = None
    prior_frame: int | None = None
    for expected, match in zip(ordered, decoded):
        if not isinstance(match, DecodedFrameMatch):
            raise TypeError("decoder must return DecodedFrameMatch values")
        if abs(float(match.target_timestamp_sec) - expected) > _TIME_TOLERANCE_SEC:
            raise RuntimeError(
                "decoder result does not align with requested target order"
            )
        actual = float(match.actual_pts_sec)
        if not math.isfinite(actual):
            raise RuntimeError("decoder returned a non-finite actual PTS")
        frame_index = int(match.decoded_frame_index)
        if frame_index < 0:
            raise RuntimeError("decoder omitted decoded_frame_index provenance")
        if prior_actual is not None and actual < prior_actual - _TIME_TOLERANCE_SEC:
            raise RuntimeError("decoder returned non-monotonic nearest PTS matches")
        if prior_frame is not None and frame_index < prior_frame:
            raise RuntimeError("decoder returned non-monotonic decoded frame indices")
        prior_actual = actual
        prior_frame = frame_index
        # Retain exact provenance and the pixel hash, but discard RGB after
        # each target. Downstream export re-opens only the selected decoded
        # frame indices, avoiding a batch-size-dependent memory spike.
        compact_match = DecodedFrameMatch(
            target_timestamp_sec=float(match.target_timestamp_sec),
            actual_pts_sec=actual,
            decode_error_ms=float(match.decode_error_ms),
            pixel_hash=str(match.pixel_hash),
            rgb=np.empty((0,), dtype=np.uint8),
            decoded_frame_index=frame_index,
        )
        result[expected] = (compact_match, pass_index)
    return result, DecodePass(pass_index=pass_index, target_timestamps_sec=ordered)


def _attempt(
    state: _ArmState,
    *,
    canonical_index: int,
    target_source: str,
    candidate_rank: int,
    stage: str,
    repair_round: int,
    distance_relaxed: bool,
    nearest_distance: float | None,
    match: DecodedFrameMatch,
    decode_pass_index: int,
    status: str,
    duplicate_winner_target_sec: float | None = None,
    duplicate_resolution_stage: str | None = None,
) -> CanonicalTargetAttempt:
    request = state.request
    target = float(request.lattice_timestamps_sec[canonical_index])
    actual = float(match.actual_pts_sec)
    return CanonicalTargetAttempt(
        method=request.method,
        role=request.role,
        attempt_index=len(state.attempts),
        stage=stage,
        repair_round=repair_round,
        repair_source=(
            None if stage == "primary" else "dynamic_frozen_residual_priority"
        ),
        target_source=target_source,
        canonical_index=canonical_index,
        candidate_rank=candidate_rank,
        candidate_score=float(request.candidate_scores[canonical_index]),
        target_sec=target,
        actual_pts_sec=actual,
        abs_error_sec=abs(actual - target),
        decoded_frame_index=int(match.decoded_frame_index),
        pixel_hash=str(match.pixel_hash),
        decode_pass_index=decode_pass_index,
        distance_relaxed=distance_relaxed,
        priority_nearest_distance_sec=nearest_distance,
        priority_capped_distance_sec=(
            None
            if nearest_distance is None
            else min(nearest_distance, REPAIR_COVERAGE_CAP_SEC)
        ),
        status=status,
        duplicate_winner_target_sec=duplicate_winner_target_sec,
        duplicate_resolution_stage=duplicate_resolution_stage,
        rgb=match.rgb,
    )


def _initialize_arm(
    request: CanonicalArmRequest,
    decoded: Mapping[float, tuple[DecodedFrameMatch, int]],
) -> _ArmState:
    state = _ArmState(request=request)
    indices = request.primary_canonical_indices
    provisional: list[tuple[int, int, DecodedFrameMatch, int]] = []
    for primary_position, (target, canonical_index) in enumerate(
        zip(request.primary_target_timestamps_sec, indices)
    ):
        match, pass_index = decoded[target]
        provisional.append((primary_position, canonical_index, match, pass_index))
        state.attempted_canonical_indices.add(canonical_index)

    winner_by_frame: dict[int, tuple[int, int, DecodedFrameMatch, int]] = {}
    for item in provisional:
        primary_position, canonical_index, match, _ = item
        frame_index = int(match.decoded_frame_index)
        target = request.lattice_timestamps_sec[canonical_index]
        error = abs(float(match.actual_pts_sec) - target)
        prior = winner_by_frame.get(frame_index)
        if prior is None:
            winner_by_frame[frame_index] = item
            continue
        prior_target = request.lattice_timestamps_sec[prior[1]]
        prior_error = abs(float(prior[2].actual_pts_sec) - prior_target)
        if (error, target) < (prior_error, prior_target):
            winner_by_frame[frame_index] = item

    for primary_position, canonical_index, match, pass_index in provisional:
        winner = winner_by_frame[int(match.decoded_frame_index)]
        is_winner = winner[0] == primary_position
        winner_target = request.lattice_timestamps_sec[winner[1]]
        attempt = _attempt(
            state,
            canonical_index=canonical_index,
            target_source=request.primary_sources[primary_position],
            candidate_rank=request.primary_ranks[primary_position],
            stage="primary",
            repair_round=0,
            distance_relaxed=False,
            nearest_distance=None,
            match=match,
            decode_pass_index=pass_index,
            status="selected" if is_winner else "rejected_duplicate",
            duplicate_winner_target_sec=None if is_winner else winner_target,
            duplicate_resolution_stage=None if is_winner else "initial",
        )
        state.attempts.append(attempt)
        if is_winner:
            state.accepted_by_frame[attempt.decoded_frame_index] = attempt
    return state


def _apply_repair_match(
    state: _ArmState,
    *,
    canonical_index: int,
    distance_relaxed: bool,
    nearest_distance: float,
    match: DecodedFrameMatch,
    decode_pass_index: int,
) -> None:
    """Resolve one repair target against the accepted decoded-frame set.

    The frozen collision rule applies at every stage: retain the target with
    the lexicographically smaller ``(absolute decode error, target time)``.
    When a repair target wins, the former accepted attempt is explicitly
    marked as superseded and the accepted mapping is replaced.  The unique
    frame count therefore stays unchanged and repair continues toward K.
    """

    request = state.request
    target = float(request.lattice_timestamps_sec[canonical_index])
    frame_index = int(match.decoded_frame_index)
    existing = state.accepted_by_frame.get(frame_index)
    new_error = abs(float(match.actual_pts_sec) - target)
    new_wins = existing is not None and (new_error, target) < (
        existing.abs_error_sec,
        existing.target_sec,
    )

    if existing is None or new_wins:
        status = "selected"
        duplicate_winner_target_sec = None
        duplicate_resolution_stage = None
    else:
        status = "rejected_duplicate"
        duplicate_winner_target_sec = existing.target_sec
        duplicate_resolution_stage = "repair"

    attempt = _attempt(
        state,
        canonical_index=canonical_index,
        target_source=request.candidate_sources[canonical_index],
        candidate_rank=request.candidate_ranks[canonical_index],
        stage="repair",
        repair_round=state.repair_round,
        distance_relaxed=distance_relaxed,
        nearest_distance=nearest_distance,
        match=match,
        decode_pass_index=decode_pass_index,
        status=status,
        duplicate_winner_target_sec=duplicate_winner_target_sec,
        duplicate_resolution_stage=duplicate_resolution_stage,
    )

    if new_wins:
        for index, prior in enumerate(state.attempts):
            if prior is existing:
                state.attempts[index] = replace(
                    existing,
                    status="replaced_by_lower_error",
                    duplicate_winner_target_sec=target,
                    duplicate_resolution_stage="repair",
                )
                break
        else:  # pragma: no cover - internal accepted/attempt invariant
            raise RuntimeError("accepted attempt is missing from provenance")

    state.attempts.append(attempt)
    if existing is None or new_wins:
        state.accepted_by_frame[frame_index] = attempt


def _next_repair_candidate(state: _ArmState) -> tuple[int, bool, float] | None:
    request = state.request
    remaining = [
        index
        for index in range(len(request.lattice_timestamps_sec))
        if index not in state.attempted_canonical_indices
    ]
    if not remaining:
        return None
    accepted_targets = tuple(
        attempt.target_sec for attempt in state.accepted_by_frame.values()
    )
    distances = {
        index: min(
            abs(request.lattice_timestamps_sec[index] - selected)
            for selected in accepted_targets
        )
        for index in remaining
    }
    eligible = [
        index
        for index in remaining
        if distances[index] + _TIME_TOLERANCE_SEC >= REPAIR_MIN_DISTANCE_SEC
    ]
    distance_relaxed = not eligible
    pool = eligible if eligible else remaining
    best = max(
        pool,
        key=lambda index: (
            request.candidate_scores[index],
            min(distances[index], REPAIR_COVERAGE_CAP_SEC),
            -request.lattice_timestamps_sec[index],
        ),
    )
    return best, distance_relaxed, float(distances[best])


def _failure_provenance(
    state: _ArmState,
    *,
    video_path: str | Path,
    decoder_backend: str,
    decode_passes: Sequence[DecodePass],
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_RESAMPLE_SCHEMA_VERSION,
        "video_path": str(Path(video_path)),
        "method": state.request.method,
        "role": state.request.role,
        "frame_budget": FRAME_BUDGET,
        "accepted_unique_frames": len(state.accepted_by_frame),
        "lattice_size": len(state.request.lattice_timestamps_sec),
        "attempted_canonical_indices": sorted(state.attempted_canonical_indices),
        "decoder_backend": decoder_backend,
        "midpoint_tie_policy": MIDPOINT_TIE_POLICY,
        "fresh_source_decode": True,
        "prohibit_scout_frame_remap": True,
        "decode_passes": [item.to_dict() for item in decode_passes],
        "distance_relaxation_count": state.distance_relaxation_count,
        "attempts": [item.to_dict() for item in state.attempts],
    }


def _build_decode_result(
    states: Sequence[_ArmState],
    cache: Mapping[float, tuple[DecodedFrameMatch, int]],
    *,
    video_path: str | Path,
    decoder_backend: str,
    decode_passes: Sequence[DecodePass],
) -> CanonicalDecodeResult:
    arm_results: dict[str, CanonicalArmResult] = {}
    for state in states:
        selected = tuple(
            sorted(state.accepted_by_frame.values(), key=lambda item: item.target_sec)
        )
        # These historical field names count collision *losses*, including an
        # earlier accepted target later superseded by a lower-error repair.
        # Resolution stage records when the collision was decided, rather than
        # the original primary/repair stage of the losing attempt.
        initial_rejections = sum(
            attempt.duplicate_resolution_stage == "initial"
            and attempt.status in _DUPLICATE_LOSS_STATUSES
            for attempt in state.attempts
        )
        repair_rejections = sum(
            attempt.duplicate_resolution_stage == "repair"
            and attempt.status in _DUPLICATE_LOSS_STATUSES
            for attempt in state.attempts
        )
        replacements = sum(
            attempt.status == "replaced_by_lower_error" for attempt in state.attempts
        )
        repair_attempts = sum(attempt.stage == "repair" for attempt in state.attempts)
        arm_results[state.request.method] = CanonicalArmResult(
            method=state.request.method,
            role=state.request.role,
            attempts=tuple(state.attempts),
            selected=selected,
            repair_attempt_count=repair_attempts,
            duplicate_rejection_count=initial_rejections + repair_rejections,
            initial_duplicate_rejection_count=initial_rejections,
            repair_duplicate_rejection_count=repair_rejections,
            distance_relaxation_count=state.distance_relaxation_count,
            duplicate_replacement_count=replacements,
        )

    attempted_targets = sorted(
        {
            attempt.target_sec
            for result in arm_results.values()
            for attempt in result.attempts
        }
    )
    attempted_by: dict[float, set[str]] = {
        target: set() for target in attempted_targets
    }
    for result in arm_results.values():
        for attempt in result.attempts:
            attempted_by[attempt.target_sec].add(result.method)
    candidate_union = tuple(
        CandidateUnionEntry(
            target_sec=target,
            actual_pts_sec=float(cache[target][0].actual_pts_sec),
            abs_error_sec=abs(float(cache[target][0].actual_pts_sec) - target),
            decoded_frame_index=int(cache[target][0].decoded_frame_index),
            pixel_hash=str(cache[target][0].pixel_hash),
            decode_pass_index=int(cache[target][1]),
            attempted_by=tuple(sorted(attempted_by[target])),
            rgb=cache[target][0].rgb,
        )
        for target in attempted_targets
    )
    actual_pts = tuple(item.actual_pts_sec for item in candidate_union)
    frame_indices = tuple(item.decoded_frame_index for item in candidate_union)
    if any(right < left - _TIME_TOLERANCE_SEC for left, right in pairwise(actual_pts)):
        raise RuntimeError("candidate-union actual PTS are not non-decreasing")
    if any(right < left for left, right in pairwise(frame_indices)):
        raise RuntimeError(
            "candidate-union decoded frame indices are not non-decreasing"
        )

    return CanonicalDecodeResult(
        video_path=str(Path(video_path)),
        decoder_backend=decoder_backend,
        midpoint_tie_policy=MIDPOINT_TIE_POLICY,
        decode_passes=tuple(decode_passes),
        candidate_union=candidate_union,
        arms=arm_results,
    )


def decode_canonical_targets(
    video_path: str | Path,
    *,
    rc12: CanonicalArmRequest,
    canonical_uniform: CanonicalArmRequest,
    stream_index: int = 0,
    decoder: Decoder | None = None,
    decoder_backend: str = DEFAULT_DECODER_BACKEND,
) -> CanonicalDecodeResult:
    """Fresh-decode RC12 and its canonical-uniform control to exact K=16.

    The initial target union is decoded in one sequential pass.  When either
    arm has a decoded-frame collision, one dynamic backup per unfinished arm is
    selected from the shared canonical lattice.  New backup targets for that
    round are unioned into a single additional sequential pass.  A backup that
    maps to an already accepted frame is rejected; priorities are then
    recomputed against the current accepted target set.
    """

    CanonicalRequestPair("single", rc12, canonical_uniform)
    if isinstance(stream_index, bool) or not isinstance(stream_index, Integral):
        raise TypeError("stream_index must be an integer")
    if int(stream_index) < 0:
        raise ValueError("stream_index must be non-negative")
    if not isinstance(decoder_backend, str) or not decoder_backend.strip():
        raise ValueError("decoder_backend must be a non-empty string")
    initial_targets = sorted(
        set(rc12.primary_target_timestamps_sec)
        | set(canonical_uniform.primary_target_timestamps_sec)
    )
    cache, first_pass = _decode_batch(
        video_path,
        initial_targets,
        stream_index=int(stream_index),
        decoder=decoder,
        pass_index=1,
    )
    decode_passes = [first_pass]
    states = [
        _initialize_arm(rc12, cache),
        _initialize_arm(canonical_uniform, cache),
    ]

    while any(len(state.accepted_by_frame) < FRAME_BUDGET for state in states):
        pending: list[tuple[_ArmState, int, bool, float]] = []
        new_targets: list[float] = []
        for state in states:
            if len(state.accepted_by_frame) >= FRAME_BUDGET:
                continue
            choice = _next_repair_candidate(state)
            if choice is None:
                raise CanonicalDecodeExhaustedError(
                    state.request.method,
                    _failure_provenance(
                        state,
                        video_path=video_path,
                        decoder_backend=decoder_backend,
                        decode_passes=decode_passes,
                    ),
                )
            canonical_index, relaxed, nearest_distance = choice
            state.attempted_canonical_indices.add(canonical_index)
            state.repair_round += 1
            if relaxed:
                state.distance_relaxation_count += 1
            pending.append((state, canonical_index, relaxed, nearest_distance))
            target = state.request.lattice_timestamps_sec[canonical_index]
            if target not in cache:
                new_targets.append(target)

        if new_targets:
            decoded, decode_pass = _decode_batch(
                video_path,
                new_targets,
                stream_index=int(stream_index),
                decoder=decoder,
                pass_index=len(decode_passes) + 1,
            )
            cache.update(decoded)
            decode_passes.append(decode_pass)

        for state, canonical_index, relaxed, nearest_distance in pending:
            request = state.request
            target = request.lattice_timestamps_sec[canonical_index]
            match, pass_index = cache[target]
            _apply_repair_match(
                state,
                canonical_index=canonical_index,
                distance_relaxed=relaxed,
                nearest_distance=nearest_distance,
                match=match,
                decode_pass_index=pass_index,
            )

    return _build_decode_result(
        states,
        cache,
        video_path=video_path,
        decoder_backend=decoder_backend,
        decode_passes=decode_passes,
    )


def decode_canonical_request_batch(
    video_path: str | Path,
    requests: Sequence[CanonicalRequestPair],
    *,
    stream_index: int = 0,
    decoder: Decoder | None = None,
    decoder_backend: str = DEFAULT_DECODER_BACKEND,
) -> CanonicalDecodeBatchResult:
    """Serve item/origin requests with video-level unioned decode passes.

    Pass one contains only the union of all primary canonical targets.  If a
    decoded-frame collision leaves an arm below K, every unfinished arm chooses
    one dynamic backup against its current accepted targets and those backups
    are unioned into the next pass.  Thus the usual no-repair case is exactly
    one source-video pass without decoding or RGB-hashing unused lattice ticks.
    No scout PTS or scout frame index enters this function.
    """

    if isinstance(requests, (str, bytes)):
        raise TypeError("requests must be a sequence of CanonicalRequestPair values")
    try:
        pairs = tuple(requests)
    except TypeError as exc:
        raise TypeError(
            "requests must be a sequence of CanonicalRequestPair values"
        ) from exc
    if not pairs:
        raise ValueError("requests must not be empty")
    if any(not isinstance(pair, CanonicalRequestPair) for pair in pairs):
        raise TypeError("every requests entry must be a CanonicalRequestPair")
    request_ids = tuple(pair.request_id for pair in pairs)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request_id values must be unique within one video batch")
    reference_lattice = pairs[0].rc12.lattice_timestamps_sec
    if any(pair.rc12.lattice_timestamps_sec != reference_lattice for pair in pairs):
        raise ValueError(
            "every request for one video must use the exact same canonical lattice"
        )
    if isinstance(stream_index, bool) or not isinstance(stream_index, Integral):
        raise TypeError("stream_index must be an integer")
    if int(stream_index) < 0:
        raise ValueError("stream_index must be non-negative")
    if not isinstance(decoder_backend, str) or not decoder_backend.strip():
        raise ValueError("decoder_backend must be a non-empty string")

    primary_union = sorted(
        {target for pair in pairs for target in pair.rc12.primary_target_timestamps_sec}
        | {
            target
            for pair in pairs
            for target in pair.canonical_uniform.primary_target_timestamps_sec
        }
    )
    cache, first_pass = _decode_batch(
        video_path,
        primary_union,
        stream_index=int(stream_index),
        decoder=decoder,
        pass_index=1,
    )
    decode_passes = [first_pass]
    states_by_request: dict[str, list[_ArmState]] = {}
    for pair in pairs:
        states_by_request[pair.request_id] = [
            _initialize_arm(pair.rc12, cache),
            _initialize_arm(pair.canonical_uniform, cache),
        ]

    all_states = [state for states in states_by_request.values() for state in states]
    while any(len(state.accepted_by_frame) < FRAME_BUDGET for state in all_states):
        pending: list[tuple[_ArmState, int, bool, float]] = []
        new_targets: list[float] = []
        for state in all_states:
            if len(state.accepted_by_frame) >= FRAME_BUDGET:
                continue
            choice = _next_repair_candidate(state)
            if choice is None:
                raise CanonicalDecodeExhaustedError(
                    state.request.method,
                    _failure_provenance(
                        state,
                        video_path=video_path,
                        decoder_backend=decoder_backend,
                        decode_passes=decode_passes,
                    ),
                )
            canonical_index, relaxed, nearest_distance = choice
            state.attempted_canonical_indices.add(canonical_index)
            state.repair_round += 1
            if relaxed:
                state.distance_relaxation_count += 1
            pending.append((state, canonical_index, relaxed, nearest_distance))
            target = state.request.lattice_timestamps_sec[canonical_index]
            if target not in cache:
                new_targets.append(target)

        if new_targets:
            decoded, decode_pass = _decode_batch(
                video_path,
                new_targets,
                stream_index=int(stream_index),
                decoder=decoder,
                pass_index=len(decode_passes) + 1,
            )
            cache.update(decoded)
            decode_passes.append(decode_pass)

        for state, canonical_index, relaxed, nearest_distance in pending:
            request = state.request
            target = request.lattice_timestamps_sec[canonical_index]
            match, pass_index = cache[target]
            _apply_repair_match(
                state,
                canonical_index=canonical_index,
                distance_relaxed=relaxed,
                nearest_distance=nearest_distance,
                match=match,
                decode_pass_index=pass_index,
            )

    results: dict[str, CanonicalDecodeResult] = {}
    frozen_passes = tuple(decode_passes)
    for pair in pairs:
        results[pair.request_id] = _build_decode_result(
            states_by_request[pair.request_id],
            cache,
            video_path=video_path,
            decoder_backend=decoder_backend,
            decode_passes=frozen_passes,
        )

    return CanonicalDecodeBatchResult(
        video_path=str(Path(video_path)),
        decoder_backend=decoder_backend,
        midpoint_tie_policy=MIDPOINT_TIE_POLICY,
        decode_passes=frozen_passes,
        request_results=results,
    )


__all__ = [
    "CANONICAL_RESAMPLE_SCHEMA_VERSION",
    "DEFAULT_DECODER_BACKEND",
    "FRAME_BUDGET",
    "MIDPOINT_TIE_POLICY",
    "REPAIR_COVERAGE_CAP_SEC",
    "REPAIR_MIN_DISTANCE_SEC",
    "CandidateUnionEntry",
    "CanonicalArmRequest",
    "CanonicalArmResult",
    "CanonicalDecodeBatchResult",
    "CanonicalDecodeExhaustedError",
    "CanonicalDecodeResult",
    "CanonicalRequestPair",
    "CanonicalTargetAttempt",
    "DecodePass",
    "decode_canonical_request_batch",
    "decode_canonical_targets",
]
