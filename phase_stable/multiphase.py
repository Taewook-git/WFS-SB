"""Physical timestamp contracts for multi-phase sampling.

This module is deliberately independent of the phase-fusion algorithm.  It
only defines how one *outer* sampling origin is expanded into a dense physical
timestamp lattice, how that lattice is partitioned into ``P`` interleaved
base-rate phase streams, and how decoded source frames are de-duplicated and
scattered back to dense targets.

The distinction between timestamps is important:

* ``timestamps_sec`` are requested physical target times on the dense lattice;
* ``actual_pts_sec`` identify decoded source frames and may repeat when nearby
  dense targets resolve to the same frame;
* ``phase_id`` is a label.  It may be permuted without changing the physical
  lattice or any phase slot.

Fusion code may index dense features with :attr:`PhaseSignalStream.dense_indices`,
but must not infer time from array indices.  Resampling is restricted to the
intersection returned by :func:`common_valid_support`; the interpolation helper
in this module rejects extrapolation rather than silently extending endpoints.

``OriginSignalRecord.visual_features_path`` always denotes a dense-target-
aligned array with ``manifest.dense_candidate_count`` rows.  Feature extraction
may run once per unique source PTS, but the result must be expanded with
:meth:`SourcePTSScatter.scatter_unique` before it is stored.  This keeps the
artifact compatible with existing record consumers and makes phase indexing
unambiguous.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import OriginSignalRecord
from .repro import sha256_file
from .sampling import generate_stratified_origins

MULTIPHASE_SCHEMA_VERSION = 1


def _integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _real_tuple(
    name: str,
    values: Sequence[Real],
    *,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    return tuple(
        _finite_real(f"{name}[{index}]", value, minimum=minimum)
        for index, value in enumerate(raw)
    )


def _int_tuple(
    name: str,
    values: Sequence[Integral],
    *,
    minimum: int | None = None,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    return tuple(
        _integer(f"{name}[{index}]", value, minimum=minimum)
        for index, value in enumerate(raw)
    )


def _strictly_increasing(name: str, values: tuple[float, ...]) -> None:
    if any(current <= previous for previous, current in pairwise(values)):
        raise ValueError(f"{name} must be strictly increasing")


def _close(first: float, second: float, *, atol: float = 1e-12) -> bool:
    return math.isclose(first, second, rel_tol=1e-12, abs_tol=atol)


def common_valid_support(
    timestamp_streams: Sequence[Sequence[Real]],
    *,
    atol: float = 1e-12,
) -> tuple[float, float]:
    """Return the closed intersection of timestamp extents.

    The function only identifies a continuous support interval; it does not
    imply that staggered phase streams share exact sample timestamps.  Callers
    that resample a stream must stay inside this interval.
    """

    tolerance = _finite_real("atol", atol, minimum=0.0)
    if isinstance(timestamp_streams, (str, bytes)):
        raise TypeError("timestamp_streams must be a sequence of streams")
    try:
        streams = tuple(timestamp_streams)
    except TypeError as exc:
        raise TypeError("timestamp_streams must be an iterable of streams") from exc
    if not streams:
        raise ValueError("timestamp_streams must contain at least one stream")

    normalized: list[tuple[float, ...]] = []
    for index, stream in enumerate(streams):
        values = _real_tuple(f"timestamp_streams[{index}]", stream, minimum=0.0)
        if not values:
            raise ValueError("every timestamp stream must be non-empty")
        _strictly_increasing(f"timestamp_streams[{index}]", values)
        normalized.append(values)

    start = max(stream[0] for stream in normalized)
    stop = min(stream[-1] for stream in normalized)
    if stop < start - tolerance:
        raise ValueError("timestamp streams have no common valid support")
    stop = max(stop, start)
    return float(start), float(stop)


def interpolate_no_extrapolation(
    source_timestamps_sec: Sequence[Real],
    source_values: Sequence[Any] | np.ndarray,
    target_timestamps_sec: Sequence[Real],
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Linearly interpolate along axis zero and reject extrapolation.

    This small reference helper makes the data contract executable.  Fusion
    implementations may use a different interpolator, provided they enforce
    the same closed source support and operate in physical seconds.
    """

    source_times = _real_tuple(
        "source_timestamps_sec", source_timestamps_sec, minimum=0.0
    )
    targets = _real_tuple("target_timestamps_sec", target_timestamps_sec, minimum=0.0)
    tolerance = _finite_real("atol", atol, minimum=0.0)
    if not source_times:
        raise ValueError("source_timestamps_sec must not be empty")
    _strictly_increasing("source_timestamps_sec", source_times)
    if targets:
        _strictly_increasing("target_timestamps_sec", targets)
        if targets[0] < source_times[0] - tolerance or targets[-1] > source_times[-1] + tolerance:
            raise ValueError("target timestamps require extrapolation")

    try:
        values = np.asarray(source_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("source_values must be a rectangular numeric array") from exc
    if values.ndim == 0 or values.shape[0] != len(source_times):
        raise ValueError("source_values must be aligned on axis zero with source timestamps")
    if not np.all(np.isfinite(values)):
        raise ValueError("source_values must contain only finite values")
    if not targets:
        return np.empty((0, *values.shape[1:]), dtype=float)
    clipped_targets = np.clip(np.asarray(targets), source_times[0], source_times[-1])
    flattened = values.reshape(values.shape[0], -1)
    interpolated = np.column_stack(
        [
            np.interp(clipped_targets, np.asarray(source_times), flattened[:, column])
            for column in range(flattened.shape[1])
        ]
    )
    return interpolated.reshape((len(targets), *values.shape[1:]))


@dataclass(frozen=True)
class MultiPhaseOuterOrigin:
    """One outer origin and its interleaved physical dense lattice."""

    outer_origin_id: int
    outer_origin_sec: float
    dense_target_timestamps_sec: tuple[float, ...]
    phase_ids: tuple[int, ...]
    phase_order: tuple[int, ...]
    candidate_count_per_phase: int
    common_valid_support_sec: tuple[float, float]

    def __post_init__(self) -> None:
        origin_id = _integer("outer_origin_id", self.outer_origin_id, minimum=0)
        origin_sec = _finite_real("outer_origin_sec", self.outer_origin_sec, minimum=0.0)
        targets = _real_tuple(
            "dense_target_timestamps_sec",
            self.dense_target_timestamps_sec,
            minimum=0.0,
        )
        if not targets:
            raise ValueError("dense_target_timestamps_sec must not be empty")
        _strictly_increasing("dense_target_timestamps_sec", targets)
        order = _int_tuple("phase_order", self.phase_order, minimum=0)
        if not order or set(order) != set(range(len(order))):
            raise ValueError("phase_order must be a permutation of range(num_phases)")
        phase_ids = _int_tuple("phase_ids", self.phase_ids, minimum=0)
        count = _integer(
            "candidate_count_per_phase", self.candidate_count_per_phase, minimum=2
        )
        if len(targets) != len(order) * count:
            raise ValueError("dense target count must equal num_phases * per-phase count")
        expected_ids = tuple(order[index % len(order)] for index in range(len(targets)))
        if phase_ids != expected_ids:
            raise ValueError("phase_ids do not match the interleaved phase_order")
        support = _real_tuple(
            "common_valid_support_sec", self.common_valid_support_sec, minimum=0.0
        )
        if len(support) != 2 or support[1] < support[0]:
            raise ValueError("common_valid_support_sec must be a non-empty [start, stop]")
        streams = [self.phase_timestamps(phase_id) for phase_id in range(len(order))]
        expected_support = common_valid_support(streams)
        if not all(_close(actual, expected) for actual, expected in zip(support, expected_support)):
            raise ValueError("common_valid_support_sec does not match phase extents")

        object.__setattr__(self, "outer_origin_id", origin_id)
        object.__setattr__(self, "outer_origin_sec", origin_sec)
        object.__setattr__(self, "dense_target_timestamps_sec", targets)
        object.__setattr__(self, "phase_ids", phase_ids)
        object.__setattr__(self, "phase_order", order)
        object.__setattr__(self, "candidate_count_per_phase", count)
        object.__setattr__(self, "common_valid_support_sec", support)

    @property
    def num_phases(self) -> int:
        return len(self.phase_order)

    @property
    def dense_candidate_count(self) -> int:
        return len(self.dense_target_timestamps_sec)

    def phase_slot(self, phase_id: int) -> int:
        stable_id = _integer("phase_id", phase_id, minimum=0)
        try:
            return self.phase_order.index(stable_id)
        except ValueError as exc:
            raise ValueError(f"unknown phase_id {stable_id}") from exc

    def phase_dense_indices(self, phase_id: int) -> tuple[int, ...]:
        slot = self.phase_slot(phase_id)
        return tuple(range(slot, self.dense_candidate_count, self.num_phases))

    def phase_timestamps(self, phase_id: int) -> tuple[float, ...]:
        return tuple(
            self.dense_target_timestamps_sec[index]
            for index in self.phase_dense_indices(phase_id)
        )


@dataclass(frozen=True)
class MultiPhaseManifest:
    """Validated multi-phase grids for all evaluation outer origins.

    Unlike :class:`phase_stable.sampling.SamplingManifest`, the outer-origin
    period is ``1 / base_sample_fps`` while dense targets are spaced at
    ``1 / (P * base_sample_fps)``.  Keeping this as a separate type prevents a
    dense grid from being misrepresented as an ordinary single-rate manifest.
    """

    video_id: str
    master_seed: int
    duration_sec: float
    base_sample_fps: float
    num_phases: int
    dense_sample_fps: float
    phase_order: tuple[int, ...]
    candidate_count_per_phase: int
    outer_origins: tuple[MultiPhaseOuterOrigin, ...]
    common_valid_support_sec: tuple[float, float]
    epsilon_sec: float = 1e-9
    schema_version: int = MULTIPHASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        video_id = _nonempty_text("video_id", self.video_id)
        seed = _integer("master_seed", self.master_seed)
        duration = _finite_real("duration_sec", self.duration_sec, strictly_positive=True)
        base_fps = _finite_real(
            "base_sample_fps", self.base_sample_fps, strictly_positive=True
        )
        phases = _integer("num_phases", self.num_phases, minimum=2)
        dense_fps = _finite_real(
            "dense_sample_fps", self.dense_sample_fps, strictly_positive=True
        )
        if not _close(dense_fps, phases * base_fps):
            raise ValueError("dense_sample_fps must equal num_phases * base_sample_fps")
        order = _int_tuple("phase_order", self.phase_order, minimum=0)
        if len(order) != phases or set(order) != set(range(phases)):
            raise ValueError("phase_order must be a permutation of range(num_phases)")
        count = _integer(
            "candidate_count_per_phase", self.candidate_count_per_phase, minimum=2
        )
        epsilon = _finite_real("epsilon_sec", self.epsilon_sec, minimum=0.0)
        schema = _integer("schema_version", self.schema_version, minimum=1)
        if schema != MULTIPHASE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported multiphase schema_version {schema}; "
                f"expected {MULTIPHASE_SCHEMA_VERSION}"
            )
        try:
            origins = tuple(self.outer_origins)
        except TypeError as exc:
            raise TypeError("outer_origins must be an iterable") from exc
        if not origins or any(not isinstance(item, MultiPhaseOuterOrigin) for item in origins):
            raise TypeError("outer_origins must contain MultiPhaseOuterOrigin objects")
        if tuple(item.outer_origin_id for item in origins) != tuple(range(len(origins))):
            raise ValueError("outer_origin_id values must be consecutive and ordered")
        if any(item.phase_order != order for item in origins):
            raise ValueError("all outer origins must use the manifest phase_order")
        if any(item.candidate_count_per_phase != count for item in origins):
            raise ValueError("all outer origins must use the common per-phase count")
        outer_period = 1.0 / base_fps
        outer_offsets = tuple(item.outer_origin_sec for item in origins)
        if any(value >= outer_period for value in outer_offsets):
            raise ValueError("outer origins must lie in [0, 1/base_sample_fps)")
        _strictly_increasing("outer origin offsets", outer_offsets)
        dense_period = 1.0 / dense_fps
        for origin in origins:
            expected = tuple(
                origin.outer_origin_sec + index * dense_period
                for index in range(phases * count)
            )
            if not np.allclose(
                origin.dense_target_timestamps_sec,
                expected,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError("outer origin does not follow the physical dense lattice")
            if origin.dense_target_timestamps_sec[-1] > duration - epsilon + 1e-12:
                raise ValueError("dense target timestamp exceeds the no-extrapolation duration")
        support = _real_tuple(
            "common_valid_support_sec", self.common_valid_support_sec, minimum=0.0
        )
        if len(support) != 2:
            raise ValueError("common_valid_support_sec must contain start and stop")
        all_streams = [
            origin.phase_timestamps(phase_id)
            for origin in origins
            for phase_id in range(phases)
        ]
        expected_support = common_valid_support(all_streams)
        if not all(_close(actual, expected) for actual, expected in zip(support, expected_support)):
            raise ValueError("manifest common support does not match all phase streams")

        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "master_seed", seed)
        object.__setattr__(self, "duration_sec", duration)
        object.__setattr__(self, "base_sample_fps", base_fps)
        object.__setattr__(self, "num_phases", phases)
        object.__setattr__(self, "dense_sample_fps", dense_fps)
        object.__setattr__(self, "phase_order", order)
        object.__setattr__(self, "candidate_count_per_phase", count)
        object.__setattr__(self, "outer_origins", origins)
        object.__setattr__(self, "common_valid_support_sec", support)
        object.__setattr__(self, "epsilon_sec", epsilon)
        object.__setattr__(self, "schema_version", schema)

    @property
    def num_outer_origins(self) -> int:
        return len(self.outer_origins)

    @property
    def dense_candidate_count(self) -> int:
        return self.num_phases * self.candidate_count_per_phase

    @property
    def outer_period_sec(self) -> float:
        return 1.0 / self.base_sample_fps

    @property
    def dense_period_sec(self) -> float:
        return 1.0 / self.dense_sample_fps

    def origin(self, outer_origin_id: int) -> MultiPhaseOuterOrigin:
        stable_id = _integer("outer_origin_id", outer_origin_id, minimum=0)
        if stable_id >= len(self.outer_origins):
            raise ValueError(f"unknown outer_origin_id {stable_id}")
        return self.outer_origins[stable_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "master_seed": self.master_seed,
            "duration_sec": self.duration_sec,
            "base_sample_fps": self.base_sample_fps,
            "dense_sample_fps": self.dense_sample_fps,
            "num_phases": self.num_phases,
            "phase_order": list(self.phase_order),
            "candidate_count_per_phase": self.candidate_count_per_phase,
            "dense_candidate_count": self.dense_candidate_count,
            "epsilon_sec": self.epsilon_sec,
            "common_valid_support_sec": list(self.common_valid_support_sec),
            "outer_origins": [
                {
                    "outer_origin_id": origin.outer_origin_id,
                    "outer_origin_sec": origin.outer_origin_sec,
                    "dense_target_timestamps_sec": list(
                        origin.dense_target_timestamps_sec
                    ),
                    "phase_ids": list(origin.phase_ids),
                    "common_valid_support_sec": list(
                        origin.common_valid_support_sec
                    ),
                }
                for origin in self.outer_origins
            ],
        }


@dataclass(frozen=True)
class DenseDecodePlan:
    """One-call decode plan for every outer-origin dense target in a video.

    Targets are concatenated in outer-origin order.  The existing PyAV decoder
    accepts unsorted targets and stable-sorts them internally, so callers can
    pass :attr:`target_timestamps_sec` in one call and then use
    :meth:`split_aligned` on the returned target-aligned matches.
    """

    outer_origin_ids: tuple[int, ...]
    outer_slices: tuple[tuple[int, int], ...]
    target_timestamps_sec: tuple[float, ...]

    def __post_init__(self) -> None:
        origin_ids = _int_tuple(
            "outer_origin_ids", self.outer_origin_ids, minimum=0
        )
        if origin_ids != tuple(range(len(origin_ids))):
            raise ValueError("outer_origin_ids must be consecutive and ordered")
        try:
            raw_slices = tuple(self.outer_slices)
        except TypeError as exc:
            raise TypeError("outer_slices must be an iterable") from exc
        slices: list[tuple[int, int]] = []
        cursor = 0
        for index, raw_slice in enumerate(raw_slices):
            if len(raw_slice) != 2:
                raise ValueError("each outer slice must contain start and stop")
            start = _integer(f"outer_slices[{index}][0]", raw_slice[0], minimum=0)
            stop = _integer(f"outer_slices[{index}][1]", raw_slice[1], minimum=0)
            if start != cursor or stop <= start:
                raise ValueError("outer_slices must be non-empty and contiguous")
            slices.append((start, stop))
            cursor = stop
        targets = _real_tuple(
            "target_timestamps_sec", self.target_timestamps_sec, minimum=0.0
        )
        if len(slices) != len(origin_ids) or cursor != len(targets):
            raise ValueError("decode slices must cover all targets and outer origins")
        object.__setattr__(self, "outer_origin_ids", origin_ids)
        object.__setattr__(self, "outer_slices", tuple(slices))
        object.__setattr__(self, "target_timestamps_sec", targets)

    def split_aligned(self, values: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
        """Split target-aligned decoder output back into outer-origin tuples."""

        if isinstance(values, (str, bytes)):
            raise TypeError("values must be a target-aligned sequence")
        try:
            stable_values = tuple(values)
        except TypeError as exc:
            raise TypeError("values must be an iterable") from exc
        if len(stable_values) != len(self.target_timestamps_sec):
            raise ValueError("values must align with every decode-plan target")
        return tuple(
            stable_values[start:stop] for start, stop in self.outer_slices
        )


def build_dense_decode_plan(manifest: MultiPhaseManifest) -> DenseDecodePlan:
    """Flatten all outer-origin targets for one sequential PyAV decode call."""

    if not isinstance(manifest, MultiPhaseManifest):
        raise TypeError("manifest must be a MultiPhaseManifest")
    targets: list[float] = []
    slices: list[tuple[int, int]] = []
    for origin in manifest.outer_origins:
        start = len(targets)
        targets.extend(origin.dense_target_timestamps_sec)
        slices.append((start, len(targets)))
    return DenseDecodePlan(
        outer_origin_ids=tuple(origin.outer_origin_id for origin in manifest.outer_origins),
        outer_slices=tuple(slices),
        target_timestamps_sec=tuple(targets),
    )


def build_multiphase_manifest(
    video_id: str,
    duration_sec: float,
    *,
    base_sample_fps: float = 1.0,
    num_phases: int = 4,
    master_seed: int = 0,
    num_outer_origins: int = 5,
    outer_origins_sec: Sequence[Real] | None = None,
    phase_order: Sequence[Integral] | None = None,
    epsilon_sec: float = 1e-9,
) -> MultiPhaseManifest:
    """Build equal-length phase streams on one physical dense lattice.

    The common per-phase count is computed using the latest combination of an
    outer origin and an internal phase slot.  Thus every target is inside the
    annotated duration and all outer-origin/phase cells have equal length.
    """

    stable_video_id = _nonempty_text("video_id", video_id)
    duration = _finite_real("duration_sec", duration_sec, strictly_positive=True)
    base_fps = _finite_real(
        "base_sample_fps", base_sample_fps, strictly_positive=True
    )
    phases = _integer("num_phases", num_phases, minimum=2)
    seed = _integer("master_seed", master_seed)
    outer_count = _integer("num_outer_origins", num_outer_origins, minimum=1)
    epsilon = _finite_real("epsilon_sec", epsilon_sec, minimum=0.0)
    outer_period = 1.0 / base_fps
    if outer_origins_sec is None:
        outer_offsets = generate_stratified_origins(
            seed,
            stable_video_id,
            outer_count,
            period_sec=outer_period,
        )
    else:
        outer_offsets = _real_tuple(
            "outer_origins_sec", outer_origins_sec, minimum=0.0
        )
        if len(outer_offsets) != outer_count:
            raise ValueError("outer_origins_sec length must equal num_outer_origins")
    if any(value >= outer_period for value in outer_offsets):
        raise ValueError("outer origins must lie in [0, 1/base_sample_fps)")
    _strictly_increasing("outer_origins_sec", outer_offsets)
    if phase_order is None:
        order = tuple(range(phases))
    else:
        order = _int_tuple("phase_order", phase_order, minimum=0)
    if len(order) != phases or set(order) != set(range(phases)):
        raise ValueError("phase_order must be a permutation of range(num_phases)")

    dense_fps = phases * base_fps
    dense_period = 1.0 / dense_fps
    latest_phase_start = max(outer_offsets) + (phases - 1) * dense_period
    raw_count = math.floor((duration - latest_phase_start - epsilon) * base_fps) + 1
    per_phase_count = max(0, raw_count)
    if per_phase_count < 2:
        raise ValueError("duration leaves fewer than two samples per phase")

    origins: list[MultiPhaseOuterOrigin] = []
    for origin_id, origin_sec in enumerate(outer_offsets):
        dense_targets = tuple(
            origin_sec + index * dense_period
            for index in range(phases * per_phase_count)
        )
        phase_ids = tuple(order[index % phases] for index in range(len(dense_targets)))
        phase_streams = [
            dense_targets[slot::phases]
            for slot in range(phases)
        ]
        origins.append(
            MultiPhaseOuterOrigin(
                outer_origin_id=origin_id,
                outer_origin_sec=origin_sec,
                dense_target_timestamps_sec=dense_targets,
                phase_ids=phase_ids,
                phase_order=order,
                candidate_count_per_phase=per_phase_count,
                common_valid_support_sec=common_valid_support(phase_streams),
            )
        )
    global_support = common_valid_support(
        [
            origin.phase_timestamps(phase_id)
            for origin in origins
            for phase_id in range(phases)
        ]
    )
    return MultiPhaseManifest(
        video_id=stable_video_id,
        master_seed=seed,
        duration_sec=duration,
        base_sample_fps=base_fps,
        num_phases=phases,
        dense_sample_fps=dense_fps,
        phase_order=order,
        candidate_count_per_phase=per_phase_count,
        outer_origins=tuple(origins),
        common_valid_support_sec=global_support,
        epsilon_sec=epsilon,
    )


@dataclass(frozen=True)
class SourcePTSScatter:
    """Lossless first-axis map between unique decoded frames and dense targets."""

    unique_dense_indices: tuple[int, ...]
    unique_actual_pts_sec: tuple[float, ...]
    unique_source_frame_indices: tuple[int, ...]
    dense_to_unique: tuple[int, ...]

    def __post_init__(self) -> None:
        unique_indices = _int_tuple(
            "unique_dense_indices", self.unique_dense_indices, minimum=0
        )
        unique_pts = _real_tuple(
            "unique_actual_pts_sec", self.unique_actual_pts_sec, minimum=0.0
        )
        source_indices = _int_tuple(
            "unique_source_frame_indices",
            self.unique_source_frame_indices,
            minimum=0,
        )
        inverse = _int_tuple("dense_to_unique", self.dense_to_unique, minimum=0)
        if not unique_indices or not inverse:
            raise ValueError("source PTS scatter must not be empty")
        if len({len(unique_indices), len(unique_pts), len(source_indices)}) != 1:
            raise ValueError("unique source arrays must have equal lengths")
        if tuple(sorted(unique_indices)) != unique_indices or len(set(unique_indices)) != len(unique_indices):
            raise ValueError("unique_dense_indices must be strictly increasing")
        _strictly_increasing("unique_actual_pts_sec", unique_pts)
        if any(value >= len(unique_indices) for value in inverse):
            raise ValueError("dense_to_unique contains an out-of-range index")
        expected_first = tuple(inverse.index(index) for index in range(len(unique_indices)))
        if unique_indices != expected_first:
            raise ValueError("unique_dense_indices must identify first dense occurrences")

        object.__setattr__(self, "unique_dense_indices", unique_indices)
        object.__setattr__(self, "unique_actual_pts_sec", unique_pts)
        object.__setattr__(self, "unique_source_frame_indices", source_indices)
        object.__setattr__(self, "dense_to_unique", inverse)

    @property
    def num_unique(self) -> int:
        return len(self.unique_dense_indices)

    @property
    def dense_count(self) -> int:
        return len(self.dense_to_unique)

    def scatter_unique(self, unique_values: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Repeat unique-frame values back onto the physical dense lattice."""

        values = np.asarray(unique_values)
        if values.ndim == 0 or values.shape[0] != self.num_unique:
            raise ValueError("unique_values must have num_unique entries on axis zero")
        return values[np.asarray(self.dense_to_unique, dtype=int)]

    def gather_unique(
        self,
        dense_values: Sequence[Any] | np.ndarray,
        *,
        require_duplicate_equality: bool = True,
        rtol: float = 1e-7,
        atol: float = 1e-12,
    ) -> np.ndarray:
        """Take first occurrences, optionally checking duplicate consistency."""

        values = np.asarray(dense_values)
        if values.ndim == 0 or values.shape[0] != self.dense_count:
            raise ValueError("dense_values must have dense_count entries on axis zero")
        gathered = values[np.asarray(self.unique_dense_indices, dtype=int)]
        if require_duplicate_equality:
            reconstructed = self.scatter_unique(gathered)
            if np.issubdtype(values.dtype, np.number):
                equal = np.allclose(values, reconstructed, rtol=rtol, atol=atol)
            else:
                equal = np.array_equal(values, reconstructed)
            if not equal:
                raise ValueError("dense duplicate-frame values are inconsistent")
        return gathered


def deduplicate_source_pts(
    actual_pts_sec: Sequence[Real],
    source_frame_indices: Sequence[Integral],
) -> SourcePTSScatter:
    """Deduplicate exact source PTS and return a dense inverse/scatter map.

    PTS equality is exact by design: timestamps originate from one decoded
    frame object, so approximate merging could collapse distinct source frames.
    Repeated PTS must carry the same source-frame index.
    """

    pts = _real_tuple("actual_pts_sec", actual_pts_sec, minimum=0.0)
    frames = _int_tuple(
        "source_frame_indices", source_frame_indices, minimum=0
    )
    if not pts or len(pts) != len(frames):
        raise ValueError("actual_pts_sec and source_frame_indices must be non-empty and aligned")
    if any(current < previous for previous, current in pairwise(pts)):
        raise ValueError("actual_pts_sec must be non-decreasing")
    if any(current < previous for previous, current in pairwise(frames)):
        raise ValueError("source_frame_indices must be non-decreasing")

    unique_dense_indices: list[int] = []
    unique_pts: list[float] = []
    unique_frames: list[int] = []
    inverse: list[int] = []
    frame_to_pts: dict[int, float] = {}
    for dense_index, (pts_value, frame_index) in enumerate(zip(pts, frames)):
        prior_pts = frame_to_pts.setdefault(frame_index, pts_value)
        if prior_pts != pts_value:
            raise ValueError("one source frame index maps to multiple PTS values")
        if not unique_pts or pts_value != unique_pts[-1]:
            unique_dense_indices.append(dense_index)
            unique_pts.append(pts_value)
            unique_frames.append(frame_index)
        elif frame_index != unique_frames[-1]:
            raise ValueError("one source PTS maps to multiple source frame indices")
        inverse.append(len(unique_pts) - 1)
    return SourcePTSScatter(
        unique_dense_indices=tuple(unique_dense_indices),
        unique_actual_pts_sec=tuple(unique_pts),
        unique_source_frame_indices=tuple(unique_frames),
        dense_to_unique=tuple(inverse),
    )


@dataclass(frozen=True)
class DenseOuterRecordPayload:
    """Decoded/query-scored arrays used to build one dense outer record."""

    outer_origin_id: int
    actual_pts_sec: tuple[float, ...]
    source_frame_indices: tuple[int, ...]
    relevance_scores: tuple[float, ...]
    pixel_hashes: tuple[str, ...] = ()
    visual_features_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _multiphase_metadata(
    manifest: MultiPhaseManifest,
    origin: MultiPhaseOuterOrigin,
    scatter: SourcePTSScatter,
    visual_features_path: str | None,
) -> dict[str, Any]:
    feature_sha256: str | None = None
    feature_size_bytes: int | None = None
    if visual_features_path is not None:
        feature_path = Path(visual_features_path)
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"dense visual feature artifact does not exist: {feature_path}"
            )
        feature_sha256 = sha256_file(feature_path)
        feature_size_bytes = feature_path.stat().st_size
    return {
        "schema_version": MULTIPHASE_SCHEMA_VERSION,
        "outer_origin_id": origin.outer_origin_id,
        "outer_origin_sec": origin.outer_origin_sec,
        "base_sample_fps": manifest.base_sample_fps,
        "dense_sample_fps": manifest.dense_sample_fps,
        "num_phases": manifest.num_phases,
        "phase_order": list(manifest.phase_order),
        "phase_ids": list(origin.phase_ids),
        "candidate_count_per_phase": manifest.candidate_count_per_phase,
        "dense_candidate_count": manifest.dense_candidate_count,
        "common_valid_support_sec": list(origin.common_valid_support_sec),
        "manifest_common_valid_support_sec": list(
            manifest.common_valid_support_sec
        ),
        "source_pts_unique_dense_indices": list(scatter.unique_dense_indices),
        "source_pts_dense_to_unique": list(scatter.dense_to_unique),
        "unique_actual_pts_sec": list(scatter.unique_actual_pts_sec),
        "unique_source_frame_indices": list(scatter.unique_source_frame_indices),
        "visual_features_layout": "dense_target_aligned",
        "visual_features_rows": manifest.dense_candidate_count,
        "visual_features_sha256": feature_sha256,
        "visual_features_size_bytes": feature_size_bytes,
        "no_extrapolation": True,
    }


def build_dense_outer_record(
    manifest: MultiPhaseManifest,
    *,
    dataset: str,
    question_id: str,
    payload: DenseOuterRecordPayload,
    metadata: Mapping[str, Any] | None = None,
) -> OriginSignalRecord:
    """Build one dense record and attach the executable multi-phase contract."""

    if not isinstance(manifest, MultiPhaseManifest):
        raise TypeError("manifest must be a MultiPhaseManifest")
    if not isinstance(payload, DenseOuterRecordPayload):
        raise TypeError("payload must be a DenseOuterRecordPayload")
    origin = manifest.origin(payload.outer_origin_id)
    expected = manifest.dense_candidate_count
    lengths = {
        len(payload.actual_pts_sec),
        len(payload.source_frame_indices),
        len(payload.relevance_scores),
    }
    if lengths != {expected}:
        raise ValueError("dense payload arrays must match manifest dense_candidate_count")
    if payload.pixel_hashes and len(payload.pixel_hashes) != expected:
        raise ValueError("pixel_hashes must be empty or dense-target aligned")
    scatter = deduplicate_source_pts(
        payload.actual_pts_sec, payload.source_frame_indices
    )
    if scatter.unique_actual_pts_sec[-1] > manifest.duration_sec + manifest.epsilon_sec:
        raise ValueError("actual source PTS exceeds the manifest duration")
    common_metadata = {} if metadata is None else dict(metadata)
    payload_metadata = dict(payload.metadata)
    for reserved in ("sample_fps", "multiphase"):
        if reserved in common_metadata or reserved in payload_metadata:
            raise ValueError(f"metadata field {reserved!r} is reserved by the dense contract")
    record_metadata = {
        **common_metadata,
        **payload_metadata,
        "sample_fps": manifest.dense_sample_fps,
        "multiphase": _multiphase_metadata(
            manifest, origin, scatter, payload.visual_features_path
        ),
    }
    record = OriginSignalRecord(
        dataset=dataset,
        video_id=manifest.video_id,
        question_id=question_id,
        origin_id=origin.outer_origin_id,
        origin_sec=origin.outer_origin_sec,
        timestamps_sec=origin.dense_target_timestamps_sec,
        actual_pts_sec=payload.actual_pts_sec,
        source_frame_indices=payload.source_frame_indices,
        relevance_scores=payload.relevance_scores,
        pixel_hashes=payload.pixel_hashes,
        visual_features_path=payload.visual_features_path,
        metadata=record_metadata,
    )
    validate_dense_outer_record(record, manifest)
    return record


def build_dense_outer_records(
    manifest: MultiPhaseManifest,
    *,
    dataset: str,
    question_id: str,
    payloads: Sequence[DenseOuterRecordPayload],
    metadata: Mapping[str, Any] | None = None,
) -> tuple[OriginSignalRecord, ...]:
    """Build exactly one dense :class:`OriginSignalRecord` per outer origin."""

    if isinstance(payloads, (str, bytes)):
        raise TypeError("payloads must be a sequence")
    try:
        stable_payloads = tuple(payloads)
    except TypeError as exc:
        raise TypeError("payloads must be an iterable") from exc
    if any(not isinstance(payload, DenseOuterRecordPayload) for payload in stable_payloads):
        raise TypeError("payloads must contain DenseOuterRecordPayload objects")
    by_id: dict[int, DenseOuterRecordPayload] = {}
    for payload in stable_payloads:
        origin_id = _integer("outer_origin_id", payload.outer_origin_id, minimum=0)
        if origin_id in by_id:
            raise ValueError(f"duplicate dense payload for outer origin {origin_id}")
        by_id[origin_id] = payload
    expected_ids = set(range(manifest.num_outer_origins))
    if set(by_id) != expected_ids:
        raise ValueError("payloads must cover every manifest outer origin exactly once")
    return tuple(
        build_dense_outer_record(
            manifest,
            dataset=dataset,
            question_id=question_id,
            payload=by_id[origin_id],
            metadata=metadata,
        )
        for origin_id in range(manifest.num_outer_origins)
    )


def _require_numeric_metadata(
    payload: Mapping[str, Any], name: str, expected: float
) -> None:
    if name not in payload:
        raise ValueError(f"dense record multiphase metadata is missing {name!r}")
    actual = _finite_real(f"multiphase.{name}", payload[name])
    if not _close(actual, expected):
        raise ValueError(f"dense record multiphase metadata {name!r} mismatch")


def validate_dense_outer_record(
    record: OriginSignalRecord,
    manifest: MultiPhaseManifest,
) -> SourcePTSScatter:
    """Validate grid, source scatter, support, and no-extrapolation metadata."""

    if not isinstance(record, OriginSignalRecord):
        raise TypeError("record must be an OriginSignalRecord")
    if not isinstance(manifest, MultiPhaseManifest):
        raise TypeError("manifest must be a MultiPhaseManifest")
    if record.video_id != manifest.video_id:
        raise ValueError("dense record video_id does not match manifest")
    origin = manifest.origin(record.origin_id)
    if not _close(record.origin_sec, origin.outer_origin_sec):
        raise ValueError("dense record outer origin does not match manifest")
    if len(record.timestamps_sec) != manifest.dense_candidate_count or not np.allclose(
        record.timestamps_sec,
        origin.dense_target_timestamps_sec,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("dense record timestamps do not match the physical dense lattice")
    if record.actual_pts_sec[0] < -manifest.epsilon_sec or record.actual_pts_sec[-1] > manifest.duration_sec + manifest.epsilon_sec:
        raise ValueError("actual source PTS lies outside the manifest duration")
    scatter = deduplicate_source_pts(
        record.actual_pts_sec, record.source_frame_indices
    )
    sample_fps = record.metadata.get("sample_fps")
    if sample_fps is None or not _close(
        _finite_real("metadata.sample_fps", sample_fps), manifest.dense_sample_fps
    ):
        raise ValueError("dense record metadata sample_fps mismatch")
    contract = record.metadata.get("multiphase")
    if not isinstance(contract, Mapping):
        raise TypeError("dense record metadata has no multiphase contract mapping")
    if contract.get("schema_version") != MULTIPHASE_SCHEMA_VERSION:
        raise ValueError("dense record multiphase schema_version mismatch")
    if contract.get("outer_origin_id") != origin.outer_origin_id:
        raise ValueError("dense record multiphase outer_origin_id mismatch")
    _require_numeric_metadata(contract, "outer_origin_sec", origin.outer_origin_sec)
    _require_numeric_metadata(contract, "base_sample_fps", manifest.base_sample_fps)
    _require_numeric_metadata(contract, "dense_sample_fps", manifest.dense_sample_fps)
    exact_fields = {
        "num_phases": manifest.num_phases,
        "phase_order": list(manifest.phase_order),
        "phase_ids": list(origin.phase_ids),
        "candidate_count_per_phase": manifest.candidate_count_per_phase,
        "dense_candidate_count": manifest.dense_candidate_count,
        "common_valid_support_sec": list(origin.common_valid_support_sec),
        "manifest_common_valid_support_sec": list(
            manifest.common_valid_support_sec
        ),
        "source_pts_unique_dense_indices": list(scatter.unique_dense_indices),
        "source_pts_dense_to_unique": list(scatter.dense_to_unique),
        "unique_actual_pts_sec": list(scatter.unique_actual_pts_sec),
        "unique_source_frame_indices": list(scatter.unique_source_frame_indices),
        "visual_features_layout": "dense_target_aligned",
        "visual_features_rows": manifest.dense_candidate_count,
        "visual_features_sha256": (
            None
            if record.visual_features_path is None
            else sha256_file(record.visual_features_path)
        ),
        "visual_features_size_bytes": (
            None
            if record.visual_features_path is None
            else Path(record.visual_features_path).stat().st_size
        ),
        "no_extrapolation": True,
    }
    for name, expected in exact_fields.items():
        if contract.get(name) != expected:
            raise ValueError(f"dense record multiphase metadata {name!r} mismatch")
    return scatter


@dataclass(frozen=True)
class PhaseSignalStream:
    """One base-rate view into a dense outer-origin record.

    ``dense_indices`` are the authoritative scatter locations for any external
    frame-aligned feature array.  ``common_valid_mask`` marks samples eligible
    for cross-phase interpolation or comparison; no value outside the declared
    support may be synthesized by the fusion implementation.
    """

    dataset: str
    video_id: str
    question_id: str
    outer_origin_id: int
    outer_origin_sec: float
    phase_id: int
    phase_slot: int
    dense_indices: tuple[int, ...]
    timestamps_sec: tuple[float, ...]
    actual_pts_sec: tuple[float, ...]
    source_frame_indices: tuple[int, ...]
    relevance_scores: tuple[float, ...]
    pixel_hashes: tuple[str, ...]
    common_valid_support_sec: tuple[float, float]
    common_valid_mask: tuple[bool, ...]
    source_scatter: SourcePTSScatter
    visual_features_path: str | None = None

    @property
    def valid_dense_indices(self) -> tuple[int, ...]:
        return tuple(
            dense_index
            for dense_index, valid in zip(self.dense_indices, self.common_valid_mask)
            if valid
        )


def split_dense_record(
    record: OriginSignalRecord,
    manifest: MultiPhaseManifest,
    *,
    use_manifest_common_support: bool = False,
) -> tuple[PhaseSignalStream, ...]:
    """Validate and split a dense record into logical phase-ID order.

    By default each outer origin uses the intersection of its own P streams.
    ``use_manifest_common_support=True`` applies the stricter intersection over
    every outer origin and phase, which is useful for paired outer-origin
    comparisons.  Returned stream order is always ``phase_id=0..P-1`` and is
    therefore independent of the physical ``phase_order`` permutation.
    """

    validate_dense_outer_record(record, manifest)
    origin = manifest.origin(record.origin_id)
    support = (
        manifest.common_valid_support_sec
        if use_manifest_common_support
        else origin.common_valid_support_sec
    )
    streams: list[PhaseSignalStream] = []
    for phase_id in range(manifest.num_phases):
        slot = origin.phase_slot(phase_id)
        dense_indices = origin.phase_dense_indices(phase_id)
        timestamps = tuple(record.timestamps_sec[index] for index in dense_indices)
        actual_pts = tuple(record.actual_pts_sec[index] for index in dense_indices)
        source_indices = tuple(
            record.source_frame_indices[index] for index in dense_indices
        )
        scores = tuple(record.relevance_scores[index] for index in dense_indices)
        hashes = (
            ()
            if not record.pixel_hashes
            else tuple(record.pixel_hashes[index] for index in dense_indices)
        )
        valid_mask = tuple(
            support[0] - 1e-12 <= timestamp <= support[1] + 1e-12
            for timestamp in timestamps
        )
        if not any(valid_mask):
            raise ValueError("phase stream has no sample in common valid support")
        stream_scatter = deduplicate_source_pts(actual_pts, source_indices)
        streams.append(
            PhaseSignalStream(
                dataset=record.dataset,
                video_id=record.video_id,
                question_id=record.question_id,
                outer_origin_id=record.origin_id,
                outer_origin_sec=record.origin_sec,
                phase_id=phase_id,
                phase_slot=slot,
                dense_indices=dense_indices,
                timestamps_sec=timestamps,
                actual_pts_sec=actual_pts,
                source_frame_indices=source_indices,
                relevance_scores=scores,
                pixel_hashes=hashes,
                common_valid_support_sec=support,
                common_valid_mask=valid_mask,
                source_scatter=stream_scatter,
                visual_features_path=record.visual_features_path,
            )
        )
    return tuple(streams)


__all__ = [
    "MULTIPHASE_SCHEMA_VERSION",
    "DenseDecodePlan",
    "DenseOuterRecordPayload",
    "MultiPhaseManifest",
    "MultiPhaseOuterOrigin",
    "PhaseSignalStream",
    "SourcePTSScatter",
    "build_dense_decode_plan",
    "build_dense_outer_record",
    "build_dense_outer_records",
    "build_multiphase_manifest",
    "common_valid_support",
    "deduplicate_source_pts",
    "interpolate_no_extrapolation",
    "split_dense_record",
    "validate_dense_outer_record",
]
