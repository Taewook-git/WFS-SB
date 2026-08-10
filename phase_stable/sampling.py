"""Deterministic temporal-origin sampling and timestamp-based video decoding.

The real-origin protocol samples every video on several uniformly stratified
grids.  Origins are derived from stable hashes, rather than process-global
random state, so a video receives the same grids regardless of worker count or
dataset traversal order.  All origins use a common candidate count.

PyAV is intentionally an optional dependency.  It is imported only when one
of the decoding functions is called; manifest generation and timestamp
matching therefore remain usable in lightweight analysis environments.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = 1


def _as_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _as_integer(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _as_finite_real(
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


def _real_sequence(
    name: str,
    values: Sequence[Real] | Iterable[Real],
    *,
    minimum: float | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers")
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    return tuple(
        _as_finite_real(f"{name}[{index}]", value, minimum=minimum)
        for index, value in enumerate(raw_values)
    )


def deterministic_hash_uniform(master_seed: int, video_id: str, origin_id: int) -> float:
    """Map ``(master_seed, video_id, origin_id)`` deterministically to ``[0, 1)``.

    Length-prefixed components avoid ambiguous concatenations.  The upper 53
    hash bits are used so every possible result has an exact binary64
    representation and can never round up to one.
    """

    seed = _as_integer("master_seed", master_seed)
    stable_video_id = _as_nonempty_string("video_id", video_id)
    stable_origin_id = _as_integer("origin_id", origin_id, minimum=0)

    digest = hashlib.sha256()
    for component in (str(seed), stable_video_id, str(stable_origin_id)):
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    integer53 = int.from_bytes(digest.digest()[:8], byteorder="big") >> 11
    return integer53 / float(1 << 53)


def generate_stratified_origins(
    master_seed: int,
    video_id: str,
    num_origins: int,
    *,
    period_sec: float = 1.0,
) -> tuple[float, ...]:
    """Generate one deterministic random origin inside every temporal stratum.

    For origin index ``o`` and ``O=num_origins``, this implements
    ``period_sec * (o + u(seed, video_id, o)) / O``.
    """

    count = _as_integer("num_origins", num_origins, minimum=1)
    period = _as_finite_real("period_sec", period_sec, strictly_positive=True)
    # Validate these even before the first loop iteration for clearer errors.
    _as_integer("master_seed", master_seed)
    _as_nonempty_string("video_id", video_id)

    return tuple(
        period
        * (origin_id + deterministic_hash_uniform(master_seed, video_id, origin_id))
        / count
        for origin_id in range(count)
    )


# A descriptive alias used by callers that mirror the protocol wording.
stratified_random_origins = generate_stratified_origins


def common_candidate_count(
    duration_sec: float,
    origins_sec: Sequence[Real],
    *,
    sample_fps: float = 1.0,
    epsilon_sec: float = 1e-9,
) -> int:
    """Return the largest common candidate count valid for every origin.

    The implementation follows the experiment protocol exactly and clamps the
    result to zero for videos shorter than their latest origin.
    """

    duration = _as_finite_real("duration_sec", duration_sec, minimum=0.0)
    fps = _as_finite_real("sample_fps", sample_fps, strictly_positive=True)
    epsilon = _as_finite_real("epsilon_sec", epsilon_sec, minimum=0.0)
    origins = _real_sequence("origins_sec", origins_sec, minimum=0.0)
    if not origins:
        raise ValueError("origins_sec must contain at least one origin")

    raw_count = math.floor((duration - max(origins) - epsilon) * fps) + 1
    return max(0, raw_count)


def common_target_timestamps(
    duration_sec: float,
    origins_sec: Sequence[Real],
    *,
    sample_fps: float = 1.0,
    epsilon_sec: float = 1e-9,
) -> tuple[tuple[float, ...], ...]:
    """Build equal-length absolute timestamp grids for all origins."""

    origins = _real_sequence("origins_sec", origins_sec, minimum=0.0)
    count = common_candidate_count(
        duration_sec,
        origins,
        sample_fps=sample_fps,
        epsilon_sec=epsilon_sec,
    )
    fps = float(sample_fps)
    return tuple(
        tuple(origin + sample_index / fps for sample_index in range(count))
        for origin in origins
    )


@dataclass(frozen=True)
class SamplingOrigin:
    """One sampling origin and its absolute target timestamp grid."""

    origin_id: int
    origin_sec: float
    target_timestamps_sec: tuple[float, ...]

    def __post_init__(self) -> None:
        origin_id = _as_integer("origin_id", self.origin_id, minimum=0)
        origin_sec = _as_finite_real("origin_sec", self.origin_sec, minimum=0.0)
        targets = _real_sequence(
            "target_timestamps_sec", self.target_timestamps_sec, minimum=0.0
        )
        if targets and not math.isclose(
            targets[0], origin_sec, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("the first target timestamp must equal origin_sec")
        if any(current <= previous for previous, current in zip(targets, targets[1:])):
            raise ValueError("target_timestamps_sec must be strictly increasing")

        object.__setattr__(self, "origin_id", origin_id)
        object.__setattr__(self, "origin_sec", origin_sec)
        object.__setattr__(self, "target_timestamps_sec", targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "origin_sec": self.origin_sec,
            "target_timestamps_sec": list(self.target_timestamps_sec),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplingOrigin":
        if not isinstance(payload, Mapping):
            raise TypeError("origin payload must be a mapping")
        try:
            return cls(
                origin_id=payload["origin_id"],
                origin_sec=payload["origin_sec"],
                target_timestamps_sec=tuple(payload["target_timestamps_sec"]),
            )
        except KeyError as exc:
            raise ValueError(f"origin payload is missing field {exc.args[0]!r}") from exc
        except TypeError as exc:
            if "target_timestamps_sec" in payload:
                raise TypeError("target_timestamps_sec must be an iterable") from exc
            raise


@dataclass(frozen=True)
class SamplingManifest:
    """Reproducible real-origin sampling specification for one video."""

    video_id: str
    master_seed: int
    duration_sec: float
    sample_fps: float
    period_sec: float
    epsilon_sec: float
    candidate_count: int
    origins: tuple[SamplingOrigin, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        video_id = _as_nonempty_string("video_id", self.video_id)
        master_seed = _as_integer("master_seed", self.master_seed)
        duration = _as_finite_real("duration_sec", self.duration_sec, minimum=0.0)
        sample_fps = _as_finite_real(
            "sample_fps", self.sample_fps, strictly_positive=True
        )
        period = _as_finite_real("period_sec", self.period_sec, strictly_positive=True)
        epsilon = _as_finite_real("epsilon_sec", self.epsilon_sec, minimum=0.0)
        candidate_count = _as_integer(
            "candidate_count", self.candidate_count, minimum=0
        )
        schema_version = _as_integer(
            "schema_version", self.schema_version, minimum=1
        )
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported sampling manifest schema_version {schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not math.isclose(period, 1.0 / sample_fps, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("period_sec must equal 1 / sample_fps")

        try:
            origins = tuple(self.origins)
        except TypeError as exc:
            raise TypeError("origins must be an iterable of SamplingOrigin objects") from exc
        if not origins:
            raise ValueError("origins must contain at least one origin")
        if any(not isinstance(origin, SamplingOrigin) for origin in origins):
            raise TypeError("every origins entry must be a SamplingOrigin")
        expected_ids = tuple(range(len(origins)))
        actual_ids = tuple(origin.origin_id for origin in origins)
        if actual_ids != expected_ids:
            raise ValueError("origin_id values must be consecutive and ordered from zero")
        if any(origin.origin_sec >= period for origin in origins):
            raise ValueError("each origin_sec must lie in [0, period_sec)")
        if any(
            len(origin.target_timestamps_sec) != candidate_count for origin in origins
        ):
            raise ValueError("all origins must have candidate_count target timestamps")

        expected_count = common_candidate_count(
            duration,
            [origin.origin_sec for origin in origins],
            sample_fps=sample_fps,
            epsilon_sec=epsilon,
        )
        if candidate_count != expected_count:
            raise ValueError(
                "candidate_count does not match duration, origins, sample_fps, and epsilon_sec"
            )
        expected_targets = common_target_timestamps(
            duration,
            [origin.origin_sec for origin in origins],
            sample_fps=sample_fps,
            epsilon_sec=epsilon,
        )
        for origin, expected in zip(origins, expected_targets):
            if origin.target_timestamps_sec != expected:
                raise ValueError(
                    f"origin {origin.origin_id} target timestamps do not match its sampling grid"
                )

        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "master_seed", master_seed)
        object.__setattr__(self, "duration_sec", duration)
        object.__setattr__(self, "sample_fps", sample_fps)
        object.__setattr__(self, "period_sec", period)
        object.__setattr__(self, "epsilon_sec", epsilon)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "origins", origins)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def num_origins(self) -> int:
        return len(self.origins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "master_seed": self.master_seed,
            "duration_sec": self.duration_sec,
            "sample_fps": self.sample_fps,
            "period_sec": self.period_sec,
            "epsilon_sec": self.epsilon_sec,
            "num_origins": self.num_origins,
            "candidate_count": self.candidate_count,
            "origins": [origin.to_dict() for origin in self.origins],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SamplingManifest":
        if not isinstance(payload, Mapping):
            raise TypeError("manifest payload must be a mapping")
        required = (
            "schema_version",
            "video_id",
            "master_seed",
            "duration_sec",
            "sample_fps",
            "period_sec",
            "epsilon_sec",
            "candidate_count",
            "origins",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"manifest payload is missing fields: {', '.join(missing)}")
        try:
            origins = tuple(SamplingOrigin.from_dict(item) for item in payload["origins"])
        except TypeError as exc:
            raise TypeError("manifest origins must be an iterable") from exc
        manifest = cls(
            schema_version=payload["schema_version"],
            video_id=payload["video_id"],
            master_seed=payload["master_seed"],
            duration_sec=payload["duration_sec"],
            sample_fps=payload["sample_fps"],
            period_sec=payload["period_sec"],
            epsilon_sec=payload["epsilon_sec"],
            candidate_count=payload["candidate_count"],
            origins=origins,
        )
        if "num_origins" in payload:
            declared_count = _as_integer("num_origins", payload["num_origins"], minimum=1)
            if declared_count != manifest.num_origins:
                raise ValueError("num_origins does not match the origins array")
        return manifest


def build_sampling_manifest(
    video_id: str,
    duration_sec: float,
    *,
    master_seed: int = 0,
    num_origins: int = 5,
    sample_fps: float = 1.0,
    epsilon_sec: float = 1e-9,
) -> SamplingManifest:
    """Create a validated deterministic sampling manifest for one video."""

    fps = _as_finite_real("sample_fps", sample_fps, strictly_positive=True)
    period = 1.0 / fps
    origins_sec = generate_stratified_origins(
        master_seed,
        video_id,
        num_origins,
        period_sec=period,
    )
    grids = common_target_timestamps(
        duration_sec,
        origins_sec,
        sample_fps=fps,
        epsilon_sec=epsilon_sec,
    )
    origins = tuple(
        SamplingOrigin(origin_id, origin_sec, targets)
        for origin_id, (origin_sec, targets) in enumerate(zip(origins_sec, grids))
    )
    return SamplingManifest(
        video_id=video_id,
        master_seed=master_seed,
        duration_sec=duration_sec,
        sample_fps=fps,
        period_sec=period,
        epsilon_sec=epsilon_sec,
        candidate_count=len(grids[0]),
        origins=origins,
    )


# A short alias that is convenient at preprocessing call sites.
make_sampling_manifest = build_sampling_manifest


def manifest_to_json(manifest: SamplingManifest, *, indent: int | None = 2) -> str:
    if not isinstance(manifest, SamplingManifest):
        raise TypeError("manifest must be a SamplingManifest")
    return json.dumps(
        manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent
    )


def manifest_from_json(payload: str) -> SamplingManifest:
    if not isinstance(payload, str):
        raise TypeError("payload must be a JSON string")
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sampling manifest JSON: {exc.msg}") from exc
    return SamplingManifest.from_dict(decoded)


def write_manifest_json(path: str | Path, manifest: SamplingManifest) -> None:
    """Write one sampling manifest as UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(manifest_to_json(manifest) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def read_manifest_json(path: str | Path) -> SamplingManifest:
    return manifest_from_json(Path(path).read_text(encoding="utf-8"))


def write_manifests_jsonl(
    path: str | Path, manifests: Iterable[SamplingManifest]
) -> None:
    """Write one compact manifest JSON object per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for line_number, manifest in enumerate(manifests, start=1):
            if not isinstance(manifest, SamplingManifest):
                raise TypeError(
                    f"manifests item {line_number} must be a SamplingManifest"
                )
            handle.write(manifest_to_json(manifest, indent=None))
            handle.write("\n")
    temporary.replace(output_path)


def iter_manifests_jsonl(path: str | Path) -> Iterator[SamplingManifest]:
    """Yield validated manifests from a JSONL file, skipping blank lines."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield manifest_from_json(line)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid sampling manifest at {input_path}:{line_number}: {exc}"
                ) from exc


def read_manifests_jsonl(path: str | Path) -> list[SamplingManifest]:
    return list(iter_manifests_jsonl(path))


def nearest_pts_indices(
    decoded_pts_sec: Sequence[Real], target_timestamps_sec: Sequence[Real]
) -> list[int]:
    """Match each target to its nearest decoded PTS.

    Decoded PTS values must be in nondecreasing presentation order.  An exact
    midpoint is assigned to the earlier PTS, and output order matches target
    input order.
    """

    decoded = _real_sequence("decoded_pts_sec", decoded_pts_sec)
    targets = _real_sequence(
        "target_timestamps_sec", target_timestamps_sec, minimum=0.0
    )
    if not targets:
        return []
    if not decoded:
        raise ValueError("decoded_pts_sec must not be empty when targets are present")
    if any(current < previous for previous, current in zip(decoded, decoded[1:])):
        raise ValueError("decoded_pts_sec must be in nondecreasing order")

    decoded_array = np.asarray(decoded, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    right = np.searchsorted(decoded_array, target_array, side="left")
    result: list[int] = []
    for target, right_index in zip(target_array, right):
        index = int(right_index)
        if index == 0:
            result.append(0)
        elif index == len(decoded):
            result.append(len(decoded) - 1)
        else:
            earlier = index - 1
            earlier_distance = float(target - decoded_array[earlier])
            later_distance = float(decoded_array[index] - target)
            result.append(earlier if earlier_distance <= later_distance else index)
    return result


@dataclass(frozen=True)
class DecodedFrameMatch:
    """A target timestamp paired with the nearest decoded RGB frame."""

    target_timestamp_sec: float
    actual_pts_sec: float
    decode_error_ms: float
    pixel_hash: str
    rgb: np.ndarray = field(repr=False, compare=False)
    decoded_frame_index: int = -1


def _import_pyav() -> Any:
    try:
        return importlib.import_module("av")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "PyAV is required for timestamp-based video decoding. "
            "Install the optional dependency with `pip install av`."
        ) from exc


def _pixel_hash(rgb: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(rgb)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def iter_nearest_frames_pyav_indexed(
    video_path: str | Path,
    target_timestamps_sec: Sequence[Real],
    *,
    stream_index: int = 0,
) -> Iterator[tuple[int, DecodedFrameMatch]]:
    """Yield ``(input_target_index, match)`` during one sequential decode.

    Targets may be unsorted and may contain duplicates.  Internally they are
    stable-sorted. At an equal PTS distance, the earlier frame is selected.
    Yielding lets preprocessing consume bounded batches instead of retaining
    every origin's full-resolution RGB arrays in memory.
    """

    targets = _real_sequence(
        "target_timestamps_sec", target_timestamps_sec, minimum=0.0
    )
    selected_stream_index = _as_integer(
        "stream_index", stream_index, minimum=0
    )
    if not targets:
        return

    av = _import_pyav()
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")

    ordered_targets = sorted(enumerate(targets), key=lambda item: item[1])

    def candidate(frame: Any, pts_sec: float, decoded_frame_index: int) -> dict[str, Any]:
        return {
            "frame": frame,
            "pts_sec": pts_sec,
            "decoded_frame_index": decoded_frame_index,
            "rgb": None,
            "pixel_hash": None,
        }

    def materialize(target_sec: float, item: dict[str, Any]) -> DecodedFrameMatch:
        if item["rgb"] is None:
            rgb = np.ascontiguousarray(item["frame"].to_ndarray(format="rgb24"))
            item["rgb"] = rgb
            item["pixel_hash"] = _pixel_hash(rgb)
        actual_pts = float(item["pts_sec"])
        return DecodedFrameMatch(
            target_timestamp_sec=target_sec,
            actual_pts_sec=actual_pts,
            decode_error_ms=abs(actual_pts - target_sec) * 1000.0,
            pixel_hash=str(item["pixel_hash"]),
            rgb=item["rgb"],
            decoded_frame_index=int(item["decoded_frame_index"]),
        )

    container = av.open(str(path))
    previous: dict[str, Any] | None = None
    target_cursor = 0
    try:
        video_streams = container.streams.video
        if selected_stream_index >= len(video_streams):
            raise ValueError(
                f"video has {len(video_streams)} video stream(s); "
                f"stream_index={selected_stream_index} is invalid"
            )
        stream = video_streams[selected_stream_index]

        for decoded_frame_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                continue
            time_base = frame.time_base
            if time_base is None:
                time_base = stream.time_base
            if time_base is None:
                continue
            pts_sec = float(frame.pts * time_base)
            if not math.isfinite(pts_sec):
                continue
            if previous is not None and pts_sec < previous["pts_sec"]:
                raise RuntimeError(
                    "PyAV returned non-monotonic presentation timestamps; "
                    "cannot perform sequential nearest-PTS matching"
                )

            current = candidate(frame, pts_sec, decoded_frame_index)
            while (
                target_cursor < len(ordered_targets)
                and ordered_targets[target_cursor][1] <= pts_sec
            ):
                original_index, target_sec = ordered_targets[target_cursor]
                if previous is None:
                    chosen = current
                else:
                    previous_distance = abs(target_sec - previous["pts_sec"])
                    current_distance = abs(pts_sec - target_sec)
                    chosen = previous if previous_distance <= current_distance else current
                yield original_index, materialize(target_sec, chosen)
                target_cursor += 1
            previous = current

        if previous is None:
            raise RuntimeError("video contains no decoded frame with a valid PTS")
        while target_cursor < len(ordered_targets):
            original_index, target_sec = ordered_targets[target_cursor]
            yield original_index, materialize(target_sec, previous)
            target_cursor += 1
    finally:
        container.close()



def decode_nearest_frames_pyav(
    video_path: str | Path,
    target_timestamps_sec: Sequence[Real],
    *,
    stream_index: int = 0,
) -> list[DecodedFrameMatch]:
    """Decode once and return matches restored to caller target order."""

    targets = _real_sequence(
        "target_timestamps_sec", target_timestamps_sec, minimum=0.0
    )
    matches: list[DecodedFrameMatch | None] = [None] * len(targets)
    for target_index, match in iter_nearest_frames_pyav_indexed(
        video_path,
        targets,
        stream_index=stream_index,
    ):
        matches[target_index] = match
    if any(match is None for match in matches):  # pragma: no cover - invariant guard
        raise RuntimeError("internal decoder error: one or more targets were not matched")
    return [match for match in matches if match is not None]


def decode_manifest_pyav(
    video_path: str | Path,
    manifest: SamplingManifest,
    *,
    stream_index: int = 0,
) -> dict[int, tuple[DecodedFrameMatch, ...]]:
    """Decode every origin in a manifest together in one video pass."""

    if not isinstance(manifest, SamplingManifest):
        raise TypeError("manifest must be a SamplingManifest")
    flat_targets = [
        timestamp
        for origin in manifest.origins
        for timestamp in origin.target_timestamps_sec
    ]
    flat_matches = decode_nearest_frames_pyav(
        video_path, flat_targets, stream_index=stream_index
    )
    by_origin: dict[int, tuple[DecodedFrameMatch, ...]] = {}
    cursor = 0
    for origin in manifest.origins:
        next_cursor = cursor + manifest.candidate_count
        by_origin[origin.origin_id] = tuple(flat_matches[cursor:next_cursor])
        cursor = next_cursor
    return by_origin


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "DecodedFrameMatch",
    "SamplingManifest",
    "SamplingOrigin",
    "build_sampling_manifest",
    "common_candidate_count",
    "common_target_timestamps",
    "decode_manifest_pyav",
    "decode_nearest_frames_pyav",
    "deterministic_hash_uniform",
    "generate_stratified_origins",
    "iter_nearest_frames_pyav_indexed",
    "iter_manifests_jsonl",
    "make_sampling_manifest",
    "manifest_from_json",
    "manifest_to_json",
    "nearest_pts_indices",
    "read_manifest_json",
    "read_manifests_jsonl",
    "stratified_random_origins",
    "write_manifest_json",
    "write_manifests_jsonl",
]
