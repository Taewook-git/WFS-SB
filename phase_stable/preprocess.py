"""Lightweight bridge from sampling manifests to query-conditioned signals.

This module deliberately knows nothing about a concrete vision-language model.
Callers provide an extractor with a ``compute(frames, query, batch_size)``
method.  A video's frames are decoded exactly once, then reused for every
query associated with that video.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .artifacts import OriginSignalRecord
from .sampling import (
    DecodedFrameMatch,
    SamplingManifest,
    decode_manifest_pyav,
    iter_nearest_frames_pyav_indexed,
)


@runtime_checkable
class FrameQueryExtractor(Protocol):
    """Duck-typed interface expected from a frame/query feature extractor."""

    def compute(
        self, frames: Sequence[Any], query: str, batch_size: int
    ) -> tuple[Any, Any]:
        """Return frame-aligned relevance scores and visual features."""


@dataclass(frozen=True)
class QuerySpec:
    """One query to evaluate on all origins of a decoded video."""

    question_id: str
    query: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        question_id = _nonempty_text("question_id", self.question_id)
        query = _nonempty_text("query", self.query)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        object.__setattr__(self, "question_id", question_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "QuerySpec":
        if not isinstance(payload, Mapping):
            raise TypeError("query payload must be a mapping")
        missing = [name for name in ("question_id", "query") if name not in payload]
        if missing:
            raise ValueError(f"query payload is missing fields: {', '.join(missing)}")
        return cls(
            question_id=payload["question_id"],
            query=payload["query"],
            metadata=payload.get("metadata", {}),
        )


def _nonempty_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _normalize_queries(
    queries: Sequence[QuerySpec | Mapping[str, Any]],
) -> tuple[QuerySpec, ...]:
    if isinstance(queries, (str, bytes)):
        raise TypeError("queries must be a sequence of QuerySpec objects or mappings")
    try:
        raw_queries = tuple(queries)
    except TypeError as exc:
        raise TypeError(
            "queries must be an iterable of QuerySpec objects or mappings"
        ) from exc
    if not raw_queries:
        raise ValueError("queries must contain at least one query")

    normalized: list[QuerySpec] = []
    for index, query in enumerate(raw_queries):
        if isinstance(query, QuerySpec):
            normalized.append(query)
        elif isinstance(query, Mapping):
            normalized.append(QuerySpec.from_mapping(query))
        else:
            raise TypeError(
                f"queries[{index}] must be a QuerySpec or mapping, "
                f"not {type(query).__name__}"
            )
    question_ids = [query.question_id for query in normalized]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("question_id values must be unique within one video")
    return tuple(normalized)


def _tensor_like_to_numpy(value: Any, name: str) -> np.ndarray:
    """Convert NumPy-, list-, or tensor-like output without importing torch."""

    converted = value
    for method_name in ("detach", "cpu"):
        method = getattr(converted, method_name, None)
        if callable(method):
            converted = method()
    numpy_method = getattr(converted, "numpy", None)
    if callable(numpy_method):
        converted = numpy_method()
    try:
        array = np.asarray(converted)
    except Exception as exc:
        raise TypeError(f"extractor {name} must be array-like") from exc
    if array.dtype.kind not in "iuf":
        raise TypeError(f"extractor {name} must contain real numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"extractor {name} must contain only finite values")
    return array


def _validate_extractor_output(
    output: Any,
    expected_length: int,
    *,
    question_id: str,
    origin_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        scores_raw, features_raw = output
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "extractor.compute must return exactly (scores, features)"
        ) from exc

    scores = _tensor_like_to_numpy(scores_raw, "scores")
    features = _tensor_like_to_numpy(features_raw, "features")
    context = f"question_id={question_id!r}, origin_id={origin_id}"
    if scores.ndim != 1:
        raise ValueError(f"extractor scores must be one-dimensional ({context})")
    if scores.shape[0] != expected_length:
        raise ValueError(
            f"extractor score length mismatch: expected {expected_length}, "
            f"got {scores.shape[0]} ({context})"
        )
    if features.ndim < 2:
        raise ValueError(
            f"extractor features must have shape [frame, ...] ({context})"
        )
    if features.shape[0] != expected_length:
        raise ValueError(
            f"extractor feature length mismatch: expected {expected_length}, "
            f"got {features.shape[0]} ({context})"
        )
    return np.asarray(scores, dtype=float), np.asarray(features)


def _validate_decoded_origins(
    manifest: SamplingManifest,
    decoded: Mapping[int, Sequence[DecodedFrameMatch]],
) -> dict[int, tuple[DecodedFrameMatch, ...]]:
    if not isinstance(decoded, Mapping):
        raise TypeError("decode_manifest_pyav must return a mapping by origin_id")
    expected_ids = {origin.origin_id for origin in manifest.origins}
    actual_ids = set(decoded)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "decoded origin IDs do not match the manifest "
            f"(missing={missing}, extra={extra})"
        )

    validated: dict[int, tuple[DecodedFrameMatch, ...]] = {}
    for origin in manifest.origins:
        try:
            matches = tuple(decoded[origin.origin_id])
        except TypeError as exc:
            raise TypeError(
                f"decoded origin {origin.origin_id} must be an iterable"
            ) from exc
        if len(matches) != manifest.candidate_count:
            raise ValueError(
                f"decoded origin {origin.origin_id} length mismatch: expected "
                f"{manifest.candidate_count}, got {len(matches)}"
            )
        for sample_index, (target, match) in enumerate(
            zip(origin.target_timestamps_sec, matches)
        ):
            if not isinstance(match, DecodedFrameMatch):
                raise TypeError(
                    f"decoded origin {origin.origin_id} sample {sample_index} "
                    "must be a DecodedFrameMatch"
                )
            if not np.isclose(
                match.target_timestamp_sec, target, rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    f"decoded target mismatch at origin {origin.origin_id}, "
                    f"sample {sample_index}"
                )
            if match.decoded_frame_index < 0:
                raise ValueError(
                    f"decoded_frame_index is missing at origin {origin.origin_id}, "
                    f"sample {sample_index}"
                )
        validated[origin.origin_id] = matches
    return validated


def _feature_path(
    output_dir: Path,
    dataset: str,
    video_id: str,
    question_id: str,
    origin_id: int,
) -> Path:
    logical_key = "\x1f".join(
        (dataset, video_id, question_id, str(origin_id), "visual_features")
    )
    digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()[:24]
    return output_dir / "visual_features" / f"o{origin_id:02d}_{digest}.npy"


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def preprocess_video_manifest(
    video_path: str | Path,
    manifest: SamplingManifest,
    *,
    dataset: str,
    queries: Sequence[QuerySpec | Mapping[str, Any]],
    extractor: FrameQueryExtractor,
    output_dir: str | Path,
    batch_size: int = 256,
    stream_index: int = 0,
    frame_adapter: Callable[[np.ndarray], Any] | None = None,
    record_metadata: Mapping[str, Any] | None = None,
) -> list[OriginSignalRecord]:
    """Decode a video once and extract one signal record per query and origin.

    ``frame_adapter`` is optional.  By default the extractor receives RGB
    NumPy arrays.  A caller whose processor specifically requires PIL images
    can pass ``PIL.Image.fromarray`` without making Pillow or a model framework
    a dependency of this module.
    """

    if not isinstance(manifest, SamplingManifest):
        raise TypeError("manifest must be a SamplingManifest")
    if manifest.candidate_count <= 0:
        raise ValueError("manifest must contain at least one candidate per origin")
    stable_dataset = _nonempty_text("dataset", dataset)
    stable_batch_size = _positive_integer("batch_size", batch_size)
    if isinstance(stream_index, bool) or not isinstance(stream_index, Integral):
        raise TypeError("stream_index must be an integer")
    stable_stream_index = int(stream_index)
    if stable_stream_index < 0:
        raise ValueError("stream_index must be non-negative")
    compute = getattr(extractor, "compute", None)
    if not callable(compute):
        raise TypeError(
            "extractor must provide compute(frames, query, batch_size)"
        )
    if frame_adapter is not None and not callable(frame_adapter):
        raise TypeError("frame_adapter must be callable or None")
    normalized_queries = _normalize_queries(queries)
    if record_metadata is None:
        shared_metadata: dict[str, Any] = {}
    elif not isinstance(record_metadata, Mapping):
        raise TypeError("record_metadata must be a mapping or None")
    else:
        shared_metadata = dict(record_metadata)
        try:
            json.dumps(shared_metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("record_metadata must be JSON-serializable") from exc
    output_root = Path(output_dir)

    # This is intentionally the only decode call.  The materialized RGB frames
    # below are retained and reused for every query on the same video.
    decoded = _validate_decoded_origins(
        manifest,
        decode_manifest_pyav(
            video_path,
            manifest,
            stream_index=stable_stream_index,
        ),
    )
    frames_by_origin: dict[int, tuple[Any, ...]] = {}
    for origin in manifest.origins:
        raw_frames = tuple(match.rgb for match in decoded[origin.origin_id])
        if frame_adapter is None:
            frames_by_origin[origin.origin_id] = raw_frames
        else:
            frames_by_origin[origin.origin_id] = tuple(
                frame_adapter(frame) for frame in raw_frames
            )

    records: list[OriginSignalRecord] = []
    resolved_video_path = str(Path(video_path).resolve())
    for query_spec in normalized_queries:
        for origin in manifest.origins:
            matches = decoded[origin.origin_id]
            frames = frames_by_origin[origin.origin_id]
            output = compute(frames, query_spec.query, stable_batch_size)
            scores, features = _validate_extractor_output(
                output,
                manifest.candidate_count,
                question_id=query_spec.question_id,
                origin_id=origin.origin_id,
            )

            feature_path = _feature_path(
                output_root,
                stable_dataset,
                manifest.video_id,
                query_spec.question_id,
                origin.origin_id,
            )
            _save_npy(feature_path, features)
            source_indices = tuple(int(match.decoded_frame_index) for match in matches)
            source_pts = tuple(float(match.actual_pts_sec) for match in matches)
            decode_errors = [float(match.decode_error_ms) for match in matches]
            metadata = {
                **shared_metadata,
                "query": query_spec.query,
                "query_metadata": dict(query_spec.metadata),
                "video_path": resolved_video_path,
                "sampling_manifest_schema_version": manifest.schema_version,
                "master_seed": manifest.master_seed,
                "sample_fps": manifest.sample_fps,
                "decode_error_ms": decode_errors,
                # Explicit aliases make frame provenance visible without
                # knowing the OriginSignalRecord field conventions.
                "decoded_frame_indices": list(source_indices),
                "source_pts_sec": list(source_pts),
                "feature_shape": list(features.shape),
                "feature_dtype": str(features.dtype),
            }
            records.append(
                OriginSignalRecord(
                    dataset=stable_dataset,
                    video_id=manifest.video_id,
                    question_id=query_spec.question_id,
                    origin_id=origin.origin_id,
                    origin_sec=origin.origin_sec,
                    timestamps_sec=origin.target_timestamps_sec,
                    actual_pts_sec=source_pts,
                    source_frame_indices=source_indices,
                    relevance_scores=tuple(float(value) for value in scores),
                    pixel_hashes=tuple(match.pixel_hash for match in matches),
                    visual_features_path=str(feature_path.resolve()),
                    metadata=metadata,
                )
            )
    return records


# Alias phrased around the return value, useful in small experiment scripts.
manifest_to_origin_signal_records = preprocess_video_manifest


def preprocess_video_manifest_streaming(
    video_path: str | Path,
    manifest: SamplingManifest,
    *,
    dataset: str,
    queries: Sequence[QuerySpec | Mapping[str, Any]],
    extractor: FrameQueryExtractor,
    output_dir: str | Path,
    batch_size: int = 256,
    frame_buffer_size: int = 64,
    stream_index: int = 0,
    frame_adapter: Callable[[np.ndarray], Any] | None = None,
    record_metadata: Mapping[str, Any] | None = None,
) -> list[OriginSignalRecord]:
    """Bounded-memory single-pass preprocessing for long benchmark videos.

    At most ``frame_buffer_size`` matched RGB targets are retained before they
    are sent through the extractor. Dense visual features are written into
    temporary NumPy memmaps and atomically promoted after the video completes.
    """

    if not isinstance(manifest, SamplingManifest) or manifest.candidate_count <= 0:
        raise ValueError("manifest must contain at least one candidate per origin")
    stable_dataset = _nonempty_text("dataset", dataset)
    stable_batch_size = _positive_integer("batch_size", batch_size)
    stable_buffer_size = _positive_integer("frame_buffer_size", frame_buffer_size)
    if isinstance(stream_index, bool) or not isinstance(stream_index, Integral):
        raise TypeError("stream_index must be an integer")
    stable_stream_index = int(stream_index)
    if stable_stream_index < 0:
        raise ValueError("stream_index must be non-negative")
    compute = getattr(extractor, "compute", None)
    if not callable(compute):
        raise TypeError("extractor must provide compute(frames, query, batch_size)")
    if frame_adapter is not None and not callable(frame_adapter):
        raise TypeError("frame_adapter must be callable or None")
    normalized_queries = _normalize_queries(queries)
    if record_metadata is None:
        shared_metadata: dict[str, Any] = {}
    elif not isinstance(record_metadata, Mapping):
        raise TypeError("record_metadata must be a mapping or None")
    else:
        shared_metadata = dict(record_metadata)
        try:
            json.dumps(shared_metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("record_metadata must be JSON-serializable") from exc

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    flat_targets: list[float] = []
    descriptors: list[tuple[int, int]] = []
    origin_by_id = {origin.origin_id: origin for origin in manifest.origins}
    for origin in manifest.origins:
        for sample_index, timestamp in enumerate(origin.target_timestamps_sec):
            flat_targets.append(timestamp)
            descriptors.append((origin.origin_id, sample_index))

    pending: dict[int, list[tuple[int, DecodedFrameMatch]]] = {
        origin.origin_id: [] for origin in manifest.origins
    }
    matches_by_origin: dict[int, list[DecodedFrameMatch | None]] = {
        origin.origin_id: [None] * manifest.candidate_count
        for origin in manifest.origins
    }
    score_arrays: dict[tuple[str, int], np.ndarray] = {
        (query.question_id, origin.origin_id): np.empty(
            manifest.candidate_count, dtype=float
        )
        for query in normalized_queries
        for origin in manifest.origins
    }
    feature_info: dict[tuple[str, int], tuple[Path, Path, tuple[int, ...], str]] = {}
    total_buffered = 0

    def write_feature_chunk(
        key: tuple[str, int],
        sample_indices: np.ndarray,
        chunk: np.ndarray,
    ) -> None:
        if key not in feature_info:
            feature_path = _feature_path(
                output_root,
                stable_dataset,
                manifest.video_id,
                key[0],
                key[1],
            )
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = feature_path.with_suffix(feature_path.suffix + ".tmp")
            full_shape = (manifest.candidate_count, *chunk.shape[1:])
            memmap = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=chunk.dtype,
                shape=full_shape,
            )
            del memmap
            feature_info[key] = (
                feature_path,
                temporary,
                tuple(full_shape),
                str(chunk.dtype),
            )
        _, temporary, full_shape, dtype_text = feature_info[key]
        if tuple(chunk.shape[1:]) != tuple(full_shape[1:]) or str(chunk.dtype) != dtype_text:
            raise ValueError(f"extractor feature shape/dtype changed across batches for {key}")
        memmap = np.lib.format.open_memmap(temporary, mode="r+")
        memmap[sample_indices] = chunk
        memmap.flush()
        del memmap

    def flush_pending() -> None:
        nonlocal total_buffered
        entries: list[tuple[int, int, DecodedFrameMatch]] = []
        for origin_id in sorted(pending):
            entries.extend(
                (origin_id, sample_index, match)
                for sample_index, match in pending[origin_id]
            )
        if not entries:
            return
        raw_frames = [entry[2].rgb for entry in entries]
        frames = (
            raw_frames
            if frame_adapter is None
            else [frame_adapter(frame) for frame in raw_frames]
        )
        for query in normalized_queries:
            scores, features = _validate_extractor_output(
                compute(frames, query.query, stable_batch_size),
                len(entries),
                question_id=query.question_id,
                origin_id=-1,
            )
            for origin_id in sorted(pending):
                positions = np.asarray(
                    [
                        position
                        for position, entry in enumerate(entries)
                        if entry[0] == origin_id
                    ],
                    dtype=int,
                )
                if positions.size == 0:
                    continue
                sample_indices = np.asarray(
                    [entries[position][1] for position in positions], dtype=int
                )
                key = (query.question_id, origin_id)
                score_arrays[key][sample_indices] = scores[positions]
                write_feature_chunk(key, sample_indices, features[positions])
        for origin_id, sample_index, match in entries:
            if matches_by_origin[origin_id][sample_index] is not None:
                raise RuntimeError("decoder yielded the same target more than once")
            matches_by_origin[origin_id][sample_index] = match
        for values in pending.values():
            values.clear()
        total_buffered = 0

    for flat_index, match in iter_nearest_frames_pyav_indexed(
        video_path,
        flat_targets,
        stream_index=stable_stream_index,
    ):
        if flat_index < 0 or flat_index >= len(descriptors):
            raise RuntimeError("decoder returned an invalid target index")
        origin_id, sample_index = descriptors[flat_index]
        pending[origin_id].append((sample_index, match))
        total_buffered += 1
        if total_buffered >= stable_buffer_size:
            flush_pending()
    flush_pending()

    for origin_id, matches in matches_by_origin.items():
        if any(match is None for match in matches):
            raise RuntimeError(f"origin {origin_id} has unmatched target frames")
    for key, (feature_path, temporary, _, _) in feature_info.items():
        if not temporary.is_file():
            raise RuntimeError(f"temporary feature file is missing for {key}")
        temporary.replace(feature_path)

    resolved_video_path = str(Path(video_path).resolve())
    records: list[OriginSignalRecord] = []
    for query in normalized_queries:
        for origin_id in sorted(origin_by_id):
            origin = origin_by_id[origin_id]
            complete_matches = tuple(
                match
                for match in matches_by_origin[origin_id]
                if match is not None
            )
            feature_path, _, feature_shape, feature_dtype = feature_info[
                (query.question_id, origin_id)
            ]
            source_indices = tuple(
                int(match.decoded_frame_index) for match in complete_matches
            )
            source_pts = tuple(float(match.actual_pts_sec) for match in complete_matches)
            metadata = {
                **shared_metadata,
                "query": query.query,
                "query_metadata": dict(query.metadata),
                "video_path": resolved_video_path,
                "sampling_manifest_schema_version": manifest.schema_version,
                "master_seed": manifest.master_seed,
                "sample_fps": manifest.sample_fps,
                "decode_error_ms": [
                    float(match.decode_error_ms) for match in complete_matches
                ],
                "decoded_frame_indices": list(source_indices),
                "source_pts_sec": list(source_pts),
                "feature_shape": list(feature_shape),
                "feature_dtype": feature_dtype,
                "streaming_decode": True,
                "frame_buffer_size": stable_buffer_size,
            }
            records.append(
                OriginSignalRecord(
                    dataset=stable_dataset,
                    video_id=manifest.video_id,
                    question_id=query.question_id,
                    origin_id=origin_id,
                    origin_sec=origin.origin_sec,
                    timestamps_sec=origin.target_timestamps_sec,
                    actual_pts_sec=source_pts,
                    source_frame_indices=source_indices,
                    relevance_scores=tuple(
                        float(value)
                        for value in score_arrays[(query.question_id, origin_id)]
                    ),
                    pixel_hashes=tuple(
                        match.pixel_hash for match in complete_matches
                    ),
                    visual_features_path=str(feature_path.resolve()),
                    metadata=metadata,
                )
            )
    return records


__all__ = [
    "FrameQueryExtractor",
    "QuerySpec",
    "manifest_to_origin_signal_records",
    "preprocess_video_manifest",
    "preprocess_video_manifest_streaming",
]
