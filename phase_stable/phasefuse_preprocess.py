"""Bounded-memory preprocessing for physical multi-phase video grids.

This module is deliberately separate from the PhaseFuse selector.  It owns the
expensive part of the protocol: all outer-origin and inner-phase target times
for one video are decoded in one sequential PyAV pass, while query-conditioned
scores and dense visual features are written incrementally.  A selector can
then be changed or ablated without decoding the source video again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import OriginSignalRecord
from .multiphase import (
    DenseOuterRecordPayload,
    MultiPhaseManifest,
    build_dense_outer_records,
)
from .preprocess import (
    FrameQueryExtractor,
    QuerySpec,
    _nonempty_text,
    _normalize_queries,
    _positive_integer,
    _validate_extractor_output,
)
from .sampling import DecodedFrameMatch, iter_nearest_frames_pyav_indexed

_MatchProvenance = tuple[float, float, str, int]


def _dense_feature_path(
    output_dir: Path,
    dataset: str,
    video_id: str,
    question_id: str,
    outer_origin_id: int,
) -> Path:
    logical_key = "\x1f".join(
        (
            dataset,
            video_id,
            question_id,
            str(outer_origin_id),
            "phasefuse_dense_visual_features",
        )
    )
    digest = hashlib.sha256(logical_key.encode("utf-8")).hexdigest()[:24]
    return output_dir / "visual_features" / f"o{outer_origin_id:02d}_{digest}.npy"


def _json_mapping(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc
    return result


def _validate_stream_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("stream_index must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError("stream_index must be non-negative")
    return result


def _match_key(match: DecodedFrameMatch) -> tuple[int, float]:
    if match.decoded_frame_index < 0:
        raise ValueError("decoded frame is missing decoded_frame_index provenance")
    return int(match.decoded_frame_index), float(match.actual_pts_sec)


def preprocess_multiphase_video_streaming(
    video_path: str | Path,
    manifest: MultiPhaseManifest,
    *,
    dataset: str,
    queries: Sequence[QuerySpec | Mapping[str, Any]],
    extractor: FrameQueryExtractor,
    output_dir: str | Path,
    batch_size: int = 32,
    frame_buffer_size: int = 256,
    stream_index: int = 0,
    frame_adapter: Callable[[np.ndarray], Any] | None = None,
    record_metadata: Mapping[str, Any] | None = None,
) -> list[OriginSignalRecord]:
    """Decode and score a multi-phase video using bounded host memory.

    The decoder is invoked exactly once.  Targets that resolve to the same
    source-frame index and PTS are kept in one contiguous flush group,
    evaluated once per query, and scattered back to every dense target.  A
    flush may exceed the nominal buffer by one repeated-source group; it never
    splits that group.  Feature artifacts remain dense-target aligned, which
    makes phase slicing an integer-index operation and preserves the physical
    target/actual-PTS distinction.
    """

    if not isinstance(manifest, MultiPhaseManifest):
        raise TypeError("manifest must be a MultiPhaseManifest")
    stable_dataset = _nonempty_text("dataset", dataset)
    stable_batch_size = _positive_integer("batch_size", batch_size)
    stable_buffer_size = _positive_integer("frame_buffer_size", frame_buffer_size)
    stable_stream_index = _validate_stream_index(stream_index)
    compute = getattr(extractor, "compute", None)
    if not callable(compute):
        raise TypeError("extractor must provide compute(frames, query, batch_size)")
    if frame_adapter is not None and not callable(frame_adapter):
        raise TypeError("frame_adapter must be callable or None")
    normalized_queries = _normalize_queries(queries)
    shared_metadata = _json_mapping("record_metadata", record_metadata)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    flat_targets: list[float] = []
    descriptors: list[tuple[int, int]] = []
    for origin in manifest.outer_origins:
        for sample_index, target in enumerate(origin.dense_target_timestamps_sec):
            flat_targets.append(float(target))
            descriptors.append((origin.outer_origin_id, sample_index))

    dense_count = manifest.dense_candidate_count
    matches_by_origin: dict[int, list[_MatchProvenance | None]] = {
        origin.outer_origin_id: [None] * dense_count
        for origin in manifest.outer_origins
    }
    score_arrays: dict[tuple[str, int], np.ndarray] = {
        (query.question_id, origin.outer_origin_id): np.empty(dense_count, dtype=float)
        for query in normalized_queries
        for origin in manifest.outer_origins
    }
    feature_info: dict[tuple[str, int], tuple[Path, Path, tuple[int, ...], str]] = {}
    pending: list[tuple[int, int, DecodedFrameMatch]] = []
    extractor_unique_frame_calls = 0
    decoder_target_matches = 0

    def write_feature_chunk(
        key: tuple[str, int],
        dense_indices: np.ndarray,
        chunk: np.ndarray,
    ) -> None:
        if key not in feature_info:
            final_path = _dense_feature_path(
                output_root,
                stable_dataset,
                manifest.video_id,
                key[0],
                key[1],
            )
            final_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = final_path.with_suffix(final_path.suffix + ".tmp")
            full_shape = (dense_count, *chunk.shape[1:])
            memmap = np.lib.format.open_memmap(
                temporary,
                mode="w+",
                dtype=chunk.dtype,
                shape=full_shape,
            )
            del memmap
            feature_info[key] = (
                final_path,
                temporary,
                tuple(int(value) for value in full_shape),
                str(chunk.dtype),
            )
        _, temporary, full_shape, dtype_text = feature_info[key]
        if tuple(chunk.shape[1:]) != tuple(full_shape[1:]):
            raise ValueError(f"extractor feature shape changed across batches for {key}")
        if str(chunk.dtype) != dtype_text:
            raise ValueError(f"extractor feature dtype changed across batches for {key}")
        memmap = np.lib.format.open_memmap(temporary, mode="r+")
        memmap[dense_indices] = chunk
        memmap.flush()
        del memmap

    def flush_pending() -> None:
        nonlocal extractor_unique_frame_calls
        if not pending:
            return
        unique_entries: list[DecodedFrameMatch] = []
        unique_by_key: dict[tuple[int, float], int] = {}
        entry_to_unique: list[int] = []
        for _, _, match in pending:
            key = _match_key(match)
            position = unique_by_key.get(key)
            if position is None:
                position = len(unique_entries)
                unique_by_key[key] = position
                unique_entries.append(match)
            elif unique_entries[position].pixel_hash != match.pixel_hash:
                raise ValueError("one decoded source frame maps to multiple pixel hashes")
            entry_to_unique.append(position)
        raw_frames = [match.rgb for match in unique_entries]
        frames = (
            raw_frames
            if frame_adapter is None
            else [frame_adapter(frame) for frame in raw_frames]
        )
        scatter = np.asarray(entry_to_unique, dtype=int)
        for query in normalized_queries:
            unique_scores, unique_features = _validate_extractor_output(
                compute(frames, query.query, stable_batch_size),
                len(unique_entries),
                question_id=query.question_id,
                origin_id=-1,
            )
            extractor_unique_frame_calls += len(unique_entries)
            scores = unique_scores[scatter]
            features = unique_features[scatter]
            for origin in manifest.outer_origins:
                positions = np.asarray(
                    [
                        index
                        for index, (origin_id, _, _) in enumerate(pending)
                        if origin_id == origin.outer_origin_id
                    ],
                    dtype=int,
                )
                if positions.size == 0:
                    continue
                dense_indices = np.asarray(
                    [pending[index][1] for index in positions], dtype=int
                )
                key = (query.question_id, origin.outer_origin_id)
                score_arrays[key][dense_indices] = scores[positions]
                write_feature_chunk(key, dense_indices, features[positions])
        for origin_id, dense_index, match in pending:
            if matches_by_origin[origin_id][dense_index] is not None:
                raise RuntimeError("decoder yielded the same dense target more than once")
            matches_by_origin[origin_id][dense_index] = (
                float(match.actual_pts_sec),
                float(match.decode_error_ms),
                match.pixel_hash,
                int(match.decoded_frame_index),
            )
        pending.clear()

    for flat_index, match in iter_nearest_frames_pyav_indexed(
        video_path,
        flat_targets,
        stream_index=stable_stream_index,
    ):
        if flat_index < 0 or flat_index >= len(descriptors):
            raise RuntimeError("decoder returned an invalid target index")
        origin_id, dense_index = descriptors[flat_index]
        if not isinstance(match, DecodedFrameMatch):
            raise TypeError("decoder must yield DecodedFrameMatch values")
        expected_target = flat_targets[flat_index]
        if not np.isclose(
            match.target_timestamp_sec,
            expected_target,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("decoded target timestamp does not match the manifest")
        if (
            pending
            and len(pending) >= stable_buffer_size
            and _match_key(match) != _match_key(pending[-1][2])
        ):
            flush_pending()
        pending.append((origin_id, dense_index, match))
        decoder_target_matches += 1
    flush_pending()

    expected_matches = manifest.num_outer_origins * dense_count
    if decoder_target_matches != expected_matches:
        raise RuntimeError(
            f"decoder returned {decoder_target_matches} targets; expected {expected_matches}"
        )
    for origin_id, matches in matches_by_origin.items():
        if any(match is None for match in matches):
            raise RuntimeError(f"outer origin {origin_id} has unmatched dense targets")
    expected_feature_keys = {
        (query.question_id, origin.outer_origin_id)
        for query in normalized_queries
        for origin in manifest.outer_origins
    }
    if set(feature_info) != expected_feature_keys:
        raise RuntimeError("extractor did not produce every query/origin feature artifact")
    for key, (final_path, temporary, _, _) in feature_info.items():
        if not temporary.is_file():
            raise RuntimeError(f"temporary feature file is missing for {key}")
        temporary.replace(final_path)

    resolved_video_path = str(Path(video_path).resolve())
    records: list[OriginSignalRecord] = []
    for query in normalized_queries:
        payloads: list[DenseOuterRecordPayload] = []
        for origin in manifest.outer_origins:
            complete_matches = tuple(
                match
                for match in matches_by_origin[origin.outer_origin_id]
                if match is not None
            )
            feature_path, _, feature_shape, feature_dtype = feature_info[
                (query.question_id, origin.outer_origin_id)
            ]
            payloads.append(
                DenseOuterRecordPayload(
                    outer_origin_id=origin.outer_origin_id,
                    actual_pts_sec=tuple(
                        match[0] for match in complete_matches
                    ),
                    source_frame_indices=tuple(
                        match[3] for match in complete_matches
                    ),
                    relevance_scores=tuple(
                        float(value)
                        for value in score_arrays[
                            (query.question_id, origin.outer_origin_id)
                        ]
                    ),
                    pixel_hashes=tuple(match[2] for match in complete_matches),
                    visual_features_path=str(feature_path.resolve()),
                    metadata={
                        "query": query.query,
                        "query_metadata": dict(query.metadata),
                        "video_path": resolved_video_path,
                        "feature_shape": list(feature_shape),
                        "feature_dtype": feature_dtype,
                        "feature_layout": "dense_target_aligned",
                        "decode_error_ms": [
                            match[1] for match in complete_matches
                        ],
                        "streaming_decode": True,
                        "frame_buffer_size": stable_buffer_size,
                    },
                )
            )
        records.extend(
            build_dense_outer_records(
                manifest,
                dataset=stable_dataset,
                question_id=query.question_id,
                payloads=payloads,
                metadata={
                    **shared_metadata,
                    "multiphase_decode_passes": 1,
                    "decoded_target_matches": decoder_target_matches,
                    "extractor_unique_frame_calls": extractor_unique_frame_calls,
                },
            )
        )
    return records


__all__ = ["preprocess_multiphase_video_streaming"]
