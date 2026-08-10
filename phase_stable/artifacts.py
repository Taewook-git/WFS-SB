"""Versioned JSONL/NPZ artifacts for phase-stability experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np

from .pipeline import SelectionTrace


SIGNAL_SCHEMA_VERSION = 1
TRACE_SCHEMA_VERSION = 1


def _json_default(value: Any) -> Any:
    """Convert common experiment objects to lossless JSON-compatible values."""

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite_float(name: str, value: Any, *, minimum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _integer(name: str, value: Any, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _float_tuple(name: str, values: Sequence[Any]) -> tuple[float, ...]:
    try:
        result = tuple(_finite_float(f"{name}[{index}]", value) for index, value in enumerate(values))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of real numbers") from exc
    return result


def _int_tuple(name: str, values: Sequence[Any]) -> tuple[int, ...]:
    try:
        result = tuple(_integer(f"{name}[{index}]", value, minimum=0) for index, value in enumerate(values))
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    return result


@dataclass(frozen=True)
class OriginSignalRecord:
    """One query-conditioned signal for one real sampling origin."""

    dataset: str
    video_id: str
    question_id: str
    origin_id: int
    origin_sec: float
    timestamps_sec: tuple[float, ...]
    actual_pts_sec: tuple[float, ...]
    source_frame_indices: tuple[int, ...]
    relevance_scores: tuple[float, ...]
    pixel_hashes: tuple[str, ...] = ()
    visual_features_path: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SIGNAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        dataset = _nonempty_text("dataset", self.dataset)
        video_id = _nonempty_text("video_id", self.video_id)
        question_id = _nonempty_text("question_id", self.question_id)
        origin_id = _integer("origin_id", self.origin_id, minimum=0)
        origin_sec = _finite_float("origin_sec", self.origin_sec, minimum=0.0)
        timestamps = _float_tuple("timestamps_sec", self.timestamps_sec)
        actual_pts = _float_tuple("actual_pts_sec", self.actual_pts_sec)
        source_indices = _int_tuple("source_frame_indices", self.source_frame_indices)
        scores = _float_tuple("relevance_scores", self.relevance_scores)
        schema_version = _integer("schema_version", self.schema_version, minimum=1)
        if schema_version != SIGNAL_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported signal schema_version={schema_version}; "
                f"expected {SIGNAL_SCHEMA_VERSION}"
            )
        if not timestamps:
            raise ValueError("timestamps_sec must not be empty")
        lengths = {len(timestamps), len(actual_pts), len(source_indices), len(scores)}
        if len(lengths) != 1:
            raise ValueError(
                "timestamps_sec, actual_pts_sec, source_frame_indices, and "
                "relevance_scores must have equal lengths"
            )
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps_sec must be strictly increasing")
        if any(current < previous for previous, current in zip(actual_pts, actual_pts[1:])):
            raise ValueError("actual_pts_sec must be non-decreasing")
        if any(
            current < previous
            for previous, current in zip(source_indices, source_indices[1:])
        ):
            raise ValueError("source_frame_indices must be non-decreasing")
        if not np.all(np.isfinite(np.asarray(scores))):
            raise ValueError("relevance_scores must be finite")
        hashes = tuple(str(value) for value in self.pixel_hashes)
        if hashes and len(hashes) != len(timestamps):
            raise ValueError("pixel_hashes must be empty or frame-aligned")
        feature_path = self.visual_features_path
        if feature_path is not None:
            feature_path = _nonempty_text("visual_features_path", feature_path)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "question_id", question_id)
        object.__setattr__(self, "origin_id", origin_id)
        object.__setattr__(self, "origin_sec", origin_sec)
        object.__setattr__(self, "timestamps_sec", timestamps)
        object.__setattr__(self, "actual_pts_sec", actual_pts)
        object.__setattr__(self, "source_frame_indices", source_indices)
        object.__setattr__(self, "relevance_scores", scores)
        object.__setattr__(self, "pixel_hashes", hashes)
        object.__setattr__(self, "visual_features_path", feature_path)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def item_key(self) -> tuple[str, str, str]:
        return self.dataset, self.video_id, self.question_id

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamps_sec"] = list(self.timestamps_sec)
        payload["actual_pts_sec"] = list(self.actual_pts_sec)
        payload["source_frame_indices"] = list(self.source_frame_indices)
        payload["relevance_scores"] = list(self.relevance_scores)
        payload["pixel_hashes"] = list(self.pixel_hashes)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OriginSignalRecord":
        if not isinstance(payload, Mapping):
            raise TypeError("signal payload must be a mapping")
        required = (
            "dataset",
            "video_id",
            "question_id",
            "origin_id",
            "origin_sec",
            "timestamps_sec",
            "actual_pts_sec",
            "source_frame_indices",
            "relevance_scores",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"signal payload is missing fields: {', '.join(missing)}")
        return cls(
            dataset=payload["dataset"],
            video_id=payload["video_id"],
            question_id=payload["question_id"],
            origin_id=payload["origin_id"],
            origin_sec=payload["origin_sec"],
            timestamps_sec=tuple(payload["timestamps_sec"]),
            actual_pts_sec=tuple(payload["actual_pts_sec"]),
            source_frame_indices=tuple(payload["source_frame_indices"]),
            relevance_scores=tuple(payload["relevance_scores"]),
            pixel_hashes=tuple(payload.get("pixel_hashes", ())),
            visual_features_path=payload.get("visual_features_path"),
            metadata=payload.get("metadata", {}),
            schema_version=payload.get("schema_version", SIGNAL_SCHEMA_VERSION),
        )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write generic JSON-compatible rows atomically enough for experiment use."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=_json_default,
                )
            )
            handle.write("\n")
    temporary.replace(destination)


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {source}:{line_number}")
            yield value


def write_signal_records(path: str | Path, records: Iterable[OriginSignalRecord]) -> None:
    write_jsonl(path, (record.to_dict() for record in records))


def read_signal_records(path: str | Path) -> list[OriginSignalRecord]:
    records = []
    for line_number, payload in enumerate(iter_jsonl(path), start=1):
        try:
            records.append(OriginSignalRecord.from_dict(payload))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid signal record at line {line_number}: {exc}") from exc
    return records


def artifact_id(
    dataset: str,
    video_id: str,
    question_id: str,
    origin_id: int,
    method: str,
) -> str:
    """Stable collision-resistant ID that is safe as a Windows filename."""

    logical = "\x1f".join(
        [dataset, video_id, question_id, str(origin_id), method]
    )
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()[:20]
    return f"o{int(origin_id):02d}_{method}_{digest}"


def save_trace_npz(
    output_dir: str | Path,
    record: OriginSignalRecord,
    method: str,
    trace: SelectionTrace,
) -> tuple[Path, Dict[str, Any]]:
    """Persist dense arrays to NPZ and return a compact JSONL trace row."""

    root = Path(output_dir)
    arrays_dir = root / "trace_arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    name = artifact_id(
        record.dataset,
        record.video_id,
        record.question_id,
        record.origin_id,
        method,
    )
    array_path = arrays_dir / f"{name}.npz"
    np.savez_compressed(
        array_path,
        relevance_scores=np.asarray(record.relevance_scores, dtype=float),
        representation=trace.transform.representation,
        coarse_detail=trace.transform.coarse_detail,
        saliency=trace.transform.saliency,
        saliency_norm=trace.transform.normalized_saliency,
        scale_energy_proportions=trace.transform.scale_energy_proportions,
    )

    timestamps = np.asarray(record.timestamps_sec, dtype=float)
    selected_indices = np.asarray(trace.selected_indices, dtype=int)
    peak_indices = np.asarray(trace.peaks, dtype=int)
    decode_error_ms = 1000.0 * np.abs(
        np.asarray(record.actual_pts_sec, dtype=float) - timestamps
    )
    row = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "dataset": record.dataset,
        "video_id": record.video_id,
        "question_id": record.question_id,
        "origin_id": record.origin_id,
        "origin_sec": record.origin_sec,
        "method": method,
        "timestamps_sec": list(record.timestamps_sec),
        "actual_pts_sec": list(record.actual_pts_sec),
        "decode_error_ms_mean": float(np.mean(decode_error_ms)),
        "decode_error_ms_max": float(np.max(decode_error_ms)),
        "source_frame_indices": list(record.source_frame_indices),
        "pixel_hashes": list(record.pixel_hashes),
        "visual_features_path": record.visual_features_path,
        "record_metadata": dict(record.metadata),
        "array_path": str(array_path.resolve()),
        "peaks": peak_indices.tolist(),
        "peaks_sec": timestamps[peak_indices].tolist(),
        "segments": [list(segment) for segment in trace.segments],
        "valid_segments": [list(segment) for segment in trace.valid_segments],
        "importance_scores": [float(value) for value in trace.importance_scores],
        "valid_importance_scores": [
            float(value) for value in trace.valid_importance_scores
        ],
        "allocation": {str(key): int(value) for key, value in trace.allocation.items()},
        "selected_indices": selected_indices.tolist(),
        "selected_timestamps_sec": timestamps[selected_indices].tolist(),
        "selected_actual_pts_sec": np.asarray(record.actual_pts_sec)[selected_indices].tolist(),
        "selected_source_frame_indices": np.asarray(record.source_frame_indices)[
            selected_indices
        ].astype(int).tolist(),
        "selected_pixel_hashes": (
            []
            if not record.pixel_hashes
            else np.asarray(record.pixel_hashes, dtype=str)[selected_indices].tolist()
        ),
        "used_fallback": trace.used_fallback,
        "transform": trace.transform.summary(),
    }
    return array_path, row


def load_trace_arrays(trace_row: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    path_value = trace_row.get("array_path")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("trace row has no array_path")
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(f"trace array does not exist: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "relevance_scores",
            "representation",
            "coarse_detail",
            "saliency",
            "saliency_norm",
            "scale_energy_proportions",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"trace array is missing: {', '.join(missing)}")
        return {name: np.asarray(payload[name]) for name in required}


__all__ = [
    "OriginSignalRecord",
    "SIGNAL_SCHEMA_VERSION",
    "TRACE_SCHEMA_VERSION",
    "artifact_id",
    "iter_jsonl",
    "load_trace_arrays",
    "read_signal_records",
    "save_trace_npz",
    "write_jsonl",
    "write_signal_records",
]
