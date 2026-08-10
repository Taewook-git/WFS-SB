"""Benchmark catalog and preprocessing orchestration for phase-stable runs."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from .artifacts import OriginSignalRecord, write_signal_records
from .preprocess import (
    FrameQueryExtractor,
    QuerySpec,
    preprocess_video_manifest,
    preprocess_video_manifest_streaming,
)
from .sampling import SamplingManifest, build_sampling_manifest


SUPPORTED_BENCHMARKS = {"videomme", "lvb", "longvideobench", "mlvu"}


def normalize_benchmark(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized == "longvideobench":
        normalized = "lvb"
    if normalized not in {"videomme", "lvb", "mlvu"}:
        raise ValueError(f"unsupported benchmark: {name!r}")
    return normalized


@dataclass(frozen=True)
class BenchmarkVideo:
    """One unique source video and every query that shares its origin grid."""

    dataset: str
    video_id: str
    video_path: Path
    queries: tuple[QuerySpec, ...]
    duration_sec: Optional[float] = None

    def __post_init__(self) -> None:
        dataset = normalize_benchmark(self.dataset)
        video_id = str(self.video_id).strip()
        if not video_id:
            raise ValueError("video_id must be non-empty")
        path = Path(self.video_path)
        queries = tuple(self.queries)
        if not queries or any(not isinstance(query, QuerySpec) for query in queries):
            raise ValueError("queries must contain QuerySpec objects")
        question_ids = [query.question_id for query in queries]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(f"duplicate question_id for video {video_id!r}")
        duration = self.duration_sec
        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("duration_sec must be finite and positive")
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "video_path", path)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "duration_sec", duration)


def _load_rows(path: str | Path) -> list[Mapping[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or any(not isinstance(row, Mapping) for row in payload):
        raise ValueError("benchmark annotation must be a JSON list of objects")
    return payload


def _required_text(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ValueError(f"annotation row {row_index} has invalid {field!r}")
    result = str(value).strip()
    if not result:
        raise ValueError(f"annotation row {row_index} has empty {field!r}")
    return result


def _optional_duration(row: Mapping[str, Any]) -> Optional[float]:
    value = row.get("duration")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    duration = float(value)
    return duration if math.isfinite(duration) and duration > 0 else None


def _query_text(question: str, choices: Sequence[Any], *, include_choices: bool) -> str:
    if not include_choices:
        return question
    choice_text = " ".join(str(value).strip() for value in choices if str(value).strip())
    return f"{question} {choice_text}".strip()


def load_benchmark_videos(
    benchmark: str,
    questions_file: str | Path,
    dataset_root: str | Path,
) -> list[BenchmarkVideo]:
    """Group official annotations by physical video with stable query IDs."""

    dataset = normalize_benchmark(benchmark)
    root = Path(dataset_root)
    rows = _load_rows(questions_file)
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for row_index, row in enumerate(rows):
        question = _required_text(row, "question", row_index)
        duration = _optional_duration(row)
        if dataset == "videomme":
            video_id = _required_text(row, "video_id", row_index)
            physical_id = _required_text(row, "videoID", row_index)
            question_id = _required_text(row, "question_id", row_index)
            video_path = root / "data" / f"{physical_id}.mp4"
            choices = row.get("options", [])
            query = _query_text(question, choices, include_choices=True)
        elif dataset == "lvb":
            video_id = _required_text(row, "video_id", row_index)
            question_id = _required_text(row, "id", row_index)
            video_path = root / "videos" / _required_text(row, "video_path", row_index)
            choices = row.get("candidates", [])
            query = _query_text(question, choices, include_choices=True)
        else:
            video_name = _required_text(row, "video_name", row_index)
            video_id = Path(video_name).stem
            question_id = _required_text(row, "question_id", row_index)
            video_path = root / "video" / video_name
            choices = row.get("candidates", [])
            query = _query_text(question, choices, include_choices=False)

        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ValueError(f"annotation row {row_index} choices must be a sequence")
        query_spec = QuerySpec(
            question_id=question_id,
            query=query,
            metadata={
                "annotation_index": row_index,
                "gold": row.get("answer", row.get("correct_choice")),
                "choices": list(choices),
            },
        )
        if video_id not in groups:
            groups[video_id] = {
                "path": video_path,
                "duration": duration,
                "queries": [],
            }
            order.append(video_id)
        group = groups[video_id]
        if Path(group["path"]) != video_path:
            raise ValueError(f"video_id {video_id!r} maps to multiple video paths")
        if duration is not None:
            prior = group["duration"]
            if prior is not None and not math.isclose(
                prior, duration, rel_tol=1e-6, abs_tol=0.1
            ):
                raise ValueError(f"video_id {video_id!r} has inconsistent durations")
            group["duration"] = duration if prior is None else max(prior, duration)
        group["queries"].append(query_spec)

    return [
        BenchmarkVideo(
            dataset=dataset,
            video_id=video_id,
            video_path=groups[video_id]["path"],
            duration_sec=groups[video_id]["duration"],
            queries=tuple(groups[video_id]["queries"]),
        )
        for video_id in order
    ]


def probe_video_duration_pyav(video_path: str | Path, *, stream_index: int = 0) -> float:
    """Read a video's presentation duration without relying on average FPS."""

    try:
        av = importlib.import_module("av")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PyAV is required; install it with `pip install av`.") from exc
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"video file does not exist: {path}")
    container = av.open(str(path))
    try:
        streams = container.streams.video
        if stream_index < 0 or stream_index >= len(streams):
            raise ValueError(f"invalid video stream index {stream_index} for {path}")
        stream = streams[stream_index]
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        else:
            duration = 0.0
            previous_pts: Optional[float] = None
            last_step = 0.0
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    continue
                pts = float(frame.pts * frame.time_base)
                if previous_pts is not None and pts >= previous_pts:
                    last_step = pts - previous_pts
                previous_pts = pts
                duration = max(duration, pts + max(last_step, 0.0))
    finally:
        container.close()
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"could not determine a positive duration for {path}")
    return duration


def build_benchmark_manifests(
    videos: Sequence[BenchmarkVideo],
    *,
    master_seed: int = 0,
    num_origins: int = 5,
    sample_fps: float = 1.0,
    probe_missing_duration: bool = True,
) -> list[SamplingManifest]:
    manifests: list[SamplingManifest] = []
    for video in videos:
        duration = video.duration_sec
        if duration is None:
            if not probe_missing_duration:
                raise ValueError(f"duration is missing for video {video.video_id!r}")
            duration = probe_video_duration_pyav(video.video_path)
        manifests.append(
            build_sampling_manifest(
                video.video_id,
                duration,
                master_seed=master_seed,
                num_origins=num_origins,
                sample_fps=sample_fps,
            )
        )
    return manifests


def iter_benchmark_signal_records(
    videos: Sequence[BenchmarkVideo],
    manifests: Iterable[SamplingManifest],
    *,
    extractor: FrameQueryExtractor,
    output_dir: str | Path,
    batch_size: int = 256,
    frame_adapter: Any = None,
    record_metadata: Optional[Mapping[str, Any]] = None,
    streaming: bool = True,
    frame_buffer_size: int = 64,
) -> Iterator[OriginSignalRecord]:
    """Preprocess selected benchmark videos, streaming records to the caller."""

    video_by_id = {video.video_id: video for video in videos}
    if len(video_by_id) != len(videos):
        raise ValueError("videos must have unique video_id values")
    seen: set[str] = set()
    for manifest in manifests:
        if manifest.video_id in seen:
            raise ValueError(f"duplicate manifest for video {manifest.video_id!r}")
        seen.add(manifest.video_id)
        try:
            video = video_by_id[manifest.video_id]
        except KeyError as exc:
            raise ValueError(
                f"manifest video {manifest.video_id!r} is absent from annotations"
            ) from exc
        preprocess_function = (
            preprocess_video_manifest_streaming
            if streaming
            else preprocess_video_manifest
        )
        extra_kwargs = (
            {"frame_buffer_size": frame_buffer_size} if streaming else {}
        )
        yield from preprocess_function(
            video.video_path,
            manifest,
            dataset=video.dataset,
            queries=video.queries,
            extractor=extractor,
            output_dir=output_dir,
            batch_size=batch_size,
            frame_adapter=frame_adapter,
            record_metadata=record_metadata,
            **extra_kwargs,
        )


def preprocess_benchmark_to_jsonl(
    videos: Sequence[BenchmarkVideo],
    manifests: Iterable[SamplingManifest],
    *,
    extractor: FrameQueryExtractor,
    output_dir: str | Path,
    signal_jsonl: str | Path,
    batch_size: int = 256,
    frame_adapter: Any = None,
    record_metadata: Optional[Mapping[str, Any]] = None,
    streaming: bool = True,
    frame_buffer_size: int = 64,
) -> Path:
    records = iter_benchmark_signal_records(
        videos,
        manifests,
        extractor=extractor,
        output_dir=output_dir,
        batch_size=batch_size,
        frame_adapter=frame_adapter,
        record_metadata=record_metadata,
        streaming=streaming,
        frame_buffer_size=frame_buffer_size,
    )
    destination = Path(signal_jsonl)
    write_signal_records(destination, records)
    return destination


__all__ = [
    "BenchmarkVideo",
    "SUPPORTED_BENCHMARKS",
    "build_benchmark_manifests",
    "iter_benchmark_signal_records",
    "load_benchmark_videos",
    "normalize_benchmark",
    "preprocess_benchmark_to_jsonl",
    "probe_video_duration_pyav",
]
