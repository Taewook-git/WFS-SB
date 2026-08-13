"""Query-only QV holdout loader used before the Stage-0 artifact seal."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .benchmarks import BenchmarkVideo
from .preprocess import QuerySpec


def load_blind_qv_query_videos(
    query_manifest: str | Path, dataset_root: str | Path
) -> list[BenchmarkVideo]:
    """Load only query text/identity/duration; label-bearing schemas are rejected."""

    root = Path(dataset_root)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        Path(query_manifest).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"query-only manifest row {line_number} is not an object")
        rows.append(value)
    videos: list[BenchmarkVideo] = []
    seen: set[tuple[str, str]] = set()
    prohibited = {"relevant_windows", "relevant_clip_ids", "saliency_scores"}
    expected_row_keys = {"video_id", "query_id", "query", "duration", "metadata"}
    expected_metadata_keys = {
        "annotation_row_index",
        "source_id",
        "label_firewall",
    }
    for index, row in enumerate(rows):
        if set(row) != expected_row_keys:
            raise ValueError(f"query-only row {index} schema drift")
        if prohibited & set(row):
            raise ValueError(f"query-only row {index} contains evaluation labels")
        metadata = row.get("metadata")
        if (
            not isinstance(metadata, Mapping)
            or set(metadata) != expected_metadata_keys
            or prohibited & set(metadata)
        ):
            raise ValueError(f"query-only row {index} has invalid metadata")
        try:
            video_id = str(row["video_id"]).strip()
            question_id = str(row["query_id"]).strip()
            query = str(row["query"]).strip()
            duration = float(row["duration"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"query-only row {index} is malformed") from exc
        if (
            not video_id
            or not question_id
            or not query
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise ValueError(f"query-only row {index} has invalid values")
        if (
            int(metadata["annotation_row_index"]) < 0
            or str(metadata["source_id"]) != video_id.rsplit("_", 2)[0]
            or metadata["label_firewall"] != "query_only_before_blind_seal"
        ):
            raise ValueError(f"query-only row {index} metadata binding drift")
        identity = (video_id, question_id)
        if identity in seen:
            raise ValueError(f"duplicate query-only identity: {identity}")
        seen.add(identity)
        videos.append(
            BenchmarkVideo(
                dataset="qvhighlights",
                video_id=video_id,
                video_path=root / "videos" / f"{video_id}.mp4",
                duration_sec=duration,
                queries=(QuerySpec(question_id, query, dict(metadata)),),
            )
        )
    if len(videos) != 100:
        raise ValueError("blind QV query manifest must contain exactly 100 rows")
    return videos


__all__ = ["load_blind_qv_query_videos"]
