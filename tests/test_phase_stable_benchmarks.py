import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from phase_stable.benchmarks import (
    build_benchmark_manifests,
    iter_benchmark_signal_records,
    load_benchmark_videos,
)


def _write_annotations(path: Path, rows: list[dict]) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )


def _qv_row(*, video_id: str = "qv-video", query_id: str = "qv-query") -> dict:
    return {
        "video_id": video_id,
        "query_id": query_id,
        "query": "A person enters the room.",
        "duration": 8.0,
        "fps": 30.0,
        "relevant_frame_ranges": [[60, 119], [180, 239]],
        "metadata": {
            "relevant_windows_seconds": [[2.0, 4.0], [6.0, 8.0]],
            "relevant_clip_ids": [1, 3],
            "saliency_votes": [[4, 4, 4], [2, 2, 2]],
            "source_split": "val",
            "source_annotation_sha256": "a" * 64,
        },
    }


def test_videomme_groups_queries_on_shared_video(tmp_path: Path) -> None:
    annotations = tmp_path / "questions.json"
    _write_annotations(
        annotations,
        [
            {
                "video_id": "001",
                "videoID": "physical",
                "question_id": "001-1",
                "question": "First?",
                "options": ["A. yes", "B. no"],
                "answer": "A",
            },
            {
                "video_id": "001",
                "videoID": "physical",
                "question_id": "001-2",
                "question": "Second?",
                "options": ["A. left", "B. right"],
                "answer": "B",
            },
        ],
    )
    videos = load_benchmark_videos("videomme", annotations, tmp_path)
    assert len(videos) == 1
    assert videos[0].video_id == "001"
    assert videos[0].video_path == tmp_path / "data" / "physical.mp4"
    assert [query.question_id for query in videos[0].queries] == ["001-1", "001-2"]
    assert "A. yes" in videos[0].queries[0].query


def test_lvb_and_mlvu_identity_and_query_policy(tmp_path: Path) -> None:
    lvb_path = tmp_path / "lvb.json"
    _write_annotations(
        lvb_path,
        [
            {
                "video_id": "youtube-id",
                "id": "youtube-id_0",
                "video_path": "clip.mp4",
                "question": "Where?",
                "candidates": ["home", "work"],
                "correct_choice": 0,
                "duration": 12.5,
            }
        ],
    )
    mlvu_path = tmp_path / "mlvu.json"
    _write_annotations(
        mlvu_path,
        [
            {
                "video_name": "needle_1.mp4",
                "question_id": "Q1",
                "question": "What happened?",
                "candidates": ["one", "two"],
                "answer": "A",
                "duration": 15.0,
            }
        ],
    )
    lvb = load_benchmark_videos("longvideobench", lvb_path, tmp_path)[0]
    mlvu = load_benchmark_videos("mlvu", mlvu_path, tmp_path)[0]
    assert lvb.dataset == "lvb" and lvb.video_id == "youtube-id"
    assert lvb.queries[0].query.endswith("home work")
    assert mlvu.video_id == "needle_1"
    assert mlvu.queries[0].query == "What happened?"


def test_build_manifests_uses_annotation_duration(tmp_path: Path) -> None:
    path = tmp_path / "mlvu.json"
    _write_annotations(
        path,
        [
            {
                "video_name": "v.mp4",
                "question_id": "q",
                "question": "Q?",
                "candidates": [],
                "answer": "A",
                "duration": 10.0,
            }
        ],
    )
    videos = load_benchmark_videos("mlvu", path, tmp_path)
    manifests = build_benchmark_manifests(videos, num_origins=3, master_seed=4)
    assert len(manifests) == 1
    assert manifests[0].video_id == "v"
    assert manifests[0].num_origins == 3


def test_iter_records_rejects_unknown_manifest_video(tmp_path: Path) -> None:
    path = tmp_path / "mlvu.json"
    _write_annotations(
        path,
        [
            {
                "video_name": "v.mp4",
                "question_id": "q",
                "question": "Q?",
                "candidates": [],
                "answer": "A",
                "duration": 10.0,
            }
        ],
    )
    videos = load_benchmark_videos("mlvu", path, tmp_path)
    manifest = build_benchmark_manifests(videos)[0]
    object.__setattr__(manifest, "video_id", "missing")

    class Extractor:
        def compute(self, frames, query, batch_size):
            return np.ones(len(frames)), np.ones((len(frames), 2))

    with pytest.raises(ValueError, match="absent from annotations"):
        list(
            iter_benchmark_signal_records(
                videos,
                [manifest],
                extractor=Extractor(),
                output_dir=tmp_path,
            )
        )


def test_transformed_qv_jsonl_groups_queries_and_preserves_fidelity_metadata(
    tmp_path: Path,
) -> None:
    first = _qv_row(query_id="q1")
    second = _qv_row(query_id="q2")
    second["query"] = "The person sits down."
    second["metadata"] = {
        **second["metadata"],
        "relevant_windows_seconds": [[0.0, 2.0]],
        "relevant_clip_ids": [0],
        "saliency_votes": [[3, 2, 4]],
    }
    annotations = tmp_path / "qv_val.jsonl"
    _write_jsonl(annotations, [first, second])

    videos = load_benchmark_videos("qvh", annotations, tmp_path)

    assert len(videos) == 1
    video = videos[0]
    assert video.dataset == "qvhighlights"
    assert video.video_id == "qv-video"
    assert video.video_path == tmp_path / "videos" / "qv-video.mp4"
    assert video.duration_sec == pytest.approx(8.0)
    assert [query.question_id for query in video.queries] == ["q1", "q2"]
    assert [query.query for query in video.queries] == [
        "A person enters the room.",
        "The person sits down.",
    ]
    metadata = video.queries[0].metadata
    assert metadata["relevant_windows_sec"] == [[2.0, 4.0], [6.0, 8.0]]
    assert metadata["relevant_clip_ids"] == [1, 3]
    assert metadata["saliency_votes"] == [[4, 4, 4], [2, 2, 2]]
    assert metadata["annotation_fps"] == pytest.approx(30.0)
    assert metadata["relevant_frame_ranges"] == [[60, 119], [180, 239]]
    assert metadata["source_metadata"]["source_split"] == "val"
    assert metadata["source_metadata"]["source_annotation_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "case",
    [
        "missing_duration",
        "negative_clip_id",
        "fractional_clip_id",
        "duplicate_clip_id",
        "unsorted_clip_ids",
        "clip_id_out_of_range",
        "window_clip_mismatch",
        "vote_wrong_width",
        "vote_out_of_range",
        "vote_non_integer",
        "windows_unsorted",
        "windows_overlap",
    ],
)
def test_transformed_qv_rejects_invalid_fidelity_annotations(
    tmp_path: Path,
    case: str,
) -> None:
    row = deepcopy(_qv_row())
    metadata = row["metadata"]
    if case == "missing_duration":
        row.pop("duration")
    elif case == "negative_clip_id":
        metadata["relevant_clip_ids"] = [-1, 3]
    elif case == "fractional_clip_id":
        metadata["relevant_clip_ids"] = [1.5, 3]
    elif case == "duplicate_clip_id":
        metadata["relevant_clip_ids"] = [1, 1]
    elif case == "unsorted_clip_ids":
        metadata["relevant_clip_ids"] = [3, 1]
        metadata["saliency_votes"] = [[2, 2, 2], [4, 4, 4]]
    elif case == "clip_id_out_of_range":
        metadata["relevant_clip_ids"] = [1, 4]
    elif case == "window_clip_mismatch":
        metadata["relevant_clip_ids"] = [1, 2]
    elif case == "vote_wrong_width":
        metadata["saliency_votes"] = [[4, 4], [2, 2, 2]]
    elif case == "vote_out_of_range":
        metadata["saliency_votes"] = [[5, 4, 4], [2, 2, 2]]
    elif case == "vote_non_integer":
        metadata["saliency_votes"] = [[3.5, 4, 4], [2, 2, 2]]
    elif case == "windows_unsorted":
        metadata["relevant_windows_seconds"] = [[6.0, 8.0], [2.0, 4.0]]
        metadata["relevant_clip_ids"] = [3, 1]
        metadata["saliency_votes"] = [[2, 2, 2], [4, 4, 4]]
    elif case == "windows_overlap":
        metadata["relevant_windows_seconds"] = [[2.0, 6.0], [4.0, 8.0]]
        metadata["relevant_clip_ids"] = [1, 2, 3]
        metadata["saliency_votes"] = [[4, 4, 4], [3, 3, 3], [2, 2, 2]]
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)
    annotations = tmp_path / f"invalid_{case}.jsonl"
    _write_jsonl(annotations, [row])

    with pytest.raises(ValueError):
        load_benchmark_videos("qvhighlights", annotations, tmp_path)


def test_transformed_qv_requires_globally_unique_query_ids(tmp_path: Path) -> None:
    annotations = tmp_path / "duplicate_qid.jsonl"
    _write_jsonl(
        annotations,
        [
            _qv_row(video_id="video-a", query_id="shared-query"),
            _qv_row(video_id="video-b", query_id="shared-query"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate.*query"):
        load_benchmark_videos("qvhighlights", annotations, tmp_path)


def test_transformed_qv_jsonl_reports_malformed_line(tmp_path: Path) -> None:
    annotations = tmp_path / "malformed.jsonl"
    annotations.write_text(
        f"{json.dumps(_qv_row())}\n{{not-json}}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        load_benchmark_videos("qvhighlights", annotations, tmp_path)
