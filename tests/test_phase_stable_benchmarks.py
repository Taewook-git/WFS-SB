import json
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
