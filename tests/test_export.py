import json
from pathlib import Path

import pytest

from phase_stable.artifacts import write_jsonl
from phase_stable.export import (
    ExportValidationError,
    build_lmms_keyframe_annotations,
    export_lmms_keyframe_jsons,
    export_trace_jsonl,
)


def _write_annotations(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _trace(
    dataset: str,
    video_id: str,
    question_id: str,
    *,
    method: str = "dwt",
    origin_id: int = 0,
    frames: tuple[int, ...] = (2, 8, 14),
) -> dict:
    return {
        "dataset": dataset,
        "video_id": video_id,
        "question_id": question_id,
        "method": method,
        "origin_id": origin_id,
        "selected_source_frame_indices": list(frames),
    }


def test_videomme_exports_each_method_origin_and_maps_shared_video_questions(
    tmp_path: Path,
) -> None:
    annotations = [
        {
            "video_id": "001",
            "videoID": "youtube-a",
            "question_id": "001-1",
            "question": "first",
            "options": ["A", "B"],
        },
        {
            "video_id": "001",
            "videoID": "youtube-a",
            "question_id": "001-2",
            "question": "second",
            "options": ["A", "B"],
        },
        {
            "video_id": "002",
            "videoID": "youtube-b",
            "question_id": "002-1",
            "question": "third",
            "options": ["A", "B"],
        },
    ]
    questions_file = _write_annotations(tmp_path / "videomme.json", annotations)
    rows = []
    identities = (("001", "001-1"), ("001", "001-2"), ("002", "002-1"))
    for method_index, method in enumerate(("dwt", "swt")):
        for origin_id in (0, 1):
            for item_index, (video_id, question_id) in enumerate(identities):
                start = 2 + method_index * 20 + origin_id * 8 + item_index
                rows.append(
                    _trace(
                        "videomme",
                        video_id,
                        question_id,
                        method=method,
                        origin_id=origin_id,
                        frames=(start, start + 2, start + 4),
                    )
                )

    output_paths = export_lmms_keyframe_jsons(
        rows,
        benchmark="videomme",
        questions_file=questions_file,
        dataset_root=tmp_path,
        output_dir=tmp_path / "exports",
        expected_budget=3,
    )

    assert set(output_paths) == {("dwt", 0), ("dwt", 1), ("swt", 0), ("swt", 1)}
    dwt_origin_one = json.loads(output_paths[("dwt", 1)].read_text(encoding="utf-8"))
    assert [row["question_id"] for row in dwt_origin_one] == ["001-1", "001-2", "002-1"]
    assert dwt_origin_one[0]["keyframe_indices"] == [10, 12, 14]
    assert dwt_origin_one[1]["keyframe_indices"] == [11, 13, 15]
    assert dwt_origin_one[0]["question"] == "first"


@pytest.mark.parametrize(
    ("benchmark", "annotation", "video_id", "question_id"),
    [
        (
            "lvb",
            {
                "video_id": "lvb-video",
                "id": "lvb-video_0",
                "video_path": "lvb-video.mp4",
                "question": "q",
            },
            "lvb-video",
            "lvb-video_0",
        ),
        (
            "mlvu",
            {
                "video_name": "clip.one.mp4",
                "question_id": "Q7",
                "question": "q",
            },
            "clip.one",
            "Q7",
        ),
    ],
)
def test_lvb_and_mlvu_use_official_adapter_output_shape(
    tmp_path: Path,
    benchmark: str,
    annotation: dict,
    video_id: str,
    question_id: str,
) -> None:
    questions_file = _write_annotations(tmp_path / f"{benchmark}.json", [annotation])
    exports = build_lmms_keyframe_annotations(
        [_trace(benchmark, video_id, question_id)],
        benchmark=benchmark,
        questions_file=questions_file,
        dataset_root=tmp_path,
        expected_budget=3,
    )
    output = exports[("dwt", 0)][0]
    assert output["keyframe_indices"] == [2, 8, 14]
    for name, value in annotation.items():
        assert output[name] == value


def test_export_trace_jsonl_round_trip_and_method_filter(tmp_path: Path) -> None:
    annotations = [
        {
            "video_name": "video.mp4",
            "question_id": "Q1",
            "question": "q",
        }
    ]
    questions_file = _write_annotations(tmp_path / "mlvu.json", annotations)
    trace_path = tmp_path / "traces.jsonl"
    write_jsonl(
        trace_path,
        [
            _trace("mlvu", "video", "Q1", method="dwt"),
            _trace("mlvu", "video.mp4", "Q1", method="swt"),
        ],
    )
    paths = export_trace_jsonl(
        trace_path,
        benchmark="mlvu",
        questions_file=questions_file,
        output_dir=tmp_path / "out",
        methods=("swt",),
        expected_budget=3,
    )
    assert set(paths) == {("swt", 0)}
    assert paths[("swt", 0)].name == "mlvu_swt_origin00.json"


def test_strict_export_rejects_missing_annotation_trace(tmp_path: Path) -> None:
    questions_file = _write_annotations(
        tmp_path / "videomme.json",
        [
            {"video_id": "001", "videoID": "a", "question_id": "001-1"},
            {"video_id": "001", "videoID": "a", "question_id": "001-2"},
        ],
    )
    with pytest.raises(ExportValidationError, match="missing 1 trace row"):
        build_lmms_keyframe_annotations(
            [_trace("videomme", "001", "001-1")],
            benchmark="videomme",
            questions_file=questions_file,
        )


def test_strict_export_rejects_duplicate_trace(tmp_path: Path) -> None:
    questions_file = _write_annotations(
        tmp_path / "lvb.json",
        [{"video_id": "v", "id": "v_0", "video_path": "v.mp4"}],
    )
    row = _trace("lvb", "v", "v_0")
    with pytest.raises(ExportValidationError, match="duplicate trace"):
        build_lmms_keyframe_annotations(
            [row, dict(row)],
            benchmark="lvb",
            questions_file=questions_file,
        )


def test_strict_export_rejects_duplicate_selected_frames(tmp_path: Path) -> None:
    questions_file = _write_annotations(
        tmp_path / "mlvu.json",
        [{"video_name": "v.mp4", "question_id": "Q0"}],
    )
    with pytest.raises(ExportValidationError, match="strictly increasing"):
        build_lmms_keyframe_annotations(
            [_trace("mlvu", "v", "Q0", frames=(1, 1, 3))],
            benchmark="mlvu",
            questions_file=questions_file,
        )


def test_strict_export_rejects_missing_method_origin_group(tmp_path: Path) -> None:
    questions_file = _write_annotations(
        tmp_path / "mlvu.json",
        [{"video_name": "v.mp4", "question_id": "Q0"}],
    )
    rows = [
        _trace("mlvu", "v", "Q0", method="dwt", origin_id=0),
        _trace("mlvu", "v", "Q0", method="dwt", origin_id=1),
        _trace("mlvu", "v", "Q0", method="swt", origin_id=0),
    ]
    with pytest.raises(ExportValidationError, match="missing method/origin"):
        build_lmms_keyframe_annotations(
            rows,
            benchmark="mlvu",
            questions_file=questions_file,
        )
