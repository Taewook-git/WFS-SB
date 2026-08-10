from pathlib import Path

import numpy as np
import pytest

from phase_stable.artifacts import (
    OriginSignalRecord,
    load_trace_arrays,
    read_signal_records,
    save_trace_npz,
    write_signal_records,
)
from phase_stable.pipeline import PhaseStableWFS
from phase_stable.transforms import TransformConfig, build_transform


def _record() -> OriginSignalRecord:
    timestamps = tuple(float(index) for index in range(128))
    scores = tuple(np.sin(np.linspace(0, 8 * np.pi, 128)) * 0.2 + 0.5)
    return OriginSignalRecord(
        dataset="demo",
        video_id="video/a",
        question_id="question:1",
        origin_id=0,
        origin_sec=0.0,
        timestamps_sec=timestamps,
        actual_pts_sec=timestamps,
        source_frame_indices=tuple(range(128)),
        relevance_scores=scores,
        pixel_hashes=tuple("a" * 64 for _ in timestamps),
    )


def test_signal_record_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "signals.jsonl"
    write_signal_records(path, [_record()])
    loaded = read_signal_records(path)
    assert loaded == [_record()]


def test_signal_record_rejects_misalignment() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        OriginSignalRecord(
            dataset="demo",
            video_id="v",
            question_id="q",
            origin_id=0,
            origin_sec=0.0,
            timestamps_sec=(0.0, 1.0),
            actual_pts_sec=(0.0,),
            source_frame_indices=(0, 1),
            relevance_scores=(0.1, 0.2),
        )


def test_trace_npz_roundtrip(tmp_path: Path) -> None:
    record = _record()
    transform = build_transform(TransformConfig(method="swt", level=3))
    trace = PhaseStableWFS(transform).run(record.relevance_scores, 8, 5)
    path, row = save_trace_npz(tmp_path, record, "swt", trace)
    arrays = load_trace_arrays(row)

    assert path.is_file()
    assert arrays["representation"].shape == (3, 128)
    assert len(row["selected_source_frame_indices"]) == 8
