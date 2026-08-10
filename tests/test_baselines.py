from pathlib import Path

import numpy as np

from phase_stable.artifacts import OriginSignalRecord
from phase_stable.baselines import run_selection_baselines, topk_indices, uniform_indices


def test_uniform_and_topk_are_exact_and_deterministic() -> None:
    assert uniform_indices(10, 4) == [0, 3, 6, 9]
    assert topk_indices([0.1, 0.9, 0.9, 0.2], 2) == [1, 2]


def test_selection_baselines_write_exportable_rows(tmp_path: Path) -> None:
    records = []
    for origin_id, origin_sec in enumerate((0.1, 0.4, 0.7)):
        timestamps = origin_sec + np.arange(20, dtype=float)
        records.append(
            OriginSignalRecord(
                dataset="videomme",
                video_id="001",
                question_id="001-1",
                origin_id=origin_id,
                origin_sec=origin_sec,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(20)),
                relevance_scores=tuple(np.sin(timestamps) ** 2),
            )
        )
    rows, metrics = run_selection_baselines(records, tmp_path, frame_budget=4)
    assert len(rows) == 6
    assert len(metrics) == 2
    assert all(len(row["selected_source_frame_indices"]) == 4 for row in rows)
    assert (tmp_path / "baseline_traces.jsonl").is_file()
    assert (tmp_path / "baseline_item_metrics.jsonl").is_file()
