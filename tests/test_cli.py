from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phase_stable.artifacts import OriginSignalRecord, write_signal_records
from phase_stable.cli import build_parser, main, to_jsonable
from phase_stable.sampling import read_manifests_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_recursive_json_conversion_supports_numpy_path_and_dataclass(tmp_path: Path):
    record = OriginSignalRecord(
        dataset="demo",
        video_id="v",
        question_id="q",
        origin_id=0,
        origin_sec=0.0,
        timestamps_sec=(0.0, 1.0),
        actual_pts_sec=(0.0, 1.0),
        source_frame_indices=(0, 1),
        relevance_scores=(0.1, 0.2),
    )
    converted = to_jsonable(
        {
            "array": np.array([1, 2]),
            "scalar": np.float32(0.5),
            "path": tmp_path,
            "nested": (record,),
        }
    )
    assert converted["array"] == [1, 2]
    assert converted["scalar"] == pytest.approx(0.5)
    assert converted["path"] == str(tmp_path)
    assert converted["nested"][0]["video_id"] == "v"
    with pytest.raises(ValueError, match="NaN"):
        to_jsonable(np.float64(np.nan))


def test_parser_and_python_module_help_smoke():
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "analyze-signals",
        "matched-selection",
        "controlled-shifts",
        "evaluate-predictions",
        "make-manifests",
    ):
        assert command in help_text

    completed = subprocess.run(
        [sys.executable, "-m", "phase_stable", "--help"],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "analyze-signals" in completed.stdout


def test_make_manifests_command(tmp_path: Path):
    source = tmp_path / "videos.jsonl"
    destination = tmp_path / "manifests.jsonl"
    _write_jsonl(
        source,
        [
            {"video_id": "v1", "duration_sec": 5.2},
            {"video_id": "v2", "duration_sec": 3.7},
        ],
    )
    assert main(
        [
            "make-manifests",
            str(source),
            str(destination),
            "--seed",
            "17",
            "--num-origins",
            "3",
        ]
    ) == 0
    manifests = read_manifests_jsonl(destination)
    assert [manifest.video_id for manifest in manifests] == ["v1", "v2"]
    assert all(manifest.num_origins == 3 for manifest in manifests)


def test_controlled_shifts_command(tmp_path: Path):
    signal_path = tmp_path / "signal.npy"
    output_path = tmp_path / "controlled.json"
    signal = np.zeros(128, dtype=float)
    signal[30:80] = 1.0
    np.save(signal_path, signal)
    assert main(
        [
            "controlled-shifts",
            str(signal_path),
            str(output_path),
            "--level",
            "3",
            "--shifts",
            "0",
            "1",
            "2",
        ]
    ) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(payload["metrics"]) == {"dwt", "swt"}
    assert payload["shifts"] == [0, 1, 2]


def test_evaluate_predictions_command(tmp_path: Path):
    rows = []
    for method in ("dwt", "swt"):
        for video_index in range(2):
            for origin_id in range(2):
                rows.append(
                    {
                        "dataset": "demo",
                        "video_id": f"v{video_index}",
                        "question_id": f"q{video_index}",
                        "origin_id": origin_id,
                        "method": method,
                        "prediction": (
                            "B"
                            if method == "dwt" and video_index == 0 and origin_id == 1
                            else "A"
                        ),
                        "gold": "A",
                    }
                )
    source = tmp_path / "predictions.jsonl"
    destination = tmp_path / "prediction_summary.json"
    _write_jsonl(source, rows)
    assert main(
        [
            "evaluate-predictions",
            str(source),
            str(destination),
            "--n-bootstrap",
            "20",
            "--seed",
            "9",
        ]
    ) == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["methods"]["swt"]["robust_accuracy"] == 1.0
    assert payload["comparison"]["effect_order"][1] == "delta_robust_accuracy"
    assert isinstance(payload["comparison"]["estimate"], list)


def test_analyze_signals_command_writes_csv_jsonl_and_summary(tmp_path: Path):
    records = []
    for origin_id, origin_sec in enumerate((0.1, 0.6)):
        timestamps = origin_sec + np.arange(64, dtype=float)
        scores = 0.5 + 0.25 * np.sin(timestamps / 5.0)
        records.append(
            OriginSignalRecord(
                dataset="demo",
                video_id="v1",
                question_id="q1",
                origin_id=origin_id,
                origin_sec=origin_sec,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(64)),
                relevance_scores=tuple(scores),
            )
        )
    source = tmp_path / "signals.jsonl"
    output_dir = tmp_path / "analysis"
    write_signal_records(source, records)
    assert main(
        [
            "analyze-signals",
            str(source),
            str(output_dir),
            "--level",
            "2",
            "--frame-budget",
            "4",
            "--min-distance-absolute",
            "2",
            "--n-bootstrap",
            "10",
        ]
    ) == 0
    assert (output_dir / "item_metrics.jsonl").is_file()
    assert (output_dir / "item_metrics.csv").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_item_metrics"] == 2
    assert set(summary["aggregate"]) == {"dwt", "swt"}
    assert "representation_consistency_mean" in summary["bootstrap"]["metrics"]
    assert summary["bootstrap"]["num_paired_items"] == 1


def test_matched_selection_command_writes_isolated_summary(tmp_path: Path):
    records = []
    for origin_id, origin_sec in enumerate((0.1, 0.6)):
        timestamps = origin_sec + np.arange(64, dtype=float)
        scores = 0.5 + 0.25 * np.sin(timestamps / 5.0)
        records.append(
            OriginSignalRecord(
                dataset="demo",
                video_id="v1",
                question_id="q1",
                origin_id=origin_id,
                origin_sec=origin_sec,
                timestamps_sec=tuple(timestamps),
                actual_pts_sec=tuple(timestamps),
                source_frame_indices=tuple(range(64)),
                relevance_scores=tuple(scores),
            )
        )
    source = tmp_path / "signals.jsonl"
    output_dir = tmp_path / "matched"
    config = Path(__file__).resolve().parents[1] / "configs" / "phase_stable_icassp.yaml"
    write_signal_records(source, records)
    assert main(
        [
            "matched-selection",
            str(source),
            str(output_dir),
            "--config",
            str(config),
            "--count",
            "4",
            "--n-bootstrap",
            "10",
        ]
    ) == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["boundary_policy"] == {"source": "fixed", "count": 4}
    assert set(summary["aggregate"]) == {"dwt_matched", "swt_matched"}
    assert summary["bootstrap"]["baseline_method"] == "dwt_matched"
    traces = [
        json.loads(line)
        for line in (output_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(traces) == 4
    assert all(len(row["peaks"]) == 4 for row in traces)
