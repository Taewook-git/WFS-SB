from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from phase_stable.artifacts import OriginSignalRecord, write_signal_records
from phase_stable.cli import build_parser, main, to_jsonable
from phase_stable.repro import sha256_file
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
        "policy-separation",
        "controlled-shifts",
        "evaluate-prediction-interaction",
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


def test_evaluate_prediction_interaction_command(tmp_path: Path):
    items = (("v0", "q0"), ("v0", "q1"), ("v1", "q0"), ("v1", "q1"))
    base = (("A", "A"), ("A", "A"), ("A", "B"), ("B", "B"))
    regimes = (
        (
            "adaptive",
            {
                "dwt": base,
                "swt": base,
            },
        ),
        (
            "matched",
            {
                "dwt_matched": base,
                "swt_matched": (
                    ("A", "A"),
                    ("A", "A"),
                    ("A", "A"),
                    ("B", "B"),
                ),
            },
        ),
    )
    paths = {}
    row_counts = {}
    for regime, values in regimes:
        rows = []
        for method, predictions in values.items():
            for (video_id, question_id), item_predictions in zip(items, predictions):
                for origin_id, prediction in enumerate(item_predictions):
                    rows.append(
                        {
                            "dataset": "demo",
                            "video_id": video_id,
                            "question_id": question_id,
                            "origin_id": origin_id,
                            "method": method,
                            "prediction": prediction,
                            "gold": "A",
                        }
                    )
        path = tmp_path / f"{regime}.jsonl"
        _write_jsonl(path, list(reversed(rows)))
        paths[regime] = path
        row_counts[regime] = len(rows)

    destination = tmp_path / "interaction_summary.json"
    assert main(
        [
            "evaluate-prediction-interaction",
            str(paths["adaptive"]),
            str(paths["matched"]),
            str(destination),
            "--n-bootstrap",
            "20",
            "--seed",
            "19",
        ]
    ) == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["command"] == "evaluate-prediction-interaction"
    assert payload["adaptive_input_sha256"] == sha256_file(paths["adaptive"])
    assert payload["matched_input_sha256"] == sha256_file(paths["matched"])
    assert payload["num_adaptive_prediction_rows"] == row_counts["adaptive"]
    assert payload["num_matched_prediction_rows"] == row_counts["matched"]
    assert payload["num_paired_items"] == 4
    assert payload["origin_ids"] == [0, 1]
    assert payload["cluster_unit"] == "dataset/video_id"
    assert payload["interaction"]["estimate"] == pytest.approx(
        [0.125, 0.25, -0.25, 0.25]
    )
    assert len(payload["interaction"]["leave_one_video_out"]["rows"]) == 2
    assert payload["interaction"]["n_clusters"] == 2


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


def test_policy_separation_command_writes_summary_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "signals.jsonl"
    output_dir = tmp_path / "policy"
    config = (
        Path(__file__).resolve().parents[1] / "configs" / "phase_stable_icassp.yaml"
    )
    write_signal_records(
        source,
        [
            OriginSignalRecord(
                dataset="demo",
                video_id="v1",
                question_id="q1",
                origin_id=0,
                origin_sec=0.0,
                timestamps_sec=(0.0, 1.0),
                actual_pts_sec=(0.0, 1.0),
                source_frame_indices=(0, 1),
                relevance_scores=(0.1, 0.2),
            )
        ],
    )
    observed: dict[str, object] = {}

    def fake_policy(records, destination, **kwargs):
        observed["record_count"] = len(records)
        observed["destination"] = Path(destination)
        observed.update(kwargs)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        trace_rows = [{"method": "dwt_adaptive"}]
        metric_rows = [{"method": "dwt_adaptive", "value": 1.0}]
        peak_rows = [{"base_method": "dwt", "policy_id": "adaptive"}]
        fidelity_rows: list[dict] = []
        for name, rows in (
            ("traces.jsonl", trace_rows),
            ("item_metrics.jsonl", metric_rows),
            ("peak_rows.jsonl", peak_rows),
            ("fidelity_rows.jsonl", fidelity_rows),
        ):
            _write_jsonl(destination / name, rows)
        (destination / "item_metrics.csv").write_text(
            "method,value\ndwt_adaptive,1.0\n", encoding="utf-8"
        )
        return {
            "trace_rows": trace_rows,
            "item_metric_rows": metric_rows,
            "peak_rows": peak_rows,
            "fidelity_rows": fidelity_rows,
            "stability_summary": {"aggregate": {"dwt_adaptive": {"value": 1.0}}},
            "peak_summary": {"num_rows": np.int64(1)},
            "fidelity_summary": {"num_rows": 0},
        }

    monkeypatch.setattr(
        "phase_stable.cli.run_policy_separation_experiment", fake_policy
    )
    assert (
        main(
            [
                "policy-separation",
                str(source),
                str(output_dir),
                "--config",
                str(config),
                "--b-values",
                "8",
                "15",
                "20",
                "--n-bootstrap",
                "23",
                "--confidence",
                "0.9",
                "--seed",
                "7",
            ]
        )
        == 0
    )

    assert observed["record_count"] == 1
    assert observed["destination"] == output_dir
    assert observed["b_values"] == (8, 15, 20)
    assert observed["n_bootstrap"] == 23
    assert observed["confidence"] == pytest.approx(0.9)
    assert observed["seed"] == 7

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["command"] == "policy-separation"
    assert summary["input_sha256"] == sha256_file(source)
    assert summary["config_sha256"] == sha256_file(config)
    assert summary["b_values"] == [8, 15, 20]
    assert summary["num_signal_records"] == 1
    assert summary["num_traces"] == 1
    assert summary["num_item_metrics"] == 1
    assert summary["num_peak_rows"] == 1
    assert summary["num_fidelity_rows"] == 0
    assert summary["peak_summary"]["num_rows"] == 1
    assert "item_metrics_csv" in summary["artifacts"]

    run_manifest = json.loads(
        (output_dir / "manifest" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["command"] == "policy-separation"
    assert run_manifest["config"]["b_values"] == [8, 15, 20]
    assert [entry["sha256"] for entry in run_manifest["inputs"][:2]] == [
        sha256_file(source),
        sha256_file(config),
    ]
    assert (output_dir / "manifest" / "environment.json").is_file()
