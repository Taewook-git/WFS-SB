from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_lmms_logs.py"
SPEC = importlib.util.spec_from_file_location("convert_lmms_logs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_cell(
    root: Path,
    *,
    benchmark: str,
    method: str,
    origin: int,
    annotations: list[dict],
    samples: list[dict],
) -> None:
    cell = root / method / f"origin{origin:02d}"
    cell.mkdir(parents=True)
    keyframes = cell / "keyframes.json"
    results = cell / "result.json"
    sample_log = cell / "samples.jsonl"
    keyframes.write_text(json.dumps(annotations), encoding="utf-8")
    results.write_text("{}\n", encoding="utf-8")
    sample_log.write_text(
        "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
    )
    (cell / ".complete").write_text(
        "\n".join(
            (
                f"benchmark={benchmark}",
                "task=unused-in-converter",
                f"method={method}",
                f"origin_id={origin}",
                f"keyframe_json={keyframes}",
                f"results_json={results}",
                f"samples_jsonl={sample_log}",
                "completed_utc=2026-08-10T00:00:00Z",
            )
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("benchmark", "annotation", "metric", "expected_identity"),
    [
        (
            "videomme",
            {"video_id": "001", "question_id": "001-1", "answer": "C"},
            {
                "videomme_perception_score": {
                    "question_id": "001-1",
                    "pred_answer": "B",
                    "answer": "C",
                }
            },
            ("001", "001-1"),
        ),
        (
            "mlvu",
            {"video_name": "clip.one.mp4", "question_id": "Q7", "answer": "A"},
            {
                "mlvu_percetion_score": {
                    "question_id": "question text",
                    "pred_answer": "D",
                    "answer": "A",
                }
            },
            ("clip.one", "Q7"),
        ),
        (
            "lvb",
            {
                "video_id": "video-a",
                "id": "video-a_3",
                "correct_choice": 4,
            },
            {"lvb_acc": {"parsed_pred": "E", "answer": "E"}},
            ("video-a", "video-a_3"),
        ),
    ],
)
def test_converter_uses_official_metric_payload_and_annotation_identity(
    tmp_path: Path,
    benchmark: str,
    annotation: dict,
    metric: dict,
    expected_identity: tuple[str, str],
):
    root = tmp_path / benchmark
    for method in ("dwt", "swt"):
        for origin in (0, 1):
            _write_cell(
                root,
                benchmark=benchmark,
                method=method,
                origin=origin,
                annotations=[annotation],
                samples=[{"doc_id": 0, "filtered_resps": ["raw"], **metric}],
            )
    rows = MODULE.convert_grid(
        root,
        benchmark=benchmark,
        methods=("dwt", "swt"),
        origins=(0, 1),
    )
    assert len(rows) == 4
    assert {(row["video_id"], row["question_id"]) for row in rows} == {
        expected_identity
    }
    assert all(len(row) == 7 for row in rows)
    expected_prediction = next(iter(metric.values()))[
        "parsed_pred" if benchmark == "lvb" else "pred_answer"
    ]
    assert {row["prediction"] for row in rows} == {expected_prediction}


@pytest.mark.parametrize("official_prediction", ["", "The answer was unclear"])
def test_converter_preserves_official_invalid_prediction_without_reparsing(
    tmp_path: Path, official_prediction: str
):
    root = tmp_path / "videomme"
    _write_cell(
        root,
        benchmark="videomme",
        method="dwt",
        origin=0,
        annotations=[{"video_id": "v", "question_id": "q", "answer": "A"}],
        samples=[
            {
                "doc_id": 0,
                "filtered_resps": ["The answer is B"],
                "videomme_perception_score": {
                    "question_id": "q",
                    "pred_answer": official_prediction,
                    "answer": "A",
                },
            }
        ],
    )
    rows = MODULE.convert_grid(
        root, benchmark="videomme", methods=("dwt",), origins=(0,)
    )
    assert rows[0]["prediction"] == official_prediction


def test_prediction_writer_emits_exact_ordered_seven_fields(tmp_path: Path):
    output = tmp_path / "nested" / "predictions.jsonl"
    row = {
        "method": "dwt",
        "gold": "A",
        "prediction": "B",
        "origin_id": 0,
        "question_id": "q",
        "video_id": "v",
        "dataset": "videomme",
    }
    MODULE.write_predictions(output, [row])
    raw = output.read_text(encoding="utf-8").strip()
    assert list(json.loads(raw)) == [
        "dataset",
        "video_id",
        "question_id",
        "origin_id",
        "method",
        "prediction",
        "gold",
    ]
