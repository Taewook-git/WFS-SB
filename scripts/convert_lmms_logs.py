#!/usr/bin/env python3
"""Convert verified patched-lmms-eval sample logs to prediction JSONL.

The base lmms-eval commit used by WFS-SB removes the original ``doc`` from its
persisted sample record.  It does retain ``doc_id`` and the benchmark metric
payload produced by the official ``process_results`` function.  This converter
therefore joins ``doc_id`` back to the exact exported keyframe annotation and
uses the already-parsed prediction from that metric payload.  It never attempts
to parse the raw language-model response independently.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class ConversionError(ValueError):
    """Raised when a completed cell cannot be converted without ambiguity."""


BENCHMARK_FIELDS = {
    "videomme": {
        "metric": "videomme_perception_score",
        "prediction": "pred_answer",
        "gold": "answer",
    },
    "mlvu": {
        # The typo is part of the patched task's public metric name.
        "metric": "mlvu_percetion_score",
        "prediction": "pred_answer",
        "gold": "answer",
    },
    "lvb": {
        "metric": "lvb_acc",
        "prediction": "parsed_pred",
        "gold": "answer",
    },
}


def _required_text(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    if key not in mapping:
        raise ConversionError(f"{context} is missing {key!r}")
    value = mapping[key]
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise ConversionError(f"{context}.{key} must be a string/integer identifier")
    result = str(value).strip()
    if not result:
        raise ConversionError(f"{context}.{key} must not be empty")
    return result


def _choice_label(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ConversionError(f"{context} must be an official parsed choice string")
    label = value.strip().upper()
    if not re.fullmatch(r"[A-E]", label):
        raise ConversionError(
            f"{context} is not a normalized A-E choice label: {value!r}"
        )
    return label


def _official_prediction(value: Any, *, context: str) -> str:
    """Preserve the task parser's exact string, including its invalid output."""

    if not isinstance(value, str):
        raise ConversionError(f"{context} must be an official parsed prediction string")
    # VideoMME represents an unparseable answer as ""; MLVU can retain a raw
    # response when its expected closing parenthesis is absent.  Both are the
    # official task parser result and must remain distinct for stability metrics.
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConversionError(f"file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConversionError(f"invalid JSON at {path}: {exc.msg}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConversionError(f"sample log does not exist: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ConversionError(f"expected JSON object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ConversionError(f"sample log contains no rows: {path}")
    return rows


def _parse_marker(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConversionError(f"completion marker does not exist: {path}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        if not line or "=" not in line:
            raise ConversionError(f"malformed completion marker {path}:{line_number}")
        key, value = line.split("=", 1)
        if key in values:
            raise ConversionError(f"duplicate marker field {key!r} in {path}")
        values[key] = value
    required = (
        "benchmark",
        "task",
        "method",
        "origin_id",
        "keyframe_json",
        "results_json",
        "samples_jsonl",
    )
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ConversionError(f"marker {path} missing fields: {', '.join(missing)}")
    return values


def _annotation_identity(
    benchmark: str, annotation: Mapping[str, Any], *, context: str
) -> tuple[str, str, str]:
    if benchmark == "videomme":
        video_id = _required_text(annotation, "video_id", context=context)
        question_id = _required_text(annotation, "question_id", context=context)
        gold = _choice_label(annotation.get("answer"), context=f"{context}.answer")
    elif benchmark == "mlvu":
        video_name = _required_text(annotation, "video_name", context=context)
        video_id = Path(video_name).stem
        question_id = _required_text(annotation, "question_id", context=context)
        gold = _choice_label(annotation.get("answer"), context=f"{context}.answer")
    else:
        video_id = _required_text(annotation, "video_id", context=context)
        question_id = _required_text(annotation, "id", context=context)
        correct_choice = annotation.get("correct_choice")
        if (
            isinstance(correct_choice, bool)
            or not isinstance(correct_choice, int)
            or not 0 <= correct_choice <= 4
        ):
            raise ConversionError(f"{context}.correct_choice must be an integer in [0, 4]")
        gold = chr(ord("A") + correct_choice)
    return video_id, question_id, gold


def _cell_predictions(
    marker_path: Path,
    *,
    benchmark: str,
    method: str,
    origin_id: int,
) -> list[dict[str, Any]]:
    marker = _parse_marker(marker_path)
    if marker["benchmark"] != benchmark:
        raise ConversionError(
            f"marker benchmark mismatch at {marker_path}: {marker['benchmark']!r}"
        )
    if marker["method"] != method or marker["origin_id"] != str(origin_id):
        raise ConversionError(f"marker cell identity mismatch at {marker_path}")

    keyframe_path = Path(marker["keyframe_json"])
    result_path = Path(marker["results_json"])
    sample_path = Path(marker["samples_jsonl"])
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise ConversionError(f"marker result artifact is missing/empty: {result_path}")
    annotations = _read_json(keyframe_path)
    if not isinstance(annotations, list) or not annotations:
        raise ConversionError(f"keyframe JSON must be a non-empty list: {keyframe_path}")
    if not all(isinstance(row, dict) for row in annotations):
        raise ConversionError(f"keyframe JSON rows must be objects: {keyframe_path}")

    specification = BENCHMARK_FIELDS[benchmark]
    output: list[dict[str, Any]] = []
    seen_doc_ids: set[int] = set()
    for sample_number, sample in enumerate(_read_jsonl(sample_path), start=1):
        raw_doc_id = sample.get("doc_id")
        if (
            isinstance(raw_doc_id, bool)
            or not isinstance(raw_doc_id, int)
            or not 0 <= raw_doc_id < len(annotations)
        ):
            raise ConversionError(
                f"{sample_path}:{sample_number} has invalid doc_id {raw_doc_id!r}"
            )
        if raw_doc_id in seen_doc_ids:
            raise ConversionError(f"duplicate doc_id {raw_doc_id} in {sample_path}")
        seen_doc_ids.add(raw_doc_id)
        annotation = annotations[raw_doc_id]
        video_id, question_id, annotation_gold = _annotation_identity(
            benchmark,
            annotation,
            context=f"{keyframe_path}[{raw_doc_id}]",
        )
        metric_payload = sample.get(specification["metric"])
        if not isinstance(metric_payload, Mapping):
            raise ConversionError(
                f"{sample_path}:{sample_number} missing official parser payload "
                f"{specification['metric']!r}"
            )
        prediction = _official_prediction(
            metric_payload.get(specification["prediction"]),
            context=f"{sample_path}:{sample_number}.{specification['prediction']}",
        )
        metric_gold = _choice_label(
            metric_payload.get(specification["gold"]),
            context=f"{sample_path}:{sample_number}.{specification['gold']}",
        )
        if metric_gold != annotation_gold:
            raise ConversionError(
                f"gold mismatch at {sample_path}:{sample_number}: "
                f"metric={metric_gold}, annotation={annotation_gold}"
            )
        if benchmark == "videomme" and str(
            metric_payload.get("question_id")
        ) != question_id:
            raise ConversionError(
                f"VideoMME question_id mismatch at {sample_path}:{sample_number}"
            )
        output.append(
            {
                "dataset": benchmark,
                "video_id": video_id,
                "question_id": question_id,
                "origin_id": origin_id,
                "method": method,
                "prediction": prediction,
                "gold": metric_gold,
            }
        )
    return output


def _comma_list(value: str, *, name: str) -> list[str]:
    result = [item.strip() for item in value.replace(" ", ",").split(",") if item.strip()]
    if not result:
        raise ConversionError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ConversionError(f"{name} must not contain duplicates")
    return result


def convert_grid(
    grid_root: str | Path,
    *,
    benchmark: str,
    methods: Sequence[str],
    origins: Sequence[int],
) -> list[dict[str, Any]]:
    """Convert an exact completed method-by-origin grid."""

    normalized_benchmark = benchmark.lower()
    if normalized_benchmark == "longvideobench":
        normalized_benchmark = "lvb"
    if normalized_benchmark not in BENCHMARK_FIELDS:
        raise ConversionError(f"unsupported benchmark: {benchmark}")
    method_list = [str(method) for method in methods]
    origin_list = list(origins)
    if not method_list or len(method_list) != len(set(method_list)):
        raise ConversionError("methods must be non-empty and unique")
    if (
        not origin_list
        or len(origin_list) != len(set(origin_list))
        or any(isinstance(origin, bool) or not isinstance(origin, int) or origin < 0 for origin in origin_list)
    ):
        raise ConversionError("origins must be unique non-negative integers")

    root = Path(grid_root)
    all_rows: list[dict[str, Any]] = []
    identities_by_cell: list[set[tuple[str, str]]] = []
    gold_by_identity: dict[tuple[str, str], str] = {}
    for method in method_list:
        for origin_id in origin_list:
            marker = root / method / f"origin{origin_id:02d}" / ".complete"
            rows = _cell_predictions(
                marker,
                benchmark=normalized_benchmark,
                method=method,
                origin_id=origin_id,
            )
            identities = {(row["video_id"], row["question_id"]) for row in rows}
            if len(identities) != len(rows):
                raise ConversionError(f"duplicate item identity in cell {method}/origin{origin_id:02d}")
            identities_by_cell.append(identities)
            for row in rows:
                identity = (row["video_id"], row["question_id"])
                previous_gold = gold_by_identity.setdefault(identity, row["gold"])
                if previous_gold != row["gold"]:
                    raise ConversionError(f"gold changes across cells for {identity}")
            all_rows.extend(rows)

    reference = identities_by_cell[0]
    if any(identities != reference for identities in identities_by_cell[1:]):
        raise ConversionError("method/origin cells do not contain the same item identities")
    method_order = {method: index for index, method in enumerate(method_list)}
    all_rows.sort(
        key=lambda row: (
            method_order[row["method"]],
            row["video_id"],
            row["question_id"],
            row["origin_id"],
        )
    )
    return all_rows


def write_predictions(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Atomically write strict seven-field prediction rows."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    required_order = (
        "dataset",
        "video_id",
        "question_id",
        "origin_id",
        "method",
        "prediction",
        "gold",
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if set(row) != set(required_order):
                raise ConversionError("prediction rows must contain exactly seven fields")
            ordered = {key: row[key] for key in required_order}
            handle.write(json.dumps(ordered, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge completed patched-lmms-eval cells into prediction JSONL"
    )
    parser.add_argument("--grid-root", required=True)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("videomme", "mlvu", "lvb", "longvideobench"),
    )
    parser.add_argument("--methods", default="dwt,swt")
    parser.add_argument("--origins", default="0,1,2,3,4")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = _comma_list(args.methods, name="methods")
    raw_origins = _comma_list(args.origins, name="origins")
    try:
        origins = [int(value) for value in raw_origins]
    except ValueError as exc:
        raise ConversionError("origins must contain integers") from exc
    rows = convert_grid(
        args.grid_root,
        benchmark=args.benchmark,
        methods=methods,
        origins=origins,
    )
    output = write_predictions(args.output, rows)
    print(f"Wrote {len(rows)} prediction rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
