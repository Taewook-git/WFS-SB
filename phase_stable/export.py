"""Export phase-stability traces as lmms-eval keyframe annotations.

The phase-stability experiment writes one generic trace row per
``(dataset, video, question, origin, method)``.  ``lmms-eval`` instead expects
the benchmark's original annotation rows with a ``keyframe_indices`` field.
This module performs that join without changing the official WFS-SB benchmark
adapters or the original annotation files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from wfs.benchmarks import BenchmarkAdapter, BenchmarkRecord, create_adapter

from .artifacts import iter_jsonl


ExportKey = Tuple[str, int]


class ExportValidationError(ValueError):
    """Raised when traces cannot be joined to annotations unambiguously."""


@dataclass(frozen=True)
class _AnnotationBinding:
    record: BenchmarkRecord
    video_id: str
    question_id: str

    @property
    def item_key(self) -> tuple[str, str]:
        return self.video_id, self.question_id


def _required_text(name: str, value: Any) -> str:
    if not isinstance(value, (str, Integral)) or isinstance(value, bool):
        raise ExportValidationError(f"{name} must be a string or integer identifier")
    result = str(value).strip()
    if not result:
        raise ExportValidationError(f"{name} must be non-empty")
    return result


def _required_mapping_value(row: Mapping[str, Any], name: str, row_number: int) -> Any:
    if name not in row:
        raise ExportValidationError(f"trace row {row_number} is missing {name!r}")
    return row[name]


def _normalize_benchmark(value: Any) -> str:
    name = _required_text("benchmark", value).lower()
    if name == "longvideobench":
        return "lvb"
    if name not in {"videomme", "lvb", "mlvu"}:
        raise ExportValidationError(f"unsupported benchmark: {value}")
    return name


def _canonical_mlvu_video_id(value: str) -> str:
    """Accept either an MLVU video filename or its extension-free ID."""

    path = Path(value)
    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        return path.stem
    return value


def _annotation_identity(
    adapter: BenchmarkAdapter,
    record: BenchmarkRecord,
) -> tuple[str, str]:
    """Return the canonical experiment identity for an adapter record."""

    raw = record.raw
    if adapter.name == "videomme":
        video_id = _required_text("annotation video_id", raw.get("video_id"))
        question_id = _required_text("annotation question_id", raw.get("question_id"))
    elif adapter.name == "lvb":
        video_id = _required_text("annotation video_id", raw.get("video_id"))
        question_id = _required_text("annotation id", raw.get("id"))
    elif adapter.name == "mlvu":
        video_name = _required_text("annotation video_name", raw.get("video_name"))
        video_id = _canonical_mlvu_video_id(video_name)
        question_id = _required_text("annotation question_id", raw.get("question_id"))
    else:  # pragma: no cover - create_adapter currently prevents this branch.
        raise ExportValidationError(f"unsupported adapter: {adapter.name}")
    return video_id, question_id


def _load_annotation_bindings(
    adapter: BenchmarkAdapter,
    questions_file: str | Path,
    dataset_root: str | Path,
) -> tuple[list[_AnnotationBinding], dict[tuple[str, str], _AnnotationBinding]]:
    raw_items = adapter.load_raw(Path(questions_file))
    bindings: list[_AnnotationBinding] = []
    by_key: dict[tuple[str, str], _AnnotationBinding] = {}
    root = Path(dataset_root)

    for index, item in enumerate(raw_items):
        record = adapter.build_record(index=index, item=item, dataset_root=root)
        video_id, question_id = _annotation_identity(adapter, record)
        binding = _AnnotationBinding(record, video_id, question_id)
        if binding.item_key in by_key:
            raise ExportValidationError(
                "duplicate annotation identity "
                f"video_id={video_id!r}, question_id={question_id!r}"
            )
        bindings.append(binding)
        by_key[binding.item_key] = binding

    if not bindings:
        raise ExportValidationError("annotation file contains no rows")
    return bindings, by_key


def _normalize_method_filter(methods: Optional[Sequence[str]]) -> Optional[set[str]]:
    if methods is None:
        return None
    normalized = {_required_text("method", method) for method in methods}
    if not normalized:
        raise ExportValidationError("methods must not be empty")
    return normalized


def _normalize_origin_filter(origin_ids: Optional[Sequence[int]]) -> Optional[set[int]]:
    if origin_ids is None:
        return None
    normalized: set[int] = set()
    for value in origin_ids:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ExportValidationError("origin_ids must contain non-negative integers")
        normalized.add(int(value))
    if not normalized:
        raise ExportValidationError("origin_ids must not be empty")
    return normalized


def _selected_frames(row: Mapping[str, Any], row_number: int) -> list[int]:
    values = _required_mapping_value(
        row, "selected_source_frame_indices", row_number
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ExportValidationError(
            f"trace row {row_number} selected_source_frame_indices must be a sequence"
        )

    frames: list[int] = []
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ExportValidationError(
                f"trace row {row_number} frame {position} must be an integer"
            )
        frame = int(value)
        if frame < 0:
            raise ExportValidationError(
                f"trace row {row_number} frame {position} must be non-negative"
            )
        frames.append(frame)

    if not frames:
        raise ExportValidationError(
            f"trace row {row_number} selected_source_frame_indices must not be empty"
        )
    if any(current <= previous for previous, current in zip(frames, frames[1:])):
        raise ExportValidationError(
            f"trace row {row_number} selected_source_frame_indices must be "
            "strictly increasing and duplicate-free"
        )
    return frames


def build_lmms_keyframe_annotations(
    trace_rows: Iterable[Mapping[str, Any]],
    *,
    benchmark: str,
    questions_file: str | Path,
    dataset_root: str | Path = ".",
    methods: Optional[Sequence[str]] = None,
    origin_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
    allow_annotation_subset: bool = False,
    expected_budget: Optional[int] = None,
) -> Dict[ExportKey, list[Dict[str, Any]]]:
    """Join trace rows to official annotations, grouped by method and origin.

    Args:
        trace_rows: Generic rows produced by ``save_trace_npz``/``traces.jsonl``.
        benchmark: ``videomme``, ``lvb``/``longvideobench``, or ``mlvu``.
        questions_file: Official benchmark annotation JSON.
        dataset_root: Root passed to the existing benchmark adapter.
        methods: Optional method subset. By default every observed method is used.
        origin_ids: Optional origin subset. By default every observed origin is used.
        strict: Require a rectangular method-by-origin grid and exactly one trace
            for every annotation row in every output file.
        allow_annotation_subset: With ``strict=True``, permit a cohort containing
            only a subset of the official annotations while still requiring every
            method/origin group to contain exactly the same cohort.
        expected_budget: Optional required number of selected frames per row.

    Returns:
        A dictionary keyed by ``(method, origin_id)``. Each value is a complete
        benchmark-format annotation list with ``keyframe_indices`` injected.

    Raises:
        ExportValidationError: On missing, duplicate, malformed, or unmatched
            trace/annotation records.
    """

    normalized_benchmark = _normalize_benchmark(benchmark)
    adapter = create_adapter(normalized_benchmark)
    bindings, annotation_by_key = _load_annotation_bindings(
        adapter, questions_file, dataset_root
    )
    method_filter = _normalize_method_filter(methods)
    origin_filter = _normalize_origin_filter(origin_ids)

    if expected_budget is not None:
        if (
            isinstance(expected_budget, bool)
            or not isinstance(expected_budget, Integral)
            or int(expected_budget) <= 0
        ):
            raise ExportValidationError("expected_budget must be a positive integer")
        expected_budget = int(expected_budget)

    grouped: dict[ExportKey, dict[tuple[str, str], list[int]]] = defaultdict(dict)
    observed_methods: set[str] = set()
    observed_origins: set[int] = set()
    observed_budget: Optional[int] = expected_budget

    for row_number, row in enumerate(trace_rows, start=1):
        if not isinstance(row, Mapping):
            raise ExportValidationError(f"trace row {row_number} must be a mapping")

        method = _required_text(
            "method", _required_mapping_value(row, "method", row_number)
        )
        raw_origin = _required_mapping_value(row, "origin_id", row_number)
        if isinstance(raw_origin, bool) or not isinstance(raw_origin, Integral):
            raise ExportValidationError(
                f"trace row {row_number} origin_id must be an integer"
            )
        origin_id = int(raw_origin)
        if origin_id < 0:
            raise ExportValidationError(
                f"trace row {row_number} origin_id must be non-negative"
            )

        if method_filter is not None and method not in method_filter:
            continue
        if origin_filter is not None and origin_id not in origin_filter:
            continue

        trace_dataset = _normalize_benchmark(
            _required_mapping_value(row, "dataset", row_number)
        )
        if trace_dataset != normalized_benchmark:
            raise ExportValidationError(
                f"trace row {row_number} dataset={trace_dataset!r} does not match "
                f"benchmark={normalized_benchmark!r}"
            )
        video_id = _required_text(
            "video_id", _required_mapping_value(row, "video_id", row_number)
        )
        if adapter.name == "mlvu":
            video_id = _canonical_mlvu_video_id(video_id)
        question_id = _required_text(
            "question_id", _required_mapping_value(row, "question_id", row_number)
        )
        item_key = (video_id, question_id)
        if item_key not in annotation_by_key:
            raise ExportValidationError(
                f"trace row {row_number} has no matching annotation: "
                f"video_id={video_id!r}, question_id={question_id!r}"
            )

        frames = _selected_frames(row, row_number)
        if observed_budget is None:
            observed_budget = len(frames)
        if len(frames) != observed_budget:
            raise ExportValidationError(
                f"trace row {row_number} has frame budget {len(frames)}, "
                f"expected {observed_budget}"
            )

        group_key = (method, origin_id)
        if item_key in grouped[group_key]:
            raise ExportValidationError(
                "duplicate trace for "
                f"method={method!r}, origin_id={origin_id}, "
                f"video_id={video_id!r}, question_id={question_id!r}"
            )
        grouped[group_key][item_key] = frames
        observed_methods.add(method)
        observed_origins.add(origin_id)

    if not grouped:
        raise ExportValidationError("no trace rows matched the requested method/origin filters")

    selected_methods = method_filter if method_filter is not None else observed_methods
    selected_origins = origin_filter if origin_filter is not None else observed_origins
    expected_groups = {
        (method, origin_id)
        for method in selected_methods
        for origin_id in selected_origins
    }
    missing_groups = expected_groups.difference(grouped)
    if strict and missing_groups:
        formatted = ", ".join(
            f"{method}/origin={origin_id}"
            for method, origin_id in sorted(missing_groups)
        )
        raise ExportValidationError(f"missing method/origin trace groups: {formatted}")

    annotation_keys = set(annotation_by_key)
    strict_cohort: Optional[set[tuple[str, str]]] = None
    if strict and allow_annotation_subset:
        first_group = min(grouped)
        strict_cohort = set(grouped[first_group])
        for group_key, item_frames in grouped.items():
            item_keys = set(item_frames)
            if item_keys != strict_cohort:
                missing = strict_cohort.difference(item_keys)
                extra = item_keys.difference(strict_cohort)
                method, origin_id = group_key
                raise ExportValidationError(
                    "inconsistent annotation subset for "
                    f"method={method!r}, origin_id={origin_id}: "
                    f"missing={len(missing)}, extra={len(extra)}"
                )

    exports: Dict[ExportKey, list[Dict[str, Any]]] = {}
    for group_key in sorted(grouped):
        item_frames = grouped[group_key]
        if strict and not allow_annotation_subset:
            missing_items = annotation_keys.difference(item_frames)
            if missing_items:
                preview = ", ".join(
                    f"{video_id}/{question_id}"
                    for video_id, question_id in sorted(missing_items)[:5]
                )
                suffix = "" if len(missing_items) <= 5 else ", ..."
                method, origin_id = group_key
                raise ExportValidationError(
                    f"missing {len(missing_items)} trace row(s) for "
                    f"method={method!r}, origin_id={origin_id}: {preview}{suffix}"
                )

        output_rows: list[Dict[str, Any]] = []
        for binding in bindings:
            frames = item_frames.get(binding.item_key)
            if frames is None:
                continue
            output_rows.append(adapter.to_output_item(binding.record, frames))
        exports[group_key] = output_rows

    return exports


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not token:
        raise ExportValidationError(f"method cannot form a safe filename: {value!r}")
    return token


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def export_lmms_keyframe_jsons(
    trace_rows: Iterable[Mapping[str, Any]],
    *,
    benchmark: str,
    questions_file: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path = ".",
    methods: Optional[Sequence[str]] = None,
    origin_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
    allow_annotation_subset: bool = False,
    expected_budget: Optional[int] = None,
    filename_prefix: Optional[str] = None,
) -> Dict[ExportKey, Path]:
    """Write one lmms-eval annotation JSON for every method/origin pair."""

    normalized_benchmark = _normalize_benchmark(benchmark)
    exports = build_lmms_keyframe_annotations(
        trace_rows,
        benchmark=normalized_benchmark,
        questions_file=questions_file,
        dataset_root=dataset_root,
        methods=methods,
        origin_ids=origin_ids,
        strict=strict,
        allow_annotation_subset=allow_annotation_subset,
        expected_budget=expected_budget,
    )
    prefix = (
        _safe_filename_token(filename_prefix)
        if filename_prefix is not None
        else normalized_benchmark
    )
    destination = Path(output_dir)
    paths: Dict[ExportKey, Path] = {}
    used_paths: set[Path] = set()

    for (method, origin_id), rows in exports.items():
        method_token = _safe_filename_token(method)
        path = destination / f"{prefix}_{method_token}_origin{origin_id:02d}.json"
        if path in used_paths:
            raise ExportValidationError(
                f"method names collide after filename sanitization at {path.name}"
            )
        _write_json(path, rows)
        used_paths.add(path)
        paths[(method, origin_id)] = path
    return paths


def export_trace_jsonl(
    trace_jsonl: str | Path,
    *,
    benchmark: str,
    questions_file: str | Path,
    output_dir: str | Path,
    dataset_root: str | Path = ".",
    methods: Optional[Sequence[str]] = None,
    origin_ids: Optional[Sequence[int]] = None,
    strict: bool = True,
    allow_annotation_subset: bool = False,
    expected_budget: Optional[int] = None,
    filename_prefix: Optional[str] = None,
) -> Dict[ExportKey, Path]:
    """Read generic trace JSONL and export method/origin keyframe JSON files."""

    return export_lmms_keyframe_jsons(
        iter_jsonl(trace_jsonl),
        benchmark=benchmark,
        questions_file=questions_file,
        output_dir=output_dir,
        dataset_root=dataset_root,
        methods=methods,
        origin_ids=origin_ids,
        strict=strict,
        allow_annotation_subset=allow_annotation_subset,
        expected_budget=expected_budget,
        filename_prefix=filename_prefix,
    )


# A concise public alias for callers that already hold trace rows in memory.
export_keyframe_jsons = export_lmms_keyframe_jsons


def _comma_separated_text(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    result = [token.strip() for token in value.split(",") if token.strip()]
    return result or None


def _comma_separated_ints(value: Optional[str]) -> Optional[list[int]]:
    text_values = _comma_separated_text(value)
    if text_values is None:
        return None
    try:
        return [int(token) for token in text_values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("origin_ids must be comma-separated integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export phase-stability traces as lmms-eval keyframe JSONs"
    )
    parser.add_argument("--trace_jsonl", required=True)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("videomme", "lvb", "longvideobench", "mlvu"),
    )
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--dataset_root", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--methods", default=None, help="Comma-separated method names")
    parser.add_argument("--origin_ids", default=None, help="Comma-separated origin IDs")
    parser.add_argument("--expected_budget", type=int, default=None)
    parser.add_argument("--filename_prefix", default=None)
    completeness = parser.add_mutually_exclusive_group()
    completeness.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow output groups that cover only a subset of annotation rows",
    )
    completeness.add_argument(
        "--allow_annotation_subset",
        action="store_true",
        help=(
            "Allow one strict annotation cohort subset while retaining the complete "
            "method/origin grid"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = export_trace_jsonl(
        args.trace_jsonl,
        benchmark=args.benchmark,
        questions_file=args.questions_file,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        methods=_comma_separated_text(args.methods),
        origin_ids=_comma_separated_ints(args.origin_ids),
        strict=not args.allow_partial,
        allow_annotation_subset=args.allow_annotation_subset,
        expected_budget=args.expected_budget,
        filename_prefix=args.filename_prefix,
    )
    summary = {
        f"{method}/origin={origin_id}": str(path)
        for (method, origin_id), path in paths.items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


__all__ = [
    "ExportKey",
    "ExportValidationError",
    "build_lmms_keyframe_annotations",
    "build_parser",
    "export_keyframe_jsons",
    "export_lmms_keyframe_jsons",
    "export_trace_jsonl",
    "main",
]


if __name__ == "__main__":
    main()
