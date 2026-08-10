"""Command-line entry points for the phase-stability experiment pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import (
    ExperimentConfig,
    aggregate_item_metrics,
    compute_matched_cardinality_metrics,
    controlled_shift_metrics,
    evaluate_prediction_rows,
    paired_metric_bootstrap,
    run_real_origin_experiment,
)
from .artifacts import iter_jsonl, read_signal_records
from .benchmarks import (
    build_benchmark_manifests,
    load_benchmark_videos,
    preprocess_benchmark_to_jsonl,
)
from .baselines import run_selection_baselines
from .config import load_phase_stable_config
from .export import export_trace_jsonl
from .repro import write_reproducibility_manifests
from .pipeline import SelectionConfig
from .sampling import (
    build_sampling_manifest,
    read_manifests_jsonl,
    write_manifests_jsonl,
)

DEFAULT_BOOTSTRAP_METRICS = (
    "representation_consistency_mean",
    "saliency_consistency_mean",
    "saliency_l1_mean",
    "boundary_f1_mean",
    "segment_ari_mean",
    "selected_f1_mean",
)

FEATURE_MODEL_DEFAULTS = {
    "blip2": "Salesforce/blip2-itm-vit-g",
    "blip1": "Salesforce/blip-itm-base-coco",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
}


def to_jsonable(value: Any) -> Any:
    """Recursively convert experiment objects to strict JSON-compatible data.

    NumPy arrays/scalars, dataclasses, ``Path`` objects, mappings, and nested
    sequences are supported.  Non-finite floating-point values are rejected so
    output JSON never relies on JavaScript's non-standard ``NaN`` literals.
    """

    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return to_jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cannot serialize NaN or infinity to strict JSON")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            to_jsonable(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    to_jsonable(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
    temporary.replace(destination)
    return destination


def _csv_cell(value: Any) -> Any:
    converted = to_jsonable(value)
    if converted is None:
        return ""
    if isinstance(converted, (dict, list)):
        return json.dumps(
            converted,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return converted


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    if not rows:
        raise ValueError("cannot write an item-metrics CSV with no rows")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_cell(row.get(name)) for name in fieldnames})
    temporary.replace(destination)
    return destination


def _experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        methods=tuple(args.methods),
        wavelet=args.wavelet,
        level=args.level,
        drift_level=args.drift_level,
        frame_budget=args.frame_budget,
        min_distance_ratio=args.min_distance_ratio,
        min_distance_absolute=args.min_distance_absolute,
        shared_padding=args.shared_padding,
        padding_mode=args.padding_mode,
        dwt_mode=args.dwt_mode,
        swt_norm=args.swt_norm,
        cycle_shifts=tuple(args.cycle_shifts),
        cycle_aggregation=args.cycle_aggregation,
        gaussian_sigma=args.gaussian_sigma,
        boundary_tolerance_sec=args.boundary_tolerance_sec,
        report_boundary_tolerances_sec=tuple(args.report_boundary_tolerances_sec),
        selected_tolerance_sec=args.selected_tolerance_sec,
        edge_margin_sec=args.edge_margin_sec,
    )


def _run_analyze_signals(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    loaded_config = (
        load_phase_stable_config(args.config) if args.config is not None else None
    )
    config = (
        loaded_config.experiment
        if loaded_config is not None
        else _experiment_config(args)
    )
    selection_config = (
        loaded_config.selection if loaded_config is not None else SelectionConfig()
    )
    records = read_signal_records(input_path)
    traces, metric_rows = run_real_origin_experiment(
        records,
        output_dir,
        config=config,
        selection_config=selection_config,
    )

    jsonl_path = _write_jsonl(output_dir / "item_metrics.jsonl", metric_rows)
    csv_path = _write_csv(output_dir / "item_metrics.csv", metric_rows)
    aggregate = aggregate_item_metrics(metric_rows)
    bootstrap = paired_metric_bootstrap(
        metric_rows,
        baseline_method=args.baseline_method,
        treatment_method=args.treatment_method,
        metric_names=tuple(args.bootstrap_metrics),
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    summary_path = output_dir / "summary.json"
    run_manifest_path, environment_path = write_reproducibility_manifests(
        output_dir,
        command="analyze-signals",
        config={
            "experiment": asdict(config),
            "selection": asdict(selection_config),
        },
        input_paths=(input_path,),
        extra={
            "num_signal_records": len(records),
            "num_traces": len(traces),
            "num_item_metrics": len(metric_rows),
        },
    )
    summary = {
        "command": "analyze-signals",
        "input": input_path,
        "output_dir": output_dir,
        "config": config,
        "selection_config": selection_config,
        "num_signal_records": len(records),
        "num_traces": len(traces),
        "num_item_metrics": len(metric_rows),
        "aggregate": aggregate,
        "bootstrap": bootstrap,
        "artifacts": {
            "traces_jsonl": output_dir / "traces.jsonl",
            "item_metrics_jsonl": jsonl_path,
            "item_metrics_csv": csv_path,
            "summary_json": summary_path,
            "run_manifest_json": run_manifest_path,
            "environment_json": environment_path,
        },
    }
    _write_json(summary_path, summary)
    print(f"Wrote {len(metric_rows)} item metric rows and summary to {output_dir}")
    return 0


def _run_controlled_shifts(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    signal = np.load(input_path, allow_pickle=False)
    result = controlled_shift_metrics(
        signal,
        methods=tuple(args.methods),
        shifts=tuple(args.shifts),
        wavelet=args.wavelet,
        level=args.level,
        drift_level=args.drift_level,
        shared_padding=args.shared_padding,
        edge_samples=args.edge_samples,
    )
    output_path = _write_json(
        args.output,
        {
            "command": "controlled-shifts",
            "input": input_path,
            "methods": tuple(args.methods),
            "shifts": tuple(args.shifts),
            "wavelet": args.wavelet,
            "level": args.level,
            "shared_padding": args.shared_padding,
            "edge_samples": args.edge_samples,
            "metrics": result,
        },
    )
    print(f"Wrote controlled-shift metrics to {output_path}")
    return 0


def _run_evaluate_predictions(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    rows = list(iter_jsonl(input_path))
    if not rows:
        raise ValueError("prediction JSONL must contain at least one row")
    evaluation = evaluate_prediction_rows(
        rows,
        baseline_method=args.baseline_method,
        treatment_method=args.treatment_method,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
    )
    output_path = _write_json(
        args.output,
        {
            "command": "evaluate-predictions",
            "input": input_path,
            "num_prediction_rows": len(rows),
            **evaluation,
        },
    )
    print(f"Wrote prediction evaluation to {output_path}")
    return 0


def _run_make_manifests(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    rows = list(iter_jsonl(input_path))
    if not rows:
        raise ValueError("video JSONL must contain at least one row")
    manifests = []
    seen_video_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        missing = [name for name in ("video_id", "duration_sec") if name not in row]
        if missing:
            raise ValueError(
                f"video row {line_number} missing fields: {', '.join(missing)}"
            )
        video_id = str(row["video_id"])
        if video_id in seen_video_ids:
            raise ValueError(f"duplicate video_id at row {line_number}: {video_id!r}")
        seen_video_ids.add(video_id)
        manifests.append(
            build_sampling_manifest(
                video_id,
                row["duration_sec"],
                master_seed=args.seed,
                num_origins=args.num_origins,
                sample_fps=args.sample_fps,
                epsilon_sec=args.epsilon_sec,
            )
        )
    output_path = Path(args.output)
    write_manifests_jsonl(output_path, manifests)
    print(f"Wrote {len(manifests)} sampling manifests to {output_path}")
    return 0


def _select_video_indices(values: Sequence[Any], indices: Sequence[int] | None) -> list[Any]:
    if indices is None:
        return list(values)
    selected = []
    seen: set[int] = set()
    for index in indices:
        if index in seen:
            raise ValueError(f"duplicate video index: {index}")
        if index < 0 or index >= len(values):
            raise ValueError(f"video index {index} is outside [0, {len(values)})")
        seen.add(index)
        selected.append(values[index])
    return selected


def _run_make_benchmark_manifests(args: argparse.Namespace) -> int:
    videos = load_benchmark_videos(
        args.benchmark,
        args.questions_file,
        args.dataset_root,
    )
    videos = _select_video_indices(videos, args.video_indices)
    manifests = build_benchmark_manifests(
        videos,
        master_seed=args.seed,
        num_origins=args.num_origins,
        sample_fps=args.sample_fps,
        probe_missing_duration=not args.no_probe_missing_duration,
    )
    output_path = Path(args.output)
    write_manifests_jsonl(output_path, manifests)
    if args.catalog_output is not None:
        rows = [
            {
                "dataset": video.dataset,
                "video_id": video.video_id,
                "video_path": video.video_path.resolve(),
                "duration_sec": manifest.duration_sec,
                "num_questions": len(video.queries),
                "question_ids": [query.question_id for query in video.queries],
            }
            for video, manifest in zip(videos, manifests)
        ]
        _write_jsonl(args.catalog_output, rows)
    print(f"Wrote {len(manifests)} benchmark manifests to {output_path}")
    return 0


def _load_feature_extractor(feature_model: str, model_path: str | None, device: str | None):
    resolved_model_path = model_path or FEATURE_MODEL_DEFAULTS[feature_model]
    try:
        module = importlib.import_module("preprocess.extract")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The official feature-extraction dependencies are missing. "
            "Install requirements.txt before running preprocess-benchmark."
        ) from exc
    if device is None:
        torch = importlib.import_module("torch")
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    extractor = module.build_extractor(feature_model, resolved_model_path, resolved_device)
    return extractor, resolved_model_path, resolved_device


def _run_preprocess_benchmark(args: argparse.Namespace) -> int:
    videos = load_benchmark_videos(
        args.benchmark,
        args.questions_file,
        args.dataset_root,
    )
    manifests = read_manifests_jsonl(args.manifests)
    manifest_ids = {manifest.video_id for manifest in manifests}
    videos = [video for video in videos if video.video_id in manifest_ids]
    if len(videos) != len(manifest_ids):
        known = {video.video_id for video in videos}
        missing = sorted(manifest_ids - known)
        raise ValueError(f"manifest video IDs missing from annotations: {missing[:5]}")

    extractor, model_path, device = _load_feature_extractor(
        args.feature_model,
        args.model_path,
        args.device,
    )
    try:
        image_module = importlib.import_module("PIL.Image")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Pillow is required for BLIP/CLIP preprocessing") from exc
    destination = preprocess_benchmark_to_jsonl(
        videos,
        manifests,
        extractor=extractor,
        output_dir=args.output_dir,
        signal_jsonl=args.signal_jsonl,
        batch_size=args.batch_size,
        frame_buffer_size=args.frame_buffer_size,
        frame_adapter=image_module.fromarray,
        record_metadata={
            "feature_model": args.feature_model,
            "feature_model_revision": model_path,
            "feature_device": device,
        },
    )
    write_reproducibility_manifests(
        args.output_dir,
        command="preprocess-benchmark",
        config={
            "benchmark": args.benchmark,
            "feature_model": args.feature_model,
            "feature_model_revision": model_path,
            "device": device,
            "batch_size": args.batch_size,
            "frame_buffer_size": args.frame_buffer_size,
        },
        input_paths=(args.questions_file, args.manifests),
        extra={"signal_jsonl": str(Path(destination).resolve())},
    )
    print(f"Wrote benchmark origin signals to {destination}")
    return 0


def _run_export_keyframes(args: argparse.Namespace) -> int:
    paths = export_trace_jsonl(
        args.traces,
        benchmark=args.benchmark,
        questions_file=args.questions_file,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        methods=args.methods,
        origin_ids=args.origin_ids,
        strict=not args.allow_partial,
        expected_budget=args.expected_budget,
    )
    for (method, origin_id), path in sorted(paths.items()):
        print(f"{method}/origin={origin_id}: {path}")
    return 0


def _run_selection_baselines(args: argparse.Namespace) -> int:
    records = read_signal_records(args.input)
    rows, metrics = run_selection_baselines(
        records,
        args.output_dir,
        frame_budget=args.frame_budget,
        methods=args.methods,
        selected_tolerance_sec=args.selected_tolerance_sec,
    )
    print(
        f"Wrote {len(rows)} baseline traces and {len(metrics)} item metrics "
        f"to {args.output_dir}"
    )
    return 0


def _run_matched_boundaries(args: argparse.Namespace) -> int:
    rows = list(iter_jsonl(args.traces))
    if args.count is not None:
        boundary_counts: int | Mapping[str, int] = args.count
    else:
        with Path(args.counts_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("counts JSON must be an object mapping video IDs to counts")
        boundary_counts = {str(key): int(value) for key, value in payload.items()}
    metrics = compute_matched_cardinality_metrics(
        rows,
        boundary_counts,
        tolerance_sec=args.tolerance_sec,
        edge_margin_sec=args.edge_margin_sec,
    )
    output = _write_jsonl(args.output, metrics)
    print(f"Wrote {len(metrics)} matched-cardinality rows to {output}")
    return 0


def _add_transform_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("dwt", "swt", "cycle_spin", "gaussian"),
        default=("dwt", "swt"),
        help="Temporal transforms to compare (default: dwt swt).",
    )
    parser.add_argument("--wavelet", default="db4", help="PyWavelets family.")
    parser.add_argument("--level", type=int, default=None, help="Fixed decomposition level.")
    parser.add_argument(
        "--drift-level",
        type=int,
        default=3,
        help="Automatic-level reduction when --level is omitted.",
    )
    parser.add_argument(
        "--shared-padding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use common reflection padding before either transform.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the reusable top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="phase-stable",
        description="Run sampling-origin stability experiments for WFS-SB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze-signals",
        help="Run DWT/TI-DWT analysis from OriginSignalRecord JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    analyze.add_argument("input", help="OriginSignalRecord JSONL input.")
    analyze.add_argument("output_dir", help="Directory for traces, metrics, and summary.")
    analyze.add_argument(
        "--config",
        help="Validated YAML config; experiment/selection sections override CLI defaults.",
    )
    _add_transform_options(analyze)
    analyze.add_argument("--frame-budget", type=int, default=16)
    analyze.add_argument("--min-distance-ratio", type=float, default=0.02)
    analyze.add_argument("--min-distance-absolute", type=int, default=5)
    analyze.add_argument("--padding-mode", default="reflect")
    analyze.add_argument("--dwt-mode", default="symmetric")
    analyze.add_argument(
        "--swt-norm", action=argparse.BooleanOptionalAction, default=True
    )
    analyze.add_argument(
        "--cycle-shifts", nargs="+", type=int, default=tuple(range(16))
    )
    analyze.add_argument(
        "--cycle-aggregation", choices=("mean", "median"), default="mean"
    )
    analyze.add_argument("--gaussian-sigma", type=float, default=None)
    analyze.add_argument("--boundary-tolerance-sec", type=float, default=1.0)
    analyze.add_argument(
        "--report-boundary-tolerances-sec",
        nargs="+",
        type=float,
        default=(0.5, 1.0, 2.0),
    )
    analyze.add_argument("--selected-tolerance-sec", type=float, default=1.0)
    analyze.add_argument("--edge-margin-sec", type=float, default=0.0)
    analyze.add_argument("--baseline-method", default="dwt")
    analyze.add_argument("--treatment-method", default="swt")
    analyze.add_argument(
        "--bootstrap-metrics",
        nargs="+",
        default=DEFAULT_BOOTSTRAP_METRICS,
        help="Item metric fields for paired video-cluster confidence intervals.",
    )
    analyze.add_argument("--n-bootstrap", type=int, default=10_000)
    analyze.add_argument("--confidence", type=float, default=0.95)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.set_defaults(handler=_run_analyze_signals)

    shifts = subparsers.add_parser(
        "controlled-shifts",
        help="Measure inverse-aligned circular-shift consistency for a .npy signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    shifts.add_argument("input", help="One-dimensional NumPy .npy signal.")
    shifts.add_argument("output", help="Output metric JSON path.")
    _add_transform_options(shifts)
    shifts.add_argument("--shifts", nargs="+", type=int, default=tuple(range(16)))
    shifts.add_argument(
        "--edge-samples",
        type=int,
        default=None,
        help="Interior crop per edge; default derives it from db4 support and shifts.",
    )
    shifts.set_defaults(handler=_run_controlled_shifts)

    predictions = subparsers.add_parser(
        "evaluate-predictions",
        help="Aggregate prediction JSONL and paired downstream confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predictions.add_argument("input", help="Prediction JSONL input.")
    predictions.add_argument("output", help="Output summary JSON path.")
    predictions.add_argument("--baseline-method", default="dwt")
    predictions.add_argument("--treatment-method", default="swt")
    predictions.add_argument("--n-bootstrap", type=int, default=10_000)
    predictions.add_argument("--confidence", type=float, default=0.95)
    predictions.add_argument("--seed", type=int, default=0)
    predictions.set_defaults(handler=_run_evaluate_predictions)

    manifests = subparsers.add_parser(
        "make-manifests",
        help="Create deterministic random-origin SamplingManifest JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    manifests.add_argument("input", help="JSONL containing video_id and duration_sec.")
    manifests.add_argument("output", help="Output SamplingManifest JSONL path.")
    manifests.add_argument("--seed", type=int, default=0)
    manifests.add_argument("--num-origins", type=int, default=5)
    manifests.add_argument("--sample-fps", type=float, default=1.0)
    manifests.add_argument("--epsilon-sec", type=float, default=1e-9)
    manifests.set_defaults(handler=_run_make_manifests)

    benchmark_manifests = subparsers.add_parser(
        "make-benchmark-manifests",
        help="Build origin manifests directly from benchmark annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    benchmark_manifests.add_argument(
        "--benchmark",
        required=True,
        choices=("videomme", "lvb", "longvideobench", "mlvu"),
    )
    benchmark_manifests.add_argument("--questions-file", required=True)
    benchmark_manifests.add_argument("--dataset-root", required=True)
    benchmark_manifests.add_argument("--output", required=True)
    benchmark_manifests.add_argument("--catalog-output")
    benchmark_manifests.add_argument("--video-indices", nargs="+", type=int)
    benchmark_manifests.add_argument("--seed", type=int, default=0)
    benchmark_manifests.add_argument("--num-origins", type=int, default=5)
    benchmark_manifests.add_argument("--sample-fps", type=float, default=1.0)
    benchmark_manifests.add_argument(
        "--no-probe-missing-duration",
        action="store_true",
        help="Fail instead of probing videos whose annotations omit duration.",
    )
    benchmark_manifests.set_defaults(handler=_run_make_benchmark_manifests)

    preprocess = subparsers.add_parser(
        "preprocess-benchmark",
        help="Decode manifest PTS grids and compute query-conditioned score signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    preprocess.add_argument(
        "--benchmark",
        required=True,
        choices=("videomme", "lvb", "longvideobench", "mlvu"),
    )
    preprocess.add_argument("--questions-file", required=True)
    preprocess.add_argument("--dataset-root", required=True)
    preprocess.add_argument("--manifests", required=True)
    preprocess.add_argument("--output-dir", required=True)
    preprocess.add_argument("--signal-jsonl", required=True)
    preprocess.add_argument(
        "--feature-model",
        choices=tuple(FEATURE_MODEL_DEFAULTS),
        default="blip2",
    )
    preprocess.add_argument("--model-path")
    preprocess.add_argument("--device")
    preprocess.add_argument("--batch-size", type=int, default=256)
    preprocess.add_argument(
        "--frame-buffer-size",
        type=int,
        default=256,
        help="Maximum matched RGB targets held in host RAM before extraction.",
    )
    preprocess.set_defaults(handler=_run_preprocess_benchmark)

    export = subparsers.add_parser(
        "export-keyframes",
        help="Export trace selections as method/origin lmms-eval annotation JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    export.add_argument("--traces", required=True)
    export.add_argument(
        "--benchmark",
        required=True,
        choices=("videomme", "lvb", "longvideobench", "mlvu"),
    )
    export.add_argument("--questions-file", required=True)
    export.add_argument("--dataset-root", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--methods", nargs="+")
    export.add_argument("--origin-ids", nargs="+", type=int)
    export.add_argument("--expected-budget", type=int)
    export.add_argument("--allow-partial", action="store_true")
    export.set_defaults(handler=_run_export_keyframes)

    baselines = subparsers.add_parser(
        "selection-baselines",
        help="Generate Uniform and Top-K traces on the same origin manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    baselines.add_argument("input", help="OriginSignalRecord JSONL input.")
    baselines.add_argument("output_dir")
    baselines.add_argument(
        "--methods", nargs="+", choices=("uniform", "topk"), default=("uniform", "topk")
    )
    baselines.add_argument("--frame-budget", type=int, default=16)
    baselines.add_argument("--selected-tolerance-sec", type=float, default=1.0)
    baselines.set_defaults(handler=_run_selection_baselines)

    matched = subparsers.add_parser(
        "matched-boundaries",
        help="Evaluate fixed, calibration-derived top-B boundary counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    matched.add_argument("--traces", required=True)
    matched.add_argument("--output", required=True)
    count_group = matched.add_mutually_exclusive_group(required=True)
    count_group.add_argument("--count", type=int)
    count_group.add_argument(
        "--counts-json",
        help="JSON object keyed by video_id or dataset/video_id.",
    )
    matched.add_argument("--tolerance-sec", type=float, default=1.0)
    matched.add_argument("--edge-margin-sec", type=float, default=0.0)
    matched.set_defaults(handler=_run_matched_boundaries)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and run one CLI command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_parser", "main", "to_jsonable"]
