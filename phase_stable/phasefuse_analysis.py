"""Strict, compute-matched evaluation utilities for phase-fusion experiments.

The functions in this module deliberately operate on in-memory rows.  They do
not intersect incomplete arms or infer missing origins: a comparison is valid
only when both methods contain the same items, origins, candidate frames, and
selection budget.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from numbers import Integral, Real
from typing import Any

import numpy as np

from .artifacts import OriginSignalRecord
from .metrics import (
    mllm_stability_metrics,
    selected_timestamp_metrics,
    video_cluster_paired_bootstrap,
)
from .policy import FIDELITY_METRICS, qvhighlights_selection_fidelity

ItemKey = tuple[str, str, str]
TraceKey = tuple[str, str, str, int]

_TIMESTAMP_F1_TOLERANCES = (
    ("0p25s", 0.25),
    ("0p5s", 0.5),
    ("1s", 1.0),
)
_PRIMARY_TIMESTAMP_F1_METRIC = "selected_timestamp_f1_at_0p5s_mean"
_PRIMARY_OUTER_UNCERTAINTY_METRIC = "outer_origin_selector_uncertainty"

# Timestamp agreement on decoded-frame PTS is the primary selector-stability
# family.  Exact source-frame identity remains useful as a stricter, secondary
# diagnostic, and the two unsuffixed timestamp fields are retained as aliases
# of the historical 1.0-second metric for output-schema compatibility.
_PRIMARY_CONSISTENCY_METRICS = tuple(
    f"selected_timestamp_f1_at_{label}_{summary}"
    for label, _ in _TIMESTAMP_F1_TOLERANCES
    for summary in ("mean", "worst")
) + (_PRIMARY_OUTER_UNCERTAINTY_METRIC,)
_SECONDARY_EXACT_SOURCE_METRICS = (
    "selected_set_consistency",
    "selected_set_jaccard_worst",
    "selected_set_overlap_mean",
    "selected_all_origin_intersection_fraction",
    "outer_origin_selected_set_uncertainty",
)
_LEGACY_CONSISTENCY_METRICS = (
    "selected_set_consistency",
    "selected_set_jaccard_worst",
    "selected_set_overlap_mean",
    "selected_all_origin_intersection_fraction",
    "selected_timestamp_f1_mean",
    "selected_timestamp_f1_worst",
    "outer_origin_selected_set_uncertainty",
)
_CONSISTENCY_METRICS = (
    *_LEGACY_CONSISTENCY_METRICS,
    *_PRIMARY_CONSISTENCY_METRICS,
)

_INNER_PHASE_FIELDS = (
    "selected_phase_uncertainty_mean",
    "selected_phase_uncertainty_max",
    "phase_uncertainty_mean",
)


@dataclass(frozen=True)
class _TraceGrid:
    method: str
    item_keys: tuple[ItemKey, ...]
    origin_ids: tuple[int, ...]
    frame_budget: int
    rows: Mapping[ItemKey, Mapping[int, Mapping[str, Any]]]


@dataclass(frozen=True)
class _PredictionGrid:
    method: str
    item_keys: tuple[ItemKey, ...]
    origin_ids: tuple[int, ...]
    predictions: Mapping[ItemKey, Mapping[int, Any]]
    gold: Mapping[ItemKey, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("analysis result contains NaN/inf")
        return result
    return value


def _text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _origin(row: Mapping[str, Any]) -> int:
    value = row.get("origin_id")
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("origin_id must be a non-negative integer")
    return int(value)


def _sequence(row: Mapping[str, Any], name: str) -> list[Any]:
    value = row.get(name)
    if isinstance(value, np.ndarray):
        result = value.tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = list(value)
    else:
        raise TypeError(f"{name} must be a sequence")
    return result


def _integer_sequence(
    row: Mapping[str, Any], name: str, *, nonempty: bool = False
) -> tuple[int, ...]:
    values = _sequence(row, name)
    if nonempty and not values:
        raise ValueError(f"{name} must not be empty")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
            raise ValueError(f"{name} must contain non-negative integers")
        result.append(int(value))
    return tuple(result)


def _float_sequence(
    row: Mapping[str, Any], name: str, *, nonempty: bool = False
) -> tuple[float, ...]:
    values = _sequence(row, name)
    if nonempty and not values:
        raise ValueError(f"{name} must not be empty")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain finite real numbers")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} must contain finite real numbers")
        result.append(number)
    return tuple(result)


def _validate_trace_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping):
        raise TypeError("trace rows must be mappings")
    for name in ("dataset", "video_id", "question_id", "method"):
        _text(row, name)
    _origin(row)

    candidate_ids = _integer_sequence(row, "source_frame_indices", nonempty=True)
    selected_ids = _integer_sequence(
        row, "selected_source_frame_indices", nonempty=True
    )
    selected_indices = _integer_sequence(row, "selected_indices", nonempty=True)
    candidate_times = _float_sequence(row, "actual_pts_sec", nonempty=True)
    selected_times = _float_sequence(row, "selected_actual_pts_sec", nonempty=True)
    target_times = _float_sequence(row, "timestamps_sec", nonempty=True)

    if len({len(candidate_ids), len(candidate_times), len(target_times)}) != 1:
        raise ValueError("candidate frame IDs and timestamps must be frame-aligned")
    if len({len(selected_ids), len(selected_indices), len(selected_times)}) != 1:
        raise ValueError("selected frame IDs, indices, and timestamps must be aligned")
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_source_frame_indices must be unique")
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("selected_indices must be unique")
    if any(right <= left for left, right in pairwise(target_times)):
        raise ValueError("timestamps_sec must be strictly increasing")
    if any(right < left for left, right in pairwise(candidate_times)):
        raise ValueError("actual_pts_sec must be non-decreasing")
    if any(right < left for left, right in pairwise(candidate_ids)):
        raise ValueError("source_frame_indices must be non-decreasing")
    if any(right <= left for left, right in pairwise(selected_indices)):
        raise ValueError("selected_indices must be strictly increasing")
    if any(index >= len(candidate_ids) for index in selected_indices):
        raise ValueError("selected_indices contains an out-of-range candidate index")
    expected_ids = tuple(candidate_ids[index] for index in selected_indices)
    if expected_ids != selected_ids:
        raise ValueError("selected source-frame IDs do not match selected_indices")
    expected_times = np.asarray(
        [candidate_times[index] for index in selected_indices], dtype=float
    )
    if not np.allclose(expected_times, np.asarray(selected_times), rtol=0.0, atol=1e-9):
        raise ValueError("selected actual PTS do not match selected_indices")


def _build_trace_grid(
    trace_rows: Sequence[Mapping[str, Any]], method: str
) -> _TraceGrid:
    if not isinstance(method, str) or not method.strip():
        raise ValueError("method must be a non-empty string")
    grouped: dict[ItemKey, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    budgets: set[int] = set()
    for row in trace_rows:
        if not isinstance(row, Mapping):
            raise TypeError("trace rows must be mappings")
        if row.get("method") != method:
            continue
        _validate_trace_row(row)
        item = (
            _text(row, "dataset"),
            _text(row, "video_id"),
            _text(row, "question_id"),
        )
        origin_id = _origin(row)
        if origin_id in grouped[item]:
            raise ValueError(
                f"duplicate trace row for {method}/{item}/origin={origin_id}"
            )
        grouped[item][origin_id] = row
        budgets.add(len(_integer_sequence(row, "selected_source_frame_indices")))
    if not grouped:
        raise ValueError(f"no trace rows found for method {method!r}")
    if len(budgets) != 1:
        raise ValueError(f"method {method!r} does not use one fixed frame budget")
    item_keys = tuple(sorted(grouped))
    origin_sets = [tuple(sorted(grouped[item])) for item in item_keys]
    if any(origins != origin_sets[0] for origins in origin_sets[1:]):
        raise ValueError(f"method {method!r} does not have a rectangular origin grid")
    if len(origin_sets[0]) < 2:
        raise ValueError("selected-set phase consistency requires at least two origins")
    return _TraceGrid(
        method=method,
        item_keys=item_keys,
        origin_ids=origin_sets[0],
        frame_budget=budgets.pop(),
        rows={item: dict(rows) for item, rows in grouped.items()},
    )


def _align_trace_grids(baseline: _TraceGrid, treatment: _TraceGrid) -> None:
    if baseline.item_keys != treatment.item_keys:
        raise ValueError("baseline and treatment trace items do not align exactly")
    if baseline.origin_ids != treatment.origin_ids:
        raise ValueError("baseline and treatment origin grids do not align exactly")
    if baseline.frame_budget != treatment.frame_budget:
        raise ValueError("baseline and treatment frame budgets are not compute-matched")
    for item in baseline.item_keys:
        for origin_id in baseline.origin_ids:
            first = baseline.rows[item][origin_id]
            second = treatment.rows[item][origin_id]
            first_ids = _integer_sequence(first, "source_frame_indices", nonempty=True)
            second_ids = _integer_sequence(
                second, "source_frame_indices", nonempty=True
            )
            if first_ids != second_ids:
                raise ValueError(
                    "baseline and treatment candidate source frames do not align exactly"
                )
            for field in ("timestamps_sec", "actual_pts_sec"):
                left = np.asarray(_float_sequence(first, field, nonempty=True))
                right = np.asarray(_float_sequence(second, field, nonempty=True))
                if left.shape != right.shape or not np.allclose(
                    left, right, rtol=0.0, atol=1e-9
                ):
                    raise ValueError(
                        f"baseline and treatment candidate {field} do not align exactly"
                    )


def _trace_item_metrics(grid: _TraceGrid) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in grid.item_keys:
        traces = [grid.rows[item][origin] for origin in grid.origin_ids]
        selections = [
            frozenset(_integer_sequence(row, "selected_source_frame_indices"))
            for row in traces
        ]
        selected_times = [
            _float_sequence(row, "selected_actual_pts_sec", nonempty=True)
            for row in traces
        ]
        jaccards: list[float] = []
        overlaps: list[float] = []
        timestamp_f1 = {label: [] for label, _ in _TIMESTAMP_F1_TOLERANCES}
        for left, right in combinations(range(len(traces)), 2):
            intersection = len(selections[left] & selections[right])
            union = len(selections[left] | selections[right])
            jaccards.append(float(intersection / union))
            overlaps.append(
                float(intersection / min(len(selections[left]), len(selections[right])))
            )
            for label, tolerance in _TIMESTAMP_F1_TOLERANCES:
                timestamp_f1[label].append(
                    float(
                        selected_timestamp_metrics(
                            selected_times[left],
                            selected_times[right],
                            tolerance=tolerance,
                        )["f1"]
                    )
                )
        common = set(selections[0]).intersection(*selections[1:])
        consistency = float(np.mean(jaccards))
        timestamp_metrics = {
            f"selected_timestamp_f1_at_{label}_{summary}": float(reducer(values))
            for label, values in timestamp_f1.items()
            for summary, reducer in (("mean", np.mean), ("worst", np.min))
        }
        primary_timestamp_consistency = timestamp_metrics[_PRIMARY_TIMESTAMP_F1_METRIC]
        rows.append(
            {
                "dataset": item[0],
                "video_id": item[1],
                "question_id": item[2],
                "method": grid.method,
                "num_origins": len(grid.origin_ids),
                "frame_budget": grid.frame_budget,
                "selected_set_consistency": consistency,
                "selected_set_jaccard_worst": float(np.min(jaccards)),
                "selected_set_overlap_mean": float(np.mean(overlaps)),
                "selected_all_origin_intersection_fraction": float(
                    len(common) / grid.frame_budget
                ),
                **timestamp_metrics,
                "outer_origin_selector_uncertainty": float(
                    1.0 - primary_timestamp_consistency
                ),
                # Backward-compatible aliases for the original 1.0-second
                # timestamp-F1 fields.
                "selected_timestamp_f1_mean": timestamp_metrics[
                    "selected_timestamp_f1_at_1s_mean"
                ],
                "selected_timestamp_f1_worst": timestamp_metrics[
                    "selected_timestamp_f1_at_1s_worst"
                ],
                "outer_origin_selected_set_uncertainty": float(1.0 - consistency),
            }
        )
    return rows


def _single_phase_unavailability_reason(grid: _TraceGrid) -> str | None:
    """Identify selectors for which inner-phase disagreement is undefined."""

    if grid.method == "dense_swt":
        return "dense_swt is a direct single-stream selector (num_phases=1)"

    declared_phase_counts: set[int] = set()
    selector_labels: set[str] = set()
    for item in grid.item_keys:
        for origin in grid.origin_ids:
            row = grid.rows[item][origin]
            metadata = row.get("method_metadata")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError("method_metadata must be a mapping when supplied")

            phasefuse_metadata: Mapping[str, Any] | None = None
            phasefuse_config: Mapping[str, Any] | None = None
            if isinstance(metadata, Mapping):
                raw_phasefuse = metadata.get("phasefuse")
                if raw_phasefuse is not None and not isinstance(raw_phasefuse, Mapping):
                    raise TypeError("method_metadata.phasefuse must be a mapping")
                if isinstance(raw_phasefuse, Mapping):
                    phasefuse_metadata = raw_phasefuse
                    raw_config = raw_phasefuse.get("config")
                    if raw_config is not None and not isinstance(raw_config, Mapping):
                        raise TypeError(
                            "method_metadata.phasefuse.config must be a mapping"
                        )
                    if isinstance(raw_config, Mapping):
                        phasefuse_config = raw_config

            phase_values = [
                row.get("num_phases"),
                None if metadata is None else metadata.get("num_phases"),
                (
                    None
                    if phasefuse_metadata is None
                    else phasefuse_metadata.get("num_phases")
                ),
                (
                    None
                    if phasefuse_config is None
                    else phasefuse_config.get("num_phases")
                ),
            ]
            row_counts: set[int] = set()
            for value in phase_values:
                if value is None:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or int(value) < 1
                ):
                    raise ValueError(
                        "trace metadata num_phases must be a positive integer"
                    )
                row_counts.add(int(value))
            if len(row_counts) > 1:
                raise ValueError("trace row has conflicting num_phases metadata")
            declared_phase_counts.update(row_counts)

            selector_values = [
                row.get("selector"),
                row.get("selector_kind"),
                None if metadata is None else metadata.get("selector"),
                None if metadata is None else metadata.get("selector_kind"),
            ]
            selector_labels.update(
                str(value).strip().lower()
                for value in selector_values
                if isinstance(value, str) and value.strip()
            )

    if len(declared_phase_counts) > 1:
        raise ValueError(f"method {grid.method!r} mixes num_phases metadata")
    if declared_phase_counts and next(iter(declared_phase_counts)) < 2:
        return "trace metadata declares num_phases<2, so inner-phase uncertainty is undefined"
    if selector_labels & {"dense_swt", "direct_dense_single_stream"}:
        return "trace metadata identifies a direct single-stream selector"
    return None


def _optional_inner_phase_rows(
    grid: _TraceGrid,
) -> tuple[list[dict[str, Any]] | None, tuple[str, ...], str | None]:
    """Read optional raw inner-phase uncertainty without conflating origins.

    Each available scalar is averaged over outer origins to form one value per
    dataset/video/question.  A field must be present on every row of an arm or
    none of them; partial artifacts are evidence of an interrupted or mixed
    run and are rejected.
    """

    single_phase_reason = _single_phase_unavailability_reason(grid)
    if single_phase_reason is not None:
        return None, (), single_phase_reason

    flat_rows = [
        grid.rows[item][origin] for item in grid.item_keys for origin in grid.origin_ids
    ]
    available_fields: list[str] = []
    for name in _INNER_PHASE_FIELDS:
        present = [name in row and row[name] is not None for row in flat_rows]
        if any(present) and not all(present):
            raise ValueError(
                f"method {grid.method!r} has partially missing optional field {name!r}"
            )
        if all(present):
            available_fields.append(name)
    primary = _INNER_PHASE_FIELDS[0]
    if primary not in available_fields:
        companions = [name for name in available_fields if name != primary]
        if companions:
            raise ValueError(
                f"method {grid.method!r} provides inner-phase companion fields "
                f"without {primary!r}"
            )
        return (
            None,
            (),
            ("selected_phase_uncertainty_mean is absent from every trace row"),
        )

    result: list[dict[str, Any]] = []
    for item in grid.item_keys:
        values_by_name: dict[str, list[float]] = defaultdict(list)
        for origin in grid.origin_ids:
            row = grid.rows[item][origin]
            for name in available_fields:
                value = row[name]
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError(f"{name} must be a finite non-negative scalar")
                numeric = float(value)
                if not np.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(f"{name} must be a finite non-negative scalar")
                values_by_name[name].append(numeric)
            if "selected_phase_uncertainty_max" in available_fields and float(
                row["selected_phase_uncertainty_max"]
            ) < float(row["selected_phase_uncertainty_mean"]):
                raise ValueError(
                    "selected_phase_uncertainty_max must be >= "
                    "selected_phase_uncertainty_mean"
                )
        result.append(
            {
                "dataset": item[0],
                "video_id": item[1],
                "question_id": item[2],
                "method": grid.method,
                **{
                    name: float(np.mean(values_by_name[name]))
                    for name in available_fields
                },
            }
        )
    return result, tuple(available_fields), None


def selected_set_consistency(
    trace_rows: Sequence[Mapping[str, Any]], *, method: str
) -> list[dict[str, Any]]:
    """Return exact-source-frame and tolerant-PTS consistency per item."""

    return _jsonable(_trace_item_metrics(_build_trace_grid(trace_rows, method)))


def phase_uncertainty_failure_calibration(
    uncertainty: Sequence[float] | np.ndarray,
    failure: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Measure whether phase uncertainty ranks and predicts failure.

    AURC retains low-uncertainty items first.  Tied uncertainty values use the
    expected risk under a uniform random ordering inside the tie, making the
    result row-order invariant.  ``failure`` may be binary or a soft loss in
    ``[0, 1]``; the reported Brier score is mean squared probability error.
    """

    uncertainties = np.asarray(uncertainty, dtype=float)
    failures = np.asarray(failure, dtype=float)
    if uncertainties.ndim != 1 or failures.ndim != 1 or uncertainties.size == 0:
        raise ValueError("uncertainty and failure must be non-empty 1-D arrays")
    if uncertainties.shape != failures.shape:
        raise ValueError("uncertainty and failure must have identical shapes")
    if not np.all(np.isfinite(uncertainties)) or not np.all(np.isfinite(failures)):
        raise ValueError("uncertainty and failure must be finite")
    if np.any((uncertainties < 0.0) | (uncertainties > 1.0)):
        raise ValueError("uncertainty must lie in [0, 1]")
    if np.any((failures < 0.0) | (failures > 1.0)):
        raise ValueError("failure must lie in [0, 1]")

    order = np.argsort(uncertainties, kind="stable")
    ordered_uncertainty = uncertainties[order]
    ordered_failure = failures[order]
    risks: list[float] = []
    accepted = 0
    accepted_failure = 0.0
    start = 0
    while start < ordered_uncertainty.size:
        stop = start + 1
        while (
            stop < ordered_uncertainty.size
            and ordered_uncertainty[stop] == ordered_uncertainty[start]
        ):
            stop += 1
        block = ordered_failure[start:stop]
        block_mean = float(np.mean(block))
        for within_block in range(1, block.size + 1):
            risks.append(
                float(
                    (accepted_failure + within_block * block_mean)
                    / (accepted + within_block)
                )
            )
        accepted += int(block.size)
        accepted_failure += float(np.sum(block))
        start = stop

    oracle_failure = np.sort(failures)
    denominators = np.arange(1, failures.size + 1, dtype=float)
    oracle_risks = np.cumsum(oracle_failure) / denominators
    aurc = float(np.mean(risks))
    oracle_aurc = float(np.mean(oracle_risks))
    return {
        "num_items": int(failures.size),
        "aurc": aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": float(max(0.0, aurc - oracle_aurc)),
        "brier_score": float(np.mean(np.square(uncertainties - failures))),
        "mean_uncertainty": float(np.mean(uncertainties)),
        "mean_failure": float(np.mean(failures)),
        "calibration_bias": float(np.mean(uncertainties - failures)),
        "coverage": (np.arange(1, failures.size + 1) / failures.size).tolist(),
        "risk": risks,
        "tie_policy": "expected_uniform_order_within_equal_uncertainty",
    }


def _record_index(
    qv_records: Sequence[OriginSignalRecord | Mapping[str, Any]],
) -> dict[TraceKey, OriginSignalRecord]:
    result: dict[TraceKey, OriginSignalRecord] = {}
    for raw_record in qv_records:
        record = (
            raw_record
            if isinstance(raw_record, OriginSignalRecord)
            else OriginSignalRecord.from_dict(raw_record)
        )
        key = (*record.item_key, record.origin_id)
        if key in result:
            raise ValueError(f"duplicate QV record: {key}")
        result[key] = record
    if not result:
        raise ValueError("qv_records must not be empty when supplied")
    return result


def _add_qv_fidelity(
    grid: _TraceGrid,
    item_rows: list[dict[str, Any]],
    records: Mapping[TraceKey, OriginSignalRecord],
) -> None:
    expected = {
        (*item, origin_id) for item in grid.item_keys for origin_id in grid.origin_ids
    }
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(
            "QV records do not align exactly with trace items/origins "
            f"(missing={missing[:1]}, extra={extra[:1]})"
        )
    row_by_item = {
        (row["dataset"], row["video_id"], row["question_id"]): row for row in item_rows
    }
    for item in grid.item_keys:
        origin_fidelity: list[dict[str, float | int]] = []
        for origin_id in grid.origin_ids:
            trace = grid.rows[item][origin_id]
            record = records[(*item, origin_id)]
            if tuple(record.source_frame_indices) != _integer_sequence(
                trace, "source_frame_indices", nonempty=True
            ):
                raise ValueError("QV record source frames do not match its trace row")
            if not np.allclose(
                np.asarray(record.actual_pts_sec),
                np.asarray(_float_sequence(trace, "actual_pts_sec", nonempty=True)),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("QV record actual PTS do not match its trace row")
            selected = _integer_sequence(trace, "selected_indices", nonempty=True)
            fidelity = qvhighlights_selection_fidelity(record, selected)
            if fidelity is None:
                raise ValueError(
                    f"QV record {(*item, origin_id)} lacks fidelity metadata"
                )
            origin_fidelity.append(fidelity)
        target = row_by_item[item]
        for metric in FIDELITY_METRICS:
            target[metric] = float(
                np.mean([float(values[metric]) for values in origin_fidelity])
            )
        target["qv_evidence_loss"] = float(1.0 - target["selected_relevant_fraction"])
        target["qv_zero_relevant_origin_rate"] = float(
            np.mean(
                [
                    int(values["selected_relevant_count"]) == 0
                    for values in origin_fidelity
                ]
            )
        )


def _method_summary(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[str], *, calibrated: bool
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_items": len(rows),
        "num_videos": len({(row["dataset"], row["video_id"]) for row in rows}),
        "metrics": {
            metric: float(np.mean([float(row[metric]) for row in rows]))
            for metric in metrics
        },
    }
    if calibrated:
        failure = [float(row["qv_evidence_loss"]) for row in rows]
        summary["outer_origin_selector_failure_calibration"] = (
            phase_uncertainty_failure_calibration(
                [float(row[_PRIMARY_OUTER_UNCERTAINTY_METRIC]) for row in rows],
                failure,
            )
        )
        summary["outer_origin_selector_failure_calibration"]["failure_definition"] = (
            "1 - mean selected_relevant_fraction across origins"
        )
        # Preserve the exact-source calibration block used by schema v1 as a
        # secondary diagnostic.
        summary["outer_origin_selected_set_failure_calibration"] = (
            phase_uncertainty_failure_calibration(
                [float(row["outer_origin_selected_set_uncertainty"]) for row in rows],
                failure,
            )
        )
        summary["outer_origin_selected_set_failure_calibration"][
            "failure_definition"
        ] = "1 - mean selected_relevant_fraction across origins"
    return summary


def _inner_probability(raw_uncertainty: Sequence[float]) -> np.ndarray:
    """Map non-negative raw scaled-MAD values to a fixed [0, 1) score."""

    values = np.asarray(raw_uncertainty, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "raw inner-phase uncertainty must be a finite non-empty vector"
        )
    if np.any(values < 0.0):
        raise ValueError("raw inner-phase uncertainty must be non-negative")
    return values / (1.0 + values)


def _inner_method_block(
    rows: Sequence[Mapping[str, Any]] | None,
    fields: Sequence[str],
    *,
    failure_rows: Sequence[Mapping[str, Any]] | None,
    unavailable_reason: str | None = None,
) -> dict[str, Any]:
    if rows is None:
        return {
            "status": "unavailable",
            "reason": unavailable_reason
            or "selected_phase_uncertainty_mean is absent from every trace row",
            "available_fields": [],
        }
    block: dict[str, Any] = {
        "status": "available",
        "num_items": len(rows),
        "available_fields": list(fields),
        "outer_origin_aggregation": "arithmetic mean per dataset/video/question",
        "raw_metrics": {
            name: float(np.mean([float(row[name]) for row in rows])) for name in fields
        },
    }
    if failure_rows is None:
        block["failure_calibration"] = {
            "status": "unavailable",
            "reason": "no aligned failure target was supplied",
        }
        return block
    if len(rows) != len(failure_rows):
        raise RuntimeError("inner-phase uncertainty and failure rows lost alignment")
    raw = [float(row["selected_phase_uncertainty_mean"]) for row in rows]
    probability = _inner_probability(raw)
    calibration = phase_uncertainty_failure_calibration(
        probability,
        [float(row["qv_evidence_loss"]) for row in failure_rows],
    )
    calibration.update(
        {
            "status": "available",
            "raw_uncertainty_field": "selected_phase_uncertainty_mean",
            "probability_mapping": "u / (1 + u), fixed and unfitted",
            "failure_definition": (
                "1 - mean selected_relevant_fraction across origins"
            ),
        }
    )
    block["failure_calibration"] = calibration
    return block


def _inner_phase_comparison(
    baseline_rows: Sequence[Mapping[str, Any]] | None,
    treatment_rows: Sequence[Mapping[str, Any]] | None,
    common_fields: Sequence[str],
    *,
    baseline_failure_rows: Sequence[Mapping[str, Any]] | None,
    treatment_failure_rows: Sequence[Mapping[str, Any]] | None,
    clusters: Sequence[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if baseline_rows is None or treatment_rows is None:
        return {
            "status": "unavailable",
            "reason": (
                "paired inner-phase comparison requires "
                "selected_phase_uncertainty_mean in both arms"
            ),
        }
    baseline = np.asarray(
        [[float(row[name]) for name in common_fields] for row in baseline_rows],
        dtype=float,
    )
    treatment = np.asarray(
        [[float(row[name]) for name in common_fields] for row in treatment_rows],
        dtype=float,
    )
    primary_index = list(common_fields).index("selected_phase_uncertainty_mean")
    if (baseline_failure_rows is None) != (treatment_failure_rows is None):
        raise RuntimeError("paired inner-phase failure targets are inconsistent")
    calibrated = baseline_failure_rows is not None
    baseline_failures = (
        None
        if baseline_failure_rows is None
        else np.asarray(
            [float(row["qv_evidence_loss"]) for row in baseline_failure_rows],
            dtype=float,
        )
    )
    treatment_failures = (
        None
        if treatment_failure_rows is None
        else np.asarray(
            [float(row["qv_evidence_loss"]) for row in treatment_failure_rows],
            dtype=float,
        )
    )

    def statistic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        raw_effect = np.mean(second - first, axis=0)
        if baseline_failures is None or treatment_failures is None:
            return raw_effect
        first_calibration = phase_uncertainty_failure_calibration(
            _inner_probability(first[:, primary_index]),
            baseline_failures[: first.shape[0]],
        )
        second_calibration = phase_uncertainty_failure_calibration(
            _inner_probability(second[:, primary_index]),
            treatment_failures[: second.shape[0]],
        )
        return np.concatenate(
            (
                raw_effect,
                np.asarray(
                    [
                        second_calibration["aurc"] - first_calibration["aurc"],
                        second_calibration["excess_aurc"]
                        - first_calibration["excess_aurc"],
                        second_calibration["brier_score"]
                        - first_calibration["brier_score"],
                    ]
                ),
            )
        )

    # Encode each arm's failure target beside its uncertainty so cluster
    # resampling keeps each calibration pair synchronized.
    if calibrated:
        assert baseline_failures is not None and treatment_failures is not None
        baseline_bundle = np.column_stack((baseline, baseline_failures))
        treatment_bundle = np.column_stack((treatment, treatment_failures))
        raw_width = baseline.shape[1]

        def bundled_statistic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
            first_raw, first_failure = first[:, :raw_width], first[:, -1]
            second_raw = second[:, :raw_width]
            second_failure = second[:, -1]
            raw_effect = np.mean(second_raw - first_raw, axis=0)
            first_calibration = phase_uncertainty_failure_calibration(
                _inner_probability(first_raw[:, primary_index]), first_failure
            )
            second_calibration = phase_uncertainty_failure_calibration(
                _inner_probability(second_raw[:, primary_index]), second_failure
            )
            return np.concatenate(
                (
                    raw_effect,
                    np.asarray(
                        [
                            second_calibration["aurc"] - first_calibration["aurc"],
                            second_calibration["excess_aurc"]
                            - first_calibration["excess_aurc"],
                            second_calibration["brier_score"]
                            - first_calibration["brier_score"],
                        ]
                    ),
                )
            )

        bootstrap_baseline = baseline_bundle
        bootstrap_treatment = treatment_bundle
        bootstrap_statistic = bundled_statistic
    else:
        bootstrap_baseline = baseline
        bootstrap_treatment = treatment
        bootstrap_statistic = statistic
    comparison = video_cluster_paired_bootstrap(
        clusters,
        bootstrap_baseline,
        bootstrap_treatment,
        statistic=bootstrap_statistic,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    effect_order = [f"delta_raw_{name}" for name in common_fields]
    if calibrated:
        effect_order.extend(
            (
                "delta_inner_phase_failure_aurc",
                "delta_inner_phase_failure_excess_aurc",
                "delta_inner_phase_failure_brier",
            )
        )
    return {
        "status": "available",
        "effect_definition": "treatment - baseline",
        "effect_order": effect_order,
        "common_fields": list(common_fields),
        **comparison,
    }


def evaluate_phasefuse_analysis(
    trace_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    treatment_method: str,
    qv_records: Sequence[OriginSignalRecord | Mapping[str, Any]] | None = None,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate two strictly aligned, compute-matched trace arms.

    The paired bootstrap resamples whole ``dataset/video_id`` clusters, so all
    queries belonging to a sampled video move together in every arm.
    """

    if baseline_method == treatment_method:
        raise ValueError("baseline_method and treatment_method must differ")
    baseline = _build_trace_grid(trace_rows, baseline_method)
    treatment = _build_trace_grid(trace_rows, treatment_method)
    _align_trace_grids(baseline, treatment)
    baseline_rows = _trace_item_metrics(baseline)
    treatment_rows = _trace_item_metrics(treatment)
    (
        baseline_inner_rows,
        baseline_inner_fields,
        baseline_inner_unavailable_reason,
    ) = _optional_inner_phase_rows(baseline)
    (
        treatment_inner_rows,
        treatment_inner_fields,
        treatment_inner_unavailable_reason,
    ) = _optional_inner_phase_rows(treatment)

    metric_names = list(_CONSISTENCY_METRICS)
    calibrated = qv_records is not None
    if qv_records is not None:
        records = _record_index(qv_records)
        _add_qv_fidelity(baseline, baseline_rows, records)
        _add_qv_fidelity(treatment, treatment_rows, records)
        metric_names.extend(FIDELITY_METRICS)
        metric_names.extend(("qv_evidence_loss", "qv_zero_relevant_origin_rate"))

    baseline_matrix = np.asarray(
        [[float(row[name]) for name in metric_names] for row in baseline_rows],
        dtype=float,
    )
    treatment_matrix = np.asarray(
        [[float(row[name]) for name in metric_names] for row in treatment_rows],
        dtype=float,
    )
    uncertainty_index = metric_names.index(_PRIMARY_OUTER_UNCERTAINTY_METRIC)
    legacy_uncertainty_index = metric_names.index(
        "outer_origin_selected_set_uncertainty"
    )
    if calibrated:
        failure_index = metric_names.index("qv_evidence_loss")

    def statistic(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        result = np.mean(second - first, axis=0)
        if not calibrated:
            return result
        first_calibration = phase_uncertainty_failure_calibration(
            first[:, uncertainty_index], first[:, failure_index]
        )
        second_calibration = phase_uncertainty_failure_calibration(
            second[:, uncertainty_index], second[:, failure_index]
        )
        first_legacy_calibration = phase_uncertainty_failure_calibration(
            first[:, legacy_uncertainty_index], first[:, failure_index]
        )
        second_legacy_calibration = phase_uncertainty_failure_calibration(
            second[:, legacy_uncertainty_index], second[:, failure_index]
        )
        calibration_effect = np.asarray(
            [
                second_calibration["aurc"] - first_calibration["aurc"],
                second_calibration["excess_aurc"] - first_calibration["excess_aurc"],
                second_calibration["brier_score"] - first_calibration["brier_score"],
                second_legacy_calibration["aurc"] - first_legacy_calibration["aurc"],
                second_legacy_calibration["excess_aurc"]
                - first_legacy_calibration["excess_aurc"],
                second_legacy_calibration["brier_score"]
                - first_legacy_calibration["brier_score"],
            ],
            dtype=float,
        )
        return np.concatenate((result, calibration_effect))

    clusters = [f"{item[0]}\x1f{item[1]}" for item in baseline.item_keys]
    comparison = video_cluster_paired_bootstrap(
        clusters,
        baseline_matrix,
        treatment_matrix,
        statistic=statistic,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    effect_order = [f"delta_{name}" for name in metric_names]
    if calibrated:
        effect_order.extend(
            (
                "delta_outer_origin_selector_failure_aurc",
                "delta_outer_origin_selector_failure_excess_aurc",
                "delta_outer_origin_selector_failure_brier",
                "delta_outer_origin_selected_set_failure_aurc",
                "delta_outer_origin_selected_set_failure_excess_aurc",
                "delta_outer_origin_selected_set_failure_brier",
            )
        )
    comparison["effect_order"] = effect_order
    comparison["effect_definition"] = "treatment - baseline"

    result = {
        "schema_version": 1,
        "baseline_method": baseline_method,
        "treatment_method": treatment_method,
        "num_paired_items": len(baseline.item_keys),
        "origin_ids": list(baseline.origin_ids),
        "frame_budget": baseline.frame_budget,
        "cluster_unit": "dataset/video_id",
        "selector_stability": {
            "primary": {
                "family": "pairwise tolerant timestamp F1",
                "timestamp_field": "selected_actual_pts_sec",
                "reported_tolerances_sec": [
                    tolerance for _, tolerance in _TIMESTAMP_F1_TOLERANCES
                ],
                "reference_tolerance_sec": 0.5,
                "reference_metric": _PRIMARY_TIMESTAMP_F1_METRIC,
                "uncertainty_metric": _PRIMARY_OUTER_UNCERTAINTY_METRIC,
            },
            "secondary": {
                "family": "exact source-frame identity",
                "reference_metric": "selected_set_consistency",
                "uncertainty_metric": "outer_origin_selected_set_uncertainty",
            },
            "legacy_aliases": {
                "selected_timestamp_f1_mean": ("selected_timestamp_f1_at_1s_mean"),
                "selected_timestamp_f1_worst": ("selected_timestamp_f1_at_1s_worst"),
            },
        },
        "selected_set_definition": (
            "secondary: mean pairwise Jaccard of unique selected_source_frame_indices"
        ),
        "selected_timestamp_definition": (
            "primary: pairwise tolerant F1 of selected_actual_pts_sec at "
            "0.25, 0.5, and 1.0 seconds; 0.5 seconds is the reference tolerance"
        ),
        "outer_origin_selector_uncertainty_definition": (
            "1 - mean pairwise selected_actual_pts_sec F1 at 0.5-second tolerance"
        ),
        "outer_origin_selected_set_uncertainty_definition": (
            "secondary: 1 - mean pairwise Jaccard across outer-origin selected "
            "source-frame sets"
        ),
        "metric_directions": {
            **{
                name: "higher_is_better"
                for name in _CONSISTENCY_METRICS
                if name
                not in {
                    _PRIMARY_OUTER_UNCERTAINTY_METRIC,
                    "outer_origin_selected_set_uncertainty",
                }
            },
            _PRIMARY_OUTER_UNCERTAINTY_METRIC: "lower_is_better",
            "outer_origin_selected_set_uncertainty": "lower_is_better",
            **(
                {
                    "selected_relevant_fraction": "higher_is_better",
                    "relevant_window_recall": "higher_is_better",
                    "relevant_clip_recall": "higher_is_better",
                    "mean_selected_saliency_vote": "higher_is_better",
                    "mean_gt_clip_nearest_selected_sec": "lower_is_better",
                    "qv_evidence_loss": "lower_is_better",
                    "qv_zero_relevant_origin_rate": "lower_is_better",
                    "outer_origin_selector_failure_aurc": "lower_is_better",
                    "outer_origin_selector_failure_excess_aurc": "lower_is_better",
                    "outer_origin_selector_failure_brier": "lower_is_better",
                    "outer_origin_selected_set_failure_aurc": "lower_is_better",
                    "outer_origin_selected_set_failure_excess_aurc": (
                        "lower_is_better"
                    ),
                    "outer_origin_selected_set_failure_brier": "lower_is_better",
                }
                if calibrated
                else {}
            ),
        },
        "methods": {
            baseline_method: _method_summary(
                baseline_rows, metric_names, calibrated=calibrated
            ),
            treatment_method: _method_summary(
                treatment_rows, metric_names, calibrated=calibrated
            ),
        },
        "comparison": comparison,
        "item_rows": baseline_rows + treatment_rows,
        "inner_phase_uncertainty": {
            "definition": (
                "raw scaled-MAD disagreement among aligned interleaved inner phases; "
                "distinct from outer-origin selected-set uncertainty"
            ),
            "raw_scale": "non-negative and not assumed to be a probability",
            "methods": {
                baseline_method: _inner_method_block(
                    baseline_inner_rows,
                    baseline_inner_fields,
                    failure_rows=baseline_rows if calibrated else None,
                    unavailable_reason=baseline_inner_unavailable_reason,
                ),
                treatment_method: _inner_method_block(
                    treatment_inner_rows,
                    treatment_inner_fields,
                    failure_rows=treatment_rows if calibrated else None,
                    unavailable_reason=treatment_inner_unavailable_reason,
                ),
            },
            "comparison": _inner_phase_comparison(
                baseline_inner_rows,
                treatment_inner_rows,
                tuple(
                    name
                    for name in _INNER_PHASE_FIELDS
                    if name in baseline_inner_fields and name in treatment_inner_fields
                ),
                baseline_failure_rows=baseline_rows if calibrated else None,
                treatment_failure_rows=treatment_rows if calibrated else None,
                clusters=clusters,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            ),
        },
    }
    return _jsonable(result)


def _valid_label(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, (Mapping, list, tuple, np.ndarray)):
        raise ValueError(f"{name} must be a non-missing scalar label")
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"{name} must be a non-missing scalar label")
    return value.item() if isinstance(value, np.generic) else value


def _same_label(first: Any, second: Any) -> bool:
    return bool(first == second)


def _build_prediction_grid(
    prediction_rows: Sequence[Mapping[str, Any]], method: str
) -> _PredictionGrid:
    grouped: dict[ItemKey, dict[int, Any]] = defaultdict(dict)
    gold: dict[ItemKey, Any] = {}
    for row in prediction_rows:
        if not isinstance(row, Mapping):
            raise TypeError("prediction rows must be mappings")
        if row.get("method") != method:
            continue
        item = (
            _text(row, "dataset"),
            _text(row, "video_id"),
            _text(row, "question_id"),
        )
        origin_id = _origin(row)
        prediction = _valid_label(row.get("prediction"), name="prediction")
        answer = _valid_label(row.get("gold"), name="gold")
        if origin_id in grouped[item]:
            raise ValueError(
                f"duplicate prediction row for {method}/{item}/origin={origin_id}"
            )
        grouped[item][origin_id] = prediction
        if item in gold and not _same_label(gold[item], answer):
            raise ValueError(f"gold answer changes across origins for {method}/{item}")
        gold[item] = answer
    if not grouped:
        raise ValueError(f"no prediction rows found for method {method!r}")
    item_keys = tuple(sorted(grouped))
    origin_sets = [tuple(sorted(grouped[item])) for item in item_keys]
    if any(origins != origin_sets[0] for origins in origin_sets[1:]):
        raise ValueError(f"method {method!r} does not have a rectangular origin grid")
    if len(origin_sets[0]) < 2:
        raise ValueError("prediction stability requires at least two origins")
    return _PredictionGrid(
        method=method,
        item_keys=item_keys,
        origin_ids=origin_sets[0],
        predictions={item: dict(values) for item, values in grouped.items()},
        gold=gold,
    )


def _prediction_category_rows(grid: _PredictionGrid) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in grid.item_keys:
        predictions = [grid.predictions[item][origin] for origin in grid.origin_ids]
        gold = grid.gold[item]
        correct = [_same_label(value, gold) for value in predictions]
        stable = all(_same_label(value, predictions[0]) for value in predictions[1:])
        prediction_pairs = tuple(combinations(range(len(predictions)), 2))
        pairwise_disagreement = (
            0.0
            if not prediction_pairs
            else float(
                np.mean(
                    [
                        not _same_label(predictions[left], predictions[right])
                        for left, right in prediction_pairs
                    ]
                )
            )
        )
        if stable and all(correct):
            category = "stable_correct"
        elif stable:
            category = "stable_wrong"
        else:
            category = "mixed"
        rows.append(
            {
                "dataset": item[0],
                "video_id": item[1],
                "question_id": item[2],
                "method": grid.method,
                "category": category,
                "stable": stable,
                "all_origins_correct": bool(all(correct)),
                "any_origin_correct": bool(any(correct)),
                "mean_accuracy": float(np.mean(correct)),
                "pairwise_answer_disagreement": pairwise_disagreement,
                "failure_rate": float(1.0 - np.mean(correct)),
            }
        )
    return rows


def categorize_prediction_stability(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Classify items as stable-correct, stable-wrong, or mixed by method."""

    if methods is None:
        selected_methods = tuple(
            sorted(
                {
                    _text(row, "method")
                    for row in prediction_rows
                    if isinstance(row, Mapping)
                }
            )
        )
    else:
        selected_methods = tuple(methods)
    if not selected_methods or len(set(selected_methods)) != len(selected_methods):
        raise ValueError("methods must be non-empty and unique")
    grids = [
        _build_prediction_grid(prediction_rows, method) for method in selected_methods
    ]
    reference = grids[0]
    for grid in grids[1:]:
        if grid.item_keys != reference.item_keys:
            raise ValueError("prediction method items do not align exactly")
        if grid.origin_ids != reference.origin_ids:
            raise ValueError("prediction method origin grids do not align exactly")
        for item in reference.item_keys:
            if not _same_label(grid.gold[item], reference.gold[item]):
                raise ValueError("prediction method gold answers do not align exactly")

    item_rows = [row for grid in grids for row in _prediction_category_rows(grid)]
    summaries: dict[str, Any] = {}
    grids_by_method = {grid.method: grid for grid in grids}
    for method in selected_methods:
        group = [row for row in item_rows if row["method"] == method]
        grid = grids_by_method[method]
        prediction_matrix = np.empty(
            (len(grid.item_keys), len(grid.origin_ids)), dtype=object
        )
        gold = np.empty(len(grid.item_keys), dtype=object)
        for item_index, item in enumerate(grid.item_keys):
            gold[item_index] = grid.gold[item]
            for origin_index, origin_id in enumerate(grid.origin_ids):
                prediction_matrix[item_index, origin_index] = grid.predictions[item][
                    origin_id
                ]
        counts = {
            category: sum(row["category"] == category for row in group)
            for category in ("stable_correct", "stable_wrong", "mixed")
        }
        summaries[method] = {
            "num_items": len(group),
            "counts": counts,
            "fractions": {
                category: float(count / len(group))
                for category, count in counts.items()
            },
            "mean_accuracy": float(np.mean([row["mean_accuracy"] for row in group])),
            "mllm_stability_metrics": mllm_stability_metrics(prediction_matrix, gold),
        }
    return _jsonable(
        {
            "category_order": ["stable_correct", "stable_wrong", "mixed"],
            "origin_ids": list(reference.origin_ids),
            "num_items": len(reference.item_keys),
            "methods": summaries,
            "item_rows": item_rows,
        }
    )


def evaluate_prediction_stability_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    treatment_method: str,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Categorize and jointly bootstrap two aligned VideoMME prediction arms."""

    if baseline_method == treatment_method:
        raise ValueError("baseline_method and treatment_method must differ")
    categories = categorize_prediction_stability(
        prediction_rows, methods=(baseline_method, treatment_method)
    )
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    for method in (baseline_method, treatment_method):
        by_method[method] = [
            row for row in categories["item_rows"] if row["method"] == method
        ]
    category_order = ("stable_correct", "stable_wrong", "mixed")

    def matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    *(
                        float(row["category"] == category)
                        for category in category_order
                    ),
                    float(row["mean_accuracy"]),
                    float(row["pairwise_answer_disagreement"]),
                    float(row["failure_rate"]),
                ]
                for row in rows
            ],
            dtype=float,
        )

    baseline_rows = by_method[baseline_method]
    treatment_rows = by_method[treatment_method]
    baseline = matrix(baseline_rows)
    treatment = matrix(treatment_rows)
    clusters = [f"{row['dataset']}\x1f{row['video_id']}" for row in baseline_rows]
    comparison = video_cluster_paired_bootstrap(
        clusters,
        baseline,
        treatment,
        statistic=lambda first, second: np.mean(second - first, axis=0),
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    comparison.update(
        {
            "effect_definition": "treatment - baseline",
            "effect_order": [
                "delta_stable_correct_fraction",
                "delta_stable_wrong_fraction",
                "delta_mixed_fraction",
                "delta_mean_accuracy",
                "delta_pairwise_answer_disagreement",
                "delta_failure_rate",
            ],
        }
    )
    return _jsonable(
        {
            **categories,
            "baseline_method": baseline_method,
            "treatment_method": treatment_method,
            "cluster_unit": "dataset/video_id",
            "comparison": comparison,
        }
    )


def evaluate_prediction_failure_calibration(
    trace_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
) -> dict[str, Any]:
    """Strictly join trace uncertainty to per-item VideoMME failure rate."""

    categories = categorize_prediction_stability(prediction_rows, methods=methods)
    category_index = {
        (row["method"], row["dataset"], row["video_id"], row["question_id"]): row
        for row in categories["item_rows"]
    }
    outer_primary_calibrations: dict[str, Any] = {}
    outer_calibrations: dict[str, Any] = {}
    inner_blocks: dict[str, Any] = {}
    for method in methods:
        trace_grid = _build_trace_grid(trace_rows, method)
        prediction_grid = _build_prediction_grid(prediction_rows, method)
        if trace_grid.item_keys != prediction_grid.item_keys:
            raise ValueError("trace and prediction items do not align exactly")
        if trace_grid.origin_ids != prediction_grid.origin_ids:
            raise ValueError("trace and prediction origin grids do not align exactly")
        metrics = _trace_item_metrics(trace_grid)
        failure = [
            float(
                category_index[
                    (method, row["dataset"], row["video_id"], row["question_id"])
                ]["failure_rate"]
            )
            for row in metrics
        ]
        primary_calibration = phase_uncertainty_failure_calibration(
            [float(row[_PRIMARY_OUTER_UNCERTAINTY_METRIC]) for row in metrics],
            failure,
        )
        primary_calibration["failure_definition"] = (
            "fraction of origins answered incorrectly"
        )
        outer_primary_calibrations[method] = primary_calibration
        calibration = phase_uncertainty_failure_calibration(
            [float(row["outer_origin_selected_set_uncertainty"]) for row in metrics],
            failure,
        )
        calibration["failure_definition"] = "fraction of origins answered incorrectly"
        outer_calibrations[method] = calibration
        inner_rows, inner_fields, inner_unavailable_reason = _optional_inner_phase_rows(
            trace_grid
        )
        if inner_rows is None:
            inner_blocks[method] = _inner_method_block(
                None,
                (),
                failure_rows=None,
                unavailable_reason=inner_unavailable_reason,
            )
        else:
            inner_block = _inner_method_block(
                inner_rows, inner_fields, failure_rows=None
            )
            inner_calibration = phase_uncertainty_failure_calibration(
                _inner_probability(
                    [
                        float(row["selected_phase_uncertainty_mean"])
                        for row in inner_rows
                    ]
                ),
                failure,
            )
            inner_calibration.update(
                {
                    "status": "available",
                    "raw_uncertainty_field": "selected_phase_uncertainty_mean",
                    "probability_mapping": "u / (1 + u), fixed and unfitted",
                    "failure_definition": "fraction of origins answered incorrectly",
                }
            )
            inner_block["failure_calibration"] = inner_calibration
            inner_blocks[method] = inner_block
    return _jsonable(
        {
            "cluster_key": "dataset/video_id",
            "categories": categories,
            "outer_origin_selector_failure_calibration": (outer_primary_calibrations),
            # Schema-v1 compatibility: exact-source Jaccard uncertainty.
            "outer_origin_selected_set_failure_calibration": outer_calibrations,
            "inner_phase_uncertainty": {
                "definition": (
                    "raw scaled-MAD disagreement among aligned interleaved inner "
                    "phases; distinct from outer-origin selected-set uncertainty"
                ),
                "methods": inner_blocks,
            },
        }
    )


__all__ = [
    "categorize_prediction_stability",
    "evaluate_phasefuse_analysis",
    "evaluate_prediction_failure_calibration",
    "evaluate_prediction_stability_rows",
    "phase_uncertainty_failure_calibration",
    "selected_set_consistency",
]
