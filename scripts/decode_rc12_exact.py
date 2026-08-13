#!/usr/bin/env python3
"""Fresh-decode frozen RC12 decisions in video-level primary-union batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from phase_stable.canonical_resample import (
    CanonicalArmRequest,
    CanonicalRequestPair,
    decode_canonical_request_batch,
)

from phase_stable.artifacts import write_jsonl
from phase_stable.repro import write_reproducibility_manifests

METHODS = ("canonical_uniform", "phasefuse_rc12")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"decision row {line_number} is not an object")
        rows.append(value)
    return rows


def _logical_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["dataset"]),
        str(row["video_id"]),
        str(row["question_id"]),
        int(row["origin_id"]),
    )


def _request_id(key: tuple[str, str, str, int]) -> str:
    logical = "\x1f".join((*key[:3], str(key[3])))
    return hashlib.sha256(logical.encode("utf-8")).hexdigest()


def _int_list(row: Mapping[str, Any], name: str) -> list[int]:
    values = row.get(name)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"{name} must be a list of integers")
    return [int(value) for value in values]


def _float_tuple(row: Mapping[str, Any], name: str) -> tuple[float, ...]:
    values = row.get(name)
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a list")
    return tuple(float(value) for value in values)


def _arm(row: Mapping[str, Any]) -> CanonicalArmRequest:
    method = str(row["method"])
    lattice = _float_tuple(row, "lattice_timestamps_sec")
    targets = _float_tuple(row, "target_timestamps_sec")
    scores = _float_tuple(row, "quantized_scores")
    indices = _int_list(row, "target_indices")
    if len(indices) != len(targets):
        raise ValueError("target_indices and target_timestamps_sec do not align")
    if any(index < 0 or index >= len(lattice) for index in indices):
        raise ValueError("target index is outside the canonical lattice")
    if tuple(lattice[index] for index in indices) != targets:
        raise ValueError("target timestamps are not exact indexed lattice ticks")

    if method == "canonical_uniform":
        role = "canonical_uniform"
        primary_sources = ("canonical_uniform_anchor",) * len(targets)
        anchor_indices = set(indices)
    elif method == "phasefuse_rc12":
        role = "rc12"
        decision = row.get("decision_metadata")
        if not isinstance(decision, Mapping):
            raise ValueError("RC12 decision_metadata is missing")
        anchors = set(_int_list(decision, "anchor_indices"))
        residuals = set(_int_list(decision, "residual_indices"))
        if anchors | residuals != set(indices) or anchors & residuals:
            raise ValueError("RC12 anchor/residual provenance does not match targets")
        primary_sources = tuple(
            "rc12_anchor" if index in anchors else "rc12_query_residual"
            for index in indices
        )
        anchor_indices = anchors
    else:
        raise ValueError(f"unexpected decision method {method!r}")

    candidate_sources = tuple(
        (
            "canonical_uniform_anchor_candidate"
            if role == "canonical_uniform"
            else "rc12_anchor_candidate"
            if index in anchor_indices
            else "rc12_query_residual_candidate"
        )
        for index in range(len(lattice))
    )
    return CanonicalArmRequest(
        method=method,
        role=role,
        primary_target_timestamps_sec=targets,
        lattice_timestamps_sec=lattice,
        candidate_scores=scores,
        primary_sources=primary_sources,
        primary_ranks=tuple(range(len(targets))),
        candidate_sources=candidate_sources,
        candidate_ranks=tuple(range(len(lattice))),
    )


def build_request_pairs(
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, list[CanonicalRequestPair]], dict[str, dict[str, Any]]]:
    cells: dict[tuple[str, str, str, int], dict[str, Mapping[str, Any]]] = defaultdict(
        dict
    )
    for row in decisions:
        key = _logical_key(row)
        method = str(row.get("method"))
        if method not in METHODS or method in cells[key]:
            raise ValueError(f"invalid or duplicate method for decision cell {key}")
        cells[key][method] = row

    pairs_by_video: dict[str, list[CanonicalRequestPair]] = defaultdict(list)
    video_contracts: dict[str, tuple[str, float, float, tuple[float, ...]]] = {}
    metadata_by_request: dict[str, dict[str, Any]] = {}
    for key, arms in sorted(cells.items()):
        if set(arms) != set(METHODS):
            raise ValueError(f"decision cell does not contain both frozen arms: {key}")
        uniform = arms["canonical_uniform"]
        rc12 = arms["phasefuse_rc12"]
        contract_fields = (
            "video_path",
            "origin_sec",
            "support_start_sec",
            "support_stop_sec",
            "lattice_timestamps_sec",
        )
        if any(uniform.get(name) != rc12.get(name) for name in contract_fields):
            raise ValueError(f"paired decision contract mismatch: {key}")
        request_id = _request_id(key)
        pair = CanonicalRequestPair(
            request_id=request_id,
            rc12=_arm(rc12),
            canonical_uniform=_arm(uniform),
        )
        video_path = str(Path(str(rc12["video_path"])).resolve())
        video_contract = (
            key[1],
            float(rc12["support_start_sec"]),
            float(rc12["support_stop_sec"]),
            tuple(float(value) for value in rc12["lattice_timestamps_sec"]),
        )
        prior_contract = video_contracts.setdefault(video_path, video_contract)
        if prior_contract != video_contract:
            raise ValueError(
                "one resolved video path maps to multiple video IDs or canonical contracts"
            )
        pairs_by_video[video_path].append(pair)
        metadata_by_request[request_id] = {
            "dataset": key[0],
            "video_id": key[1],
            "question_id": key[2],
            "origin_id": key[3],
            "origin_sec": float(rc12["origin_sec"]),
            "record_metadata": dict(rc12.get("record_metadata", {})),
            "support_start_sec": float(rc12["support_start_sec"]),
            "support_stop_sec": float(rc12["support_stop_sec"]),
            "decision_metadata": {
                method: dict(arms[method].get("decision_metadata", {}))
                for method in METHODS
            },
        }
    for video_path, pairs in pairs_by_video.items():
        contexts = [metadata_by_request[pair.request_id] for pair in pairs]
        if len(pairs) != 15:
            raise ValueError(
                f"dev20 video batch must contain exactly 15 pairs: {video_path}"
            )
        if len({context["question_id"] for context in contexts}) != 3:
            raise ValueError(
                f"dev20 video batch must contain exactly three questions: {video_path}"
            )
        if {context["origin_id"] for context in contexts} != set(range(5)):
            raise ValueError(
                f"dev20 video batch must contain exact five origins: {video_path}"
            )
    return dict(pairs_by_video), metadata_by_request


def decode_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs_by_video, metadata = build_request_pairs(decisions)
    traces: list[dict[str, Any]] = []
    video_passes: dict[str, list[dict[str, Any]]] = {}
    for video_path, pairs in sorted(pairs_by_video.items()):
        result = decode_canonical_request_batch(video_path, pairs)
        video_passes[video_path] = [item.to_dict() for item in result.decode_passes]
        for pair in sorted(pairs, key=lambda item: item.request_id):
            context = metadata[pair.request_id]
            decoded = result.request(pair.request_id)
            for method in METHODS:
                record_metadata = {
                    **context["record_metadata"],
                    "canonical_support_sec": [
                        context["support_start_sec"],
                        context["support_stop_sec"],
                    ],
                    "rc12_decision": context["decision_metadata"][method],
                }
                traces.append(
                    decoded.trace_row(
                        method,
                        dataset=context["dataset"],
                        video_id=context["video_id"],
                        question_id=context["question_id"],
                        origin_id=context["origin_id"],
                        origin_sec=context["origin_sec"],
                        record_metadata=record_metadata,
                    )
                )
    traces.sort(
        key=lambda row: (
            row["dataset"],
            row["video_id"],
            row["question_id"],
            int(row["origin_id"]),
            row["method"],
        )
    )
    summary = {
        "schema_version": 1,
        "methods": list(METHODS),
        "num_rows": len(traces),
        "num_videos": len(pairs_by_video),
        "video_decode_passes": video_passes,
        "fresh_source_decode": True,
        "prohibit_scout_frame_remap": True,
    }
    primary_union_sizes = [
        len(passes[0]["target_timestamps_sec"]) for passes in video_passes.values()
    ]
    pass_counts = [len(passes) for passes in video_passes.values()]
    summary["decode_cost"] = {
        "primary_union_size": {
            "min": min(primary_union_sizes),
            "median": statistics.median(primary_union_sizes),
            "mean": statistics.fmean(primary_union_sizes),
            "max": max(primary_union_sizes),
        },
        "decode_pass_count": {
            "min": min(pass_counts),
            "median": statistics.median(pass_counts),
            "mean": statistics.fmean(pass_counts),
            "max": max(pass_counts),
        },
        "decoded_target_count_total": sum(
            len(item["target_timestamps_sec"])
            for passes in video_passes.values()
            for item in passes
        ),
        "repair_attempt_count": sum(
            row["canonical_decode"]["repair_attempt_count"] for row in traces
        ),
        "duplicate_rejection_count": sum(
            row["canonical_decode"]["duplicate_rejection_count"] for row in traces
        ),
        "distance_relaxation_count": sum(
            row["canonical_decode"]["distance_relaxation_count"] for row in traces
        ),
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decisions = _read_rows(args.decisions)
    traces, summary = decode_decisions(decisions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "traces.jsonl"
    write_jsonl(trace_path, traces)
    summary_path = args.output_dir / "decode_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, summary_path)
    write_reproducibility_manifests(
        args.output_dir,
        command="decode-rc12-exact",
        config={"methods": list(METHODS), "frame_budget": 16},
        input_paths=(args.decisions,),
        extra={"traces": str(trace_path.resolve())},
        repo_root=Path(__file__).parents[1],
    )
    print(f"Wrote {len(traces)} fresh canonical trace rows to {trace_path}")


if __name__ == "__main__":
    main()
