#!/usr/bin/env python3
"""Run the frozen hierarchical QV Stage-0 selector analysis after label join."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from phase_stable.metrics import video_cluster_paired_bootstrap

METHODS = ("canonical_uniform", "phasefuse_nested_r2")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260813
METRICS = (
    "selected_relevant_fraction",
    "relevant_clip_recall",
    "mean_selected_saliency_vote",
    "mean_gt_clip_nearest_selected_sec",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fidelity(times: list[float], label: Mapping[str, Any]) -> dict[str, float]:
    selected = np.asarray(times, dtype=float)
    windows = [(float(row[0]), float(row[1])) for row in label["relevant_windows"]]
    clip_ids = np.asarray(label["relevant_clip_ids"], dtype=int)
    votes = label["saliency_scores"]
    relevant = np.asarray(
        [any(start <= time < stop for start, stop in windows) for time in selected],
        dtype=float,
    )
    selected_clips = np.floor(selected / 2.0).astype(int)
    hit = set(selected_clips.tolist())
    vote_by_clip = {
        int(clip): float(np.mean(np.asarray(raw_votes, dtype=float)))
        for clip, raw_votes in zip(clip_ids.tolist(), votes)
    }
    centers = 2.0 * clip_ids.astype(float) + 1.0
    nearest = np.min(np.abs(centers[:, None] - selected[None, :]), axis=1)
    return {
        "selected_relevant_fraction": float(np.mean(relevant)),
        "relevant_clip_recall": float(
            sum(int(clip) in hit for clip in clip_ids) / len(clip_ids)
        ),
        "mean_selected_saliency_vote": float(
            np.mean([vote_by_clip.get(int(clip), 0.0) for clip in selected_clips])
        ),
        "mean_gt_clip_nearest_selected_sec": float(np.mean(nearest)),
    }


def ordered_endpoint_gates(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    """Apply the frozen one-sided hierarchy without changing Stage-0 scope."""

    primary = dict(comparisons["selected_relevant_fraction"])
    clip = dict(comparisons["relevant_clip_recall"])
    primary_pass = float(primary["ci_low"]) > 0.0
    return primary_pass, {
        "selected_relevant_fraction": {
            **primary,
            "gate": "ci_low > 0",
            "passed": primary_pass,
        },
        "relevant_clip_recall": {
            **clip,
            "hierarchically_tested": primary_pass,
            "passed_if_tested": (
                bool(float(clip["ci_low"]) > 0.0) if primary_pass else None
            ),
            "descriptive_only": not primary_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--label-join-manifest", type=Path, required=True)
    parser.add_argument("--blind-seal", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.blind_seal.read_text(encoding="utf-8"))
    joined = json.loads(args.label_join_manifest.read_text(encoding="utf-8"))
    if seal.get("status") != "sealed_before_label_join":
        raise ValueError("analysis requires a valid pre-label blind seal")
    if joined.get("status") != "labels_joined_after_blind_seal":
        raise ValueError("analysis requires a post-seal label join")
    if joined.get("blind_seal_sha256") != sha256_file(args.blind_seal):
        raise ValueError("label join does not bind the current blind seal")
    if joined.get("output_labels_sha256") != sha256_file(args.labels):
        raise ValueError("label join does not bind the current label rows")
    if seal.get("decisions_sha256") != sha256_file(args.decisions):
        raise ValueError("blind seal/decision hash drift")
    if seal.get("traces_sha256") != sha256_file(args.traces):
        raise ValueError("blind seal/trace hash drift")

    labels = read_rows(args.labels)
    label_map = {(str(row["vid"]), str(row["qid"])): row for row in labels}
    if len(labels) != 100 or len(label_map) != 100:
        raise ValueError("label grid is not exact 100-source cohort")
    traces = read_rows(args.traces)
    decisions = read_rows(args.decisions)
    decision_map = {
        (
            str(row["video_id"]),
            str(row["question_id"]),
            int(row["origin_id"]),
            str(row["method"]),
        ): row
        for row in decisions
    }
    trace_keys = {
        (
            str(row["video_id"]),
            str(row["question_id"]),
            int(row["origin_id"]),
            str(row["method"]),
        )
        for row in traces
    }
    expected_keys = {
        (video_id, question_id, origin, method)
        for video_id, question_id in label_map
        for origin in range(5)
        for method in METHODS
    }
    if (
        len(decisions) != 1000
        or len(decision_map) != 1000
        or len(traces) != 1000
        or len(trace_keys) != 1000
        or set(decision_map) != expected_keys
        or trace_keys != expected_keys
    ):
        raise ValueError("standalone analysis grid is not exact 100 x 5 x 2")
    rows = []
    by_cell = defaultdict(dict)
    for trace in traces:
        method = str(trace["method"])
        key = (
            str(trace["video_id"]),
            str(trace["question_id"]),
            int(trace["origin_id"]),
        )
        if method not in METHODS or method in by_cell[key]:
            raise ValueError(f"invalid duplicate analysis arm: {key}/{method}")
        label = label_map[key[:2]]
        metric = fidelity(trace["selected_actual_pts_sec"], label)
        decision = decision_map[(*key, method)]
        relocation_count = (
            int(decision.get("decision_metadata", {}).get("relocation_count", 0))
            if method == "phasefuse_nested_r2"
            else 0
        )
        row = {
            "source_id": str(label["source_id"]),
            "video_id": key[0],
            "question_id": key[1],
            "origin_id": key[2],
            "method": method,
            "relocation_count": relocation_count,
            **metric,
        }
        rows.append(row)
        by_cell[key][method] = row
    if len(by_cell) != 500 or any(
        set(arms) != set(METHODS) for arms in by_cell.values()
    ):
        raise ValueError("analysis grid is not exact 100 x 5 paired cells")
    rows.sort(key=lambda row: (row["source_id"], row["origin_id"], row["method"]))
    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    temporary_rows = args.output_rows.with_suffix(args.output_rows.suffix + ".tmp")
    with temporary_rows.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary_rows, args.output_rows)

    ordered_cells = sorted(by_cell)
    clusters = np.asarray(
        [by_cell[key]["canonical_uniform"]["source_id"] for key in ordered_cells],
        dtype=object,
    )
    comparisons = {}
    for metric in METRICS:
        baseline = np.asarray(
            [by_cell[key]["canonical_uniform"][metric] for key in ordered_cells],
            dtype=float,
        )
        treatment = np.asarray(
            [by_cell[key]["phasefuse_nested_r2"][metric] for key in ordered_cells],
            dtype=float,
        )
        comparisons[metric] = video_cluster_paired_bootstrap(
            clusters,
            baseline,
            treatment,
            n_bootstrap=BOOTSTRAP_RESAMPLES,
            confidence=0.95,
            seed=BOOTSTRAP_SEED,
        )
    primary_pass, ordered_endpoints = ordered_endpoint_gates(comparisons)
    origin_effects = {}
    for origin in range(5):
        active = [key for key in ordered_cells if key[2] == origin]
        origin_effects[str(origin)] = {
            metric: float(
                np.mean(
                    [
                        by_cell[key]["phasefuse_nested_r2"][metric]
                        - by_cell[key]["canonical_uniform"][metric]
                        for key in active
                    ]
                )
            )
            for metric in METRICS
        }
    relocation_counts = [
        int(by_cell[key]["phasefuse_nested_r2"]["relocation_count"])
        for key in ordered_cells
    ]
    summary = {
        "schema_version": 1,
        "status": "pass" if primary_pass else "fail",
        "effect_definition": "phasefuse_nested_r2 - canonical_uniform",
        "num_sources": 100,
        "num_paired_cells": 500,
        "bootstrap": {
            "cluster": "source_id",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "ordered_endpoints": ordered_endpoints,
        "secondary": {
            metric: comparisons[metric]
            for metric in (
                "mean_selected_saliency_vote",
                "mean_gt_clip_nearest_selected_sec",
            )
        },
        "relocation": {
            "mean_per_cell": float(np.mean(relocation_counts)),
            "rate_any": float(np.mean(np.asarray(relocation_counts) > 0)),
            "count_0": int(sum(value == 0 for value in relocation_counts)),
            "count_1": int(sum(value == 1 for value in relocation_counts)),
            "count_2": int(sum(value == 2 for value in relocation_counts)),
        },
        "origin_effects": origin_effects,
        "blind_seal_sha256": sha256_file(args.blind_seal),
        "label_join_manifest_sha256": sha256_file(args.label_join_manifest),
        "qwen_launch_authorized": False,
        "qwen_reason": "current authorization is QV Stage0 selector analysis only",
    }
    temporary = args.output_summary.with_suffix(args.output_summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output_summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
