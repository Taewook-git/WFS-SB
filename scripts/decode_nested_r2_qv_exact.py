#!/usr/bin/env python3
"""Fresh exact-decode the frozen paired QV nested-R2 decision grid."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

from phase_stable.artifacts import write_jsonl
from phase_stable.canonical_resample import (
    CanonicalRequestPair,
    decode_canonical_request_batch,
)
from scripts.decode_rc12_exact import _arm, _logical_key, _request_id

METHODS = ("canonical_uniform", "phasefuse_nested_r2")


def read_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decisions = read_rows(args.decisions)
    cells = defaultdict(dict)
    for row in decisions:
        key = _logical_key(row)
        method = str(row["method"])
        if method not in METHODS or method in cells[key]:
            raise ValueError(f"invalid/duplicate QV decision arm: {key}/{method}")
        cells[key][method] = row
    if len(cells) != 500 or any(set(arms) != set(METHODS) for arms in cells.values()):
        raise ValueError("QV decision grid must be exact 100 x 5 paired cells")
    pairs_by_video = defaultdict(list)
    context = {}
    for key, arms in sorted(cells.items()):
        uniform = arms["canonical_uniform"]
        treatment = arms["phasefuse_nested_r2"]
        fields = (
            "video_path",
            "origin_sec",
            "support_start_sec",
            "support_stop_sec",
            "lattice_timestamps_sec",
        )
        if any(uniform.get(name) != treatment.get(name) for name in fields):
            raise ValueError(f"paired QV decision contract mismatch: {key}")
        request_id = _request_id(key)
        pair = CanonicalRequestPair(
            request_id=request_id,
            rc12=_arm(treatment),
            canonical_uniform=_arm(uniform),
        )
        video_path = str(Path(str(treatment["video_path"])).resolve())
        pairs_by_video[video_path].append(pair)
        context[request_id] = {
            "key": key,
            "support": [
                float(treatment["support_start_sec"]),
                float(treatment["support_stop_sec"]),
            ],
            "record_metadata": dict(treatment.get("record_metadata", {})),
            "decisions": {
                method: dict(arms[method].get("decision_metadata", {}))
                for method in METHODS
            },
        }
    if len(pairs_by_video) != 100 or any(
        len(pairs) != 5 for pairs in pairs_by_video.values()
    ):
        raise ValueError("QV exact decode requires five paired origins per source clip")

    traces = []
    decode_passes = {}
    for video_number, (video_path, pairs) in enumerate(
        sorted(pairs_by_video.items()), 1
    ):
        print(f"Exact decode progress: video {video_number}/100", flush=True)
        result = decode_canonical_request_batch(video_path, pairs)
        decode_passes[video_path] = [item.to_dict() for item in result.decode_passes]
        for pair in sorted(pairs, key=lambda item: item.request_id):
            item = context[pair.request_id]
            dataset, video_id, question_id, origin_id = item["key"]
            decoded = result.request(pair.request_id)
            for method in METHODS:
                metadata = {
                    **item["record_metadata"],
                    "canonical_support_sec": item["support"],
                    "nested_r2_decision": item["decisions"][method],
                }
                traces.append(
                    decoded.trace_row(
                        method,
                        dataset=dataset,
                        video_id=video_id,
                        question_id=question_id,
                        origin_id=origin_id,
                        origin_sec=float(cells[item["key"]][method]["origin_sec"]),
                        record_metadata=metadata,
                    )
                )
    traces.sort(key=_logical_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "traces.jsonl"
    write_jsonl(trace_path, traces)
    pass_counts = [len(value) for value in decode_passes.values()]
    union_sizes = [
        len(value[0]["target_timestamps_sec"]) for value in decode_passes.values()
    ]
    summary = {
        "schema_version": 1,
        "methods": list(METHODS),
        "num_rows": len(traces),
        "num_videos": 100,
        "fresh_source_decode": True,
        "prohibit_scout_frame_remap": True,
        "video_decode_passes": decode_passes,
        "decode_cost": {
            "primary_union_size_min": min(union_sizes),
            "primary_union_size_median": statistics.median(union_sizes),
            "primary_union_size_mean": statistics.fmean(union_sizes),
            "primary_union_size_max": max(union_sizes),
            "decode_pass_count_min": min(pass_counts),
            "decode_pass_count_median": statistics.median(pass_counts),
            "decode_pass_count_mean": statistics.fmean(pass_counts),
            "decode_pass_count_max": max(pass_counts),
            "decoded_target_count_total": sum(
                len(item["target_timestamps_sec"])
                for passes in decode_passes.values()
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
        },
    }
    temporary = args.output_dir / "decode_summary.json.tmp"
    temporary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output_dir / "decode_summary.json")
    print(f"Wrote {len(traces)} fresh exact QV trace rows")


if __name__ == "__main__":
    main()
