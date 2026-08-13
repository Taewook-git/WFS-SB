#!/usr/bin/env python3
"""Fail-closed validation for the RC14 exact-decode and reuse contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import av
import numpy as np

FROZEN_CONFIG = {
    "frame_budget": 16,
    "anchor_count": 14,
    "max_lattice_hz": 2.0,
    "smoothing_sigma_sec": 4.0,
    "score_quantum": 0.1,
    "min_residual_distance_sec": 2.0,
    "coverage_tiebreak_cap_sec": 8.0,
    "lattice_origin_sec": 0.0,
}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row["dataset"]),
        str(row["video_id"]),
        str(row["question_id"]),
        int(row["origin_id"]),
        str(row["method"]),
    )


def rgb_sha(frame) -> str:
    array = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def validate_sources(bundle_path: Path, decisions: list[dict]) -> dict:
    members = read_rows(bundle_path)
    if len(members) != 20:
        raise SystemExit("source bundle is not dev20")
    paths = set()
    for row in members:
        path = Path(row["path"]).resolve(strict=True)
        if path in paths or path.stat().st_size != int(row["size_bytes"]):
            raise SystemExit(f"source member size/path drift: {path}")
        if sha(path) != row["sha256"]:
            raise SystemExit(f"source member SHA drift: {path}")
        paths.add(path)
    decision_paths = {Path(row["video_path"]).resolve() for row in decisions}
    if decision_paths != paths:
        raise SystemExit("decision video paths do not exactly bind the source bundle")
    return {"num_members": 20, "sha256": sha(bundle_path)}


def validate_decisions(decisions: list[dict], reference: list[dict]) -> dict:
    decision_map = {key(row): row for row in decisions}
    if len(decisions) != 600 or len(decision_map) != 600:
        raise SystemExit("RC14 decisions are not a unique 60x5x2 grid")
    methods = {row["method"] for row in decisions}
    if methods != {"canonical_uniform", "phasefuse_rc14"}:
        raise SystemExit("RC14 decision method grid drift")
    reference_uniform = {
        key(row)[:-1]: row
        for row in reference
        if row.get("method") == "canonical_uniform"
    }
    new_uniform = {
        key(row)[:-1]: row
        for row in decisions
        if row["method"] == "canonical_uniform"
    }
    if len(reference_uniform) != 300 or new_uniform != reference_uniform:
        raise SystemExit("canonical-uniform decision payload is not exact RC12 reuse")
    by_video = defaultdict(list)
    for row in decisions:
        by_video[str(row["video_id"])].append(row)
    if len(by_video) != 20:
        raise SystemExit("RC14 decision cohort is not 20 videos")
    for video_id, rows in by_video.items():
        if len(rows) != 30:
            raise SystemExit(f"non-rectangular RC14 video grid: {video_id}")
        lattices = {tuple(row["lattice_timestamps_sec"]) for row in rows}
        supports = {
            (float(row["support_start_sec"]), float(row["support_stop_sec"]))
            for row in rows
        }
        if len(lattices) != 1 or len(supports) != 1:
            raise SystemExit(f"lattice/support drift: {video_id}")
        lattice = list(next(iter(lattices)))
        start, stop = next(iter(supports))
        expected = (
            0.5
            * np.arange(
                int(np.ceil((start - 1e-9) / 0.5)),
                int(np.floor((stop + 1e-9) / 0.5)) + 1,
            )
        ).tolist()
        if lattice != expected:
            raise SystemExit(f"non-canonical origin-zero lattice: {video_id}")
        for row in rows:
            if row["method"] != "phasefuse_rc14":
                continue
            metadata = row["decision_metadata"]
            anchors = metadata.get("anchor_indices", [])
            residuals = metadata.get("residual_indices", [])
            if (
                metadata.get("method") != "rc14"
                or metadata.get("config") != FROZEN_CONFIG
                or metadata.get("decision_signal")
                != "smoothed_query_relevance_only"
                or metadata.get("ti_dwt_role") != "diagnostic_only"
                or len(anchors) != 14
                or len(residuals) != 2
                or set(anchors) & set(residuals)
                or sorted(anchors + residuals) != row["target_indices"]
                or len(row["target_indices"]) != 16
            ):
                raise SystemExit(f"frozen RC14 decision drift: {key(row)}")
    return {
        "num_rows": 600,
        "num_uniform_rows_exactly_reused": 300,
        "config": FROZEN_CONFIG,
    }


def validate_traces(
    traces: list[dict], decisions: list[dict], decode_summary: dict
) -> tuple[dict, dict[str, dict]]:
    treatment_decisions = {
        key(row): row for row in decisions if row["method"] == "phasefuse_rc14"
    }
    trace_map = {key(row): row for row in traces}
    if len(traces) != 300 or len(trace_map) != 300 or set(trace_map) != set(
        treatment_decisions
    ):
        raise SystemExit("RC14 trace grid does not exactly match 300 decisions")
    proof = decode_summary.get("canonical_uniform_reuse_proof", {})
    if proof.get("status") != "exact_match" or proof.get("num_rows") != 300:
        raise SystemExit("fresh canonical-uniform payload reuse proof is absent")
    by_video = defaultdict(list)
    repair_attempts = duplicate_rejections = relaxations = 0
    for logical_key, trace in trace_map.items():
        decision = treatment_decisions[logical_key]
        selected = trace.get("selected_source_frame_indices")
        hashes = trace.get("selected_pixel_hashes")
        targets = trace.get("selected_timestamps_sec")
        actual = trace.get("selected_actual_pts_sec")
        if (
            not isinstance(selected, list)
            or len(selected) != 16
            or len(set(selected)) != 16
            or selected != sorted(selected)
            or not isinstance(hashes, list)
            or len(hashes) != 16
            or not isinstance(targets, list)
            or len(targets) != 16
            or not isinstance(actual, list)
            or len(actual) != 16
        ):
            raise SystemExit(f"invalid RC14 exact source budget: {logical_key}")
        decode = trace.get("canonical_decode", {})
        if (
            decode.get("fresh_source_decode") is not True
            or decode.get("prohibit_scout_frame_remap") is not True
            or decode.get("decoder_backend") != "pyav_sequential_nearest_pts"
            or decode.get("midpoint_tie_policy") != "earlier_pts"
        ):
            raise SystemExit(f"fresh decode provenance drift: {logical_key}")
        primary = [
            attempt.get("target_sec")
            for attempt in decode.get("attempts", [])
            if attempt.get("stage") == "primary"
        ]
        if primary != decision["target_timestamps_sec"]:
            raise SystemExit(f"primary targets do not reproduce decision: {logical_key}")
        lattice = decision["lattice_timestamps_sec"]
        for attempt in decode.get("attempts", []):
            index = attempt.get("canonical_index")
            if not isinstance(index, int) or attempt.get("target_sec") != lattice[index]:
                raise SystemExit(f"decode attempt escaped canonical lattice: {logical_key}")
        repair_attempts += int(decode.get("repair_attempt_count", 0))
        duplicate_rejections += int(decode.get("duplicate_rejection_count", 0))
        relaxations += int(decode.get("distance_relaxation_count", 0))
        video_path = str(Path(decision["video_path"]).resolve())
        by_video[video_path].append((selected, hashes))
    if len(by_video) != 20:
        raise SystemExit("RC14 traces are not bound to exactly 20 source videos")
    return (
        {
            "num_rows": 300,
            "frame_budget": 16,
            "repair_attempt_count": repair_attempts,
            "duplicate_rejection_count": duplicate_rejections,
            "distance_relaxation_count": relaxations,
            "canonical_uniform_reuse_proof": proof,
        },
        by_video,
    )


def audit_pixels(by_video: dict[str, list[tuple[list[int], list[str]]]]) -> dict:
    logical = unique = 0
    for video_path, entries in sorted(by_video.items()):
        wanted = {index for indices, _ in entries for index in indices}
        observed = {}
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for source_index, frame in enumerate(container.decode(stream)):
                if source_index in wanted:
                    observed[source_index] = rgb_sha(frame)
                    if len(observed) == len(wanted):
                        break
        if set(observed) != wanted:
            raise SystemExit(f"RC14 Qwen frame-index roundtrip missing: {video_path}")
        for indices, hashes in entries:
            if [observed[index] for index in indices] != hashes:
                raise SystemExit(f"RC14 Qwen pixel hash mismatch: {video_path}")
            logical += 16
        unique += len(wanted)
    return {
        "status": "validated",
        "logical_selected_hash_comparisons": logical,
        "unique_source_frames_reopened": unique,
        "num_video_decode_passes": len(by_video),
    }


def audit_keyframes(directory: Path, trace_map: dict, reference: Path) -> dict:
    names = [f"videomme_phasefuse_rc14_origin{origin:02d}.json" for origin in range(5)]
    if sorted(path.name for path in directory.glob("*.json")) != names:
        raise SystemExit("RC14 keyframe grid is not exact five cells")
    bundle = hashlib.sha256()
    rows = 0
    for name in names:
        path = directory / name
        bundle.update(name.encode()); bundle.update(b"\0")
        bundle.update(path.read_bytes()); bundle.update(b"\0")
        origin = int(name.removesuffix(".json").rsplit("origin", 1)[1])
        payload = json.loads(path.read_text())
        if len(payload) != 60:
            raise SystemExit(f"RC14 keyframe cell is not 60 rows: {name}")
        for item in payload:
            logical_key = (
                "videomme",
                str(item["video_id"]),
                str(item["question_id"]),
                origin,
                "phasefuse_rc14",
            )
            if item.get("keyframe_indices") != trace_map[logical_key].get(
                "selected_source_frame_indices"
            ):
                raise SystemExit(f"RC14 keyframe/trace mismatch: {logical_key}")
            rows += 1
    uniform_files = [
        reference / f"videomme_canonical_uniform_origin{origin:02d}.json"
        for origin in range(5)
    ]
    uniform_hashes = [sha(path) for path in uniform_files]
    if any(not path.is_file() for path in uniform_files) or len(set(uniform_hashes)) != 1:
        raise SystemExit("reference canonical-uniform keyframe payloads are not identical")
    return {
        "num_files": 5,
        "num_rows": rows,
        "filenames": names,
        "sha256_name_nul_bytes_nul": bundle.hexdigest(),
        "trace_indices_exact_match": True,
        "reference_uniform_payload_sha256": uniform_hashes[0],
        "reference_uniform_filenames": [path.name for path in uniform_files],
    }


def validate_reference_predictions(path: Path) -> dict:
    rows = [row for row in read_rows(path) if row.get("method") == "canonical_uniform"]
    identities = {key(row) for row in rows}
    if len(rows) != 300 or len(identities) != 300:
        raise SystemExit("reference canonical-uniform predictions are not exact 60x5")
    return {"num_rows": 300, "source_predictions_sha256": sha(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--decode-summary", type=Path, required=True)
    parser.add_argument("--keyframes", type=Path, required=True)
    parser.add_argument("--source-video-bundle", type=Path, required=True)
    parser.add_argument("--reference-decisions", type=Path, required=True)
    parser.add_argument("--reference-keyframes", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = read_rows(args.decisions)
    reference_decisions = read_rows(args.reference_decisions)
    traces = read_rows(args.traces)
    decode_summary = json.loads(args.decode_summary.read_text())
    decision_audit = validate_decisions(decisions, reference_decisions)
    source_audit = validate_sources(args.source_video_bundle, decisions)
    trace_audit, by_video = validate_traces(traces, decisions, decode_summary)
    trace_map = {key(row): row for row in traces}
    payload = {
        "schema_version": 1,
        "status": "validated",
        "method": "phasefuse_rc14",
        "num_rows": 300,
        "fresh_source_decode": True,
        "decisions_sha256": sha(args.decisions),
        "traces_sha256": sha(args.traces),
        "decode_summary_sha256": sha(args.decode_summary),
        "source_video_bundle": source_audit,
        "decision_audit": decision_audit,
        "trace_audit": trace_audit,
        "qwen_frame_index_pixel_hash_roundtrip": audit_pixels(by_video),
        "keyframe_bundle": audit_keyframes(
            args.keyframes, trace_map, args.reference_keyframes
        ),
        "canonical_uniform_prediction_reuse": validate_reference_predictions(
            args.reference_predictions
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
