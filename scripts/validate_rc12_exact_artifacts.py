#!/usr/bin/env python3
"""Strict validation for the two-stage RC12 exact-decode artifact grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np

FROZEN_CONFIG = {
    "frame_budget": 16,
    "anchor_count": 12,
    "max_lattice_hz": 2.0,
    "smoothing_sigma_sec": 4.0,
    "score_quantum": 0.1,
    "min_residual_distance_sec": 2.0,
    "coverage_tiebreak_cap_sec": 8.0,
    "lattice_origin_sec": 0.0,
}


def rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb_sha256(frame) -> str:
    rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
    return hashlib.sha256(memoryview(rgb).cast("B")).hexdigest()


def audit_qwen_index_roundtrip(decision_map, trace_map) -> dict:
    """Re-open every source with Qwen's enumerate(container.decode) index space."""

    entries_by_video = defaultdict(list)
    video_ids_by_path = defaultdict(set)
    for key, trace in trace_map.items():
        indices = trace.get("selected_source_frame_indices")
        hashes = trace.get("selected_pixel_hashes")
        if not isinstance(hashes, list) or len(hashes) != 16:
            raise SystemExit(f"missing selected pixel hashes: {key}")
        video_path = str(Path(decision_map[key]["video_path"]).resolve())
        video_ids_by_path[video_path].add(str(key[1]))
        entries_by_video[video_path].append((key, indices, hashes))

    if len(entries_by_video) != 20 or any(
        len(video_ids) != 1 for video_ids in video_ids_by_path.values()
    ):
        raise SystemExit("source-video paths do not map one-to-one to dev20 video IDs")

    total_logical = 0
    total_unique = 0
    per_video_unique = {}
    for video_path, entries in sorted(entries_by_video.items()):
        path = Path(video_path)
        if not path.is_file():
            raise SystemExit(f"roundtrip source video is missing: {path}")
        wanted = {index for _, indices, _ in entries for index in indices}
        observed = {}
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise SystemExit(f"roundtrip source has no video stream: {path}")
            stream = container.streams.video[0]
            for source_index, frame in enumerate(container.decode(stream)):
                if source_index in wanted:
                    observed[source_index] = _rgb_sha256(frame)
                    if len(observed) == len(wanted):
                        break
        missing = sorted(wanted - set(observed))
        if missing:
            raise SystemExit(
                f"Qwen roundtrip frame indices out of range for {path}: {missing[:8]}"
            )
        for key, indices, expected_hashes in entries:
            actual_hashes = [observed[index] for index in indices]
            if actual_hashes != expected_hashes:
                raise SystemExit(f"Qwen roundtrip pixel hash mismatch: {key}")
            total_logical += len(indices)
        total_unique += len(wanted)
        per_video_unique[str(path)] = len(wanted)
    return {
        "status": "validated",
        "enumeration_contract": "enumerate(container.decode(video_stream))",
        "logical_selected_hash_comparisons": total_logical,
        "unique_source_frames_reopened": total_unique,
        "num_video_decode_passes": len(entries_by_video),
        "unique_source_frames_by_video": per_video_unique,
    }


def audit_keyframe_exports(
    keyframes: Path, trace_map, expected_names: set[str]
) -> dict:
    """Bind every exported keyframe index array back to its exact trace row."""

    observed_names = {path.name for path in keyframes.glob("*.json")}
    if observed_names != expected_names:
        raise SystemExit(
            "keyframe export filenames are not the exact frozen 10-cell grid"
        )
    bundle = hashlib.sha256()
    row_count = 0
    for name in sorted(expected_names):
        path = keyframes / name
        bundle.update(name.encode("utf-8"))
        bundle.update(b"\0")
        bundle.update(path.read_bytes())
        bundle.update(b"\0")
        stem = name.removeprefix("videomme_").removesuffix(".json")
        method, raw_origin = stem.rsplit("_origin", 1)
        origin = int(raw_origin)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or len(payload) != 60:
            raise SystemExit(f"keyframe export is not a 60-item cell: {name}")
        seen = set()
        for item in payload:
            if not isinstance(item, dict):
                raise SystemExit(f"invalid keyframe annotation row: {name}")
            identity = (str(item.get("video_id")), str(item.get("question_id")))
            if identity in seen:
                raise SystemExit(
                    f"duplicate keyframe annotation identity: {name}/{identity}"
                )
            seen.add(identity)
            trace_key = ("videomme", *identity, origin, method)
            trace = trace_map.get(trace_key)
            if trace is None or item.get("keyframe_indices") != trace.get(
                "selected_source_frame_indices"
            ):
                raise SystemExit(
                    f"keyframe indices do not match trace: {name}/{identity}"
                )
            row_count += 1
    return {
        "sha256_name_nul_bytes_nul": bundle.hexdigest(),
        "num_files": len(expected_names),
        "num_rows": row_count,
        "filenames": sorted(expected_names),
        "trace_indices_exact_match": True,
    }


def audit_frozen_decisions(decisions: list[dict]) -> dict:
    by_video = defaultdict(list)
    for row in decisions:
        by_video[str(row["video_id"])].append(row)
    if len(by_video) != 20:
        raise SystemExit("frozen decisions require exactly 20 videos")
    resolved_paths = set()
    for video_id, video_rows in by_video.items():
        if (
            len(video_rows) != 30
            or len({str(r["question_id"]) for r in video_rows}) != 3
            or {int(r["origin_id"]) for r in video_rows} != set(range(5))
        ):
            raise SystemExit(f"non-rectangular frozen video grid: {video_id}")
        paths = {str(Path(r["video_path"]).resolve()) for r in video_rows}
        supports = {
            (float(r["support_start_sec"]), float(r["support_stop_sec"]))
            for r in video_rows
        }
        lattices = {tuple(map(float, r["lattice_timestamps_sec"])) for r in video_rows}
        if len(paths) != 1 or len(supports) != 1 or len(lattices) != 1:
            raise SystemExit(f"video path/support/lattice drift: {video_id}")
        resolved_paths.update(paths)
        start, stop = next(iter(supports))
        lattice = list(next(iter(lattices)))
        expected = (
            0.5
            * np.arange(
                int(np.ceil((start - 1e-9) / 0.5)),
                int(np.floor((stop + 1e-9) / 0.5)) + 1,
            )
        ).tolist()
        if lattice != expected:
            raise SystemExit(f"not complete origin-zero 0.5s lattice: {video_id}")
        values = np.asarray(lattice)
        available = np.ones(values.size, dtype=bool)
        uniform = []
        for center in values[0] + (np.arange(16) + 0.5) / 16 * (values[-1] - values[0]):
            index = int(np.argmin(np.where(available, np.abs(values - center), np.inf)))
            uniform.append(index)
            available[index] = False
        uniform = sorted(uniform)
        for row in video_rows:
            indices = row["target_indices"]
            meta = row["decision_metadata"]
            if row["method"] == "canonical_uniform":
                if indices != uniform or any(
                    float(x) != 0.0 for x in row["quantized_scores"]
                ):
                    raise SystemExit(f"uniform decision drift: {video_id}")
            else:
                anchors = meta.get("anchor_indices")
                residuals = meta.get("residual_indices")
                if (
                    meta.get("config") != FROZEN_CONFIG
                    or len(anchors or []) != 12
                    or len(residuals or []) != 4
                    or sorted(anchors + residuals) != indices
                    or meta.get("decision_step_sec") != 0.5
                    or meta.get("decision_signal") != "smoothed_query_relevance_only"
                    or meta.get("ti_dwt_role") != "diagnostic_only"
                ):
                    raise SystemExit(f"RC12 frozen decision drift: {video_id}")
    if len(resolved_paths) != 20:
        raise SystemExit("video paths are not one-to-one with video IDs")
    return {"status": "validated", "num_videos": 20, "config": FROZEN_CONFIG}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--traces", type=Path, required=True)
    ap.add_argument("--keyframes", type=Path, required=True)
    ap.add_argument("--source-video-bundle", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    decisions = rows(args.decisions)
    traces = rows(args.traces)
    methods = {"canonical_uniform", "phasefuse_rc12"}
    if len(decisions) != 600 or len(traces) != 600:
        raise SystemExit("RC12 dev20 requires exactly 600 decisions and 600 traces")
    decision_map = {
        (
            r["dataset"],
            r["video_id"],
            r["question_id"],
            int(r["origin_id"]),
            r["method"],
        ): r
        for r in decisions
    }
    trace_map = {
        (
            r["dataset"],
            r["video_id"],
            r["question_id"],
            int(r["origin_id"]),
            r["method"],
        ): r
        for r in traces
    }
    if len(decision_map) != 600 or set(decision_map) != set(trace_map):
        raise SystemExit("decision and trace grids do not align exactly")
    frozen_decisions = audit_frozen_decisions(decisions)
    grid = defaultdict(set)
    repair_attempts = duplicate_rejections = relaxations = 0
    uniform_payload = {}
    for key, trace in trace_map.items():
        if key[-1] not in methods:
            raise SystemExit("unexpected method")
        grid[key[:3]].add(key[3])
        selected = trace.get("selected_source_frame_indices")
        if (
            not isinstance(selected, list)
            or len(selected) != 16
            or len(set(selected)) != 16
            or selected != sorted(selected)
        ):
            raise SystemExit(f"invalid exact decoded source budget: {key}")
        targets = trace.get("selected_target_timestamps_sec") or trace.get(
            "selected_timestamps_sec"
        )
        if not isinstance(targets, list) or len(targets) != 16:
            raise SystemExit(f"missing selected canonical target timestamps: {key}")
        actual = trace.get("selected_actual_pts_sec")
        if not isinstance(actual, list) or len(actual) != 16:
            raise SystemExit(f"missing fresh actual PTS: {key}")
        metadata = trace.get("canonical_decode", {})
        if (
            metadata.get("fresh_source_decode") is not True
            or metadata.get("prohibit_scout_frame_remap") is not True
        ):
            raise SystemExit(f"fresh-decode contract absent: {key}")
        if (
            metadata.get("decoder_backend") != "pyav_sequential_nearest_pts"
            or metadata.get("midpoint_tie_policy") != "earlier_pts"
        ):
            raise SystemExit(f"unexpected exact decoder provenance: {key}")
        repair_attempts += int(metadata.get("repair_attempt_count", 0))
        duplicate_rejections += int(metadata.get("duplicate_rejection_count", 0))
        relaxations += int(metadata.get("distance_relaxation_count", 0))
        decision = decision_map[key]
        lattice = decision["lattice_timestamps_sec"]
        support = (
            float(decision["support_start_sec"]),
            float(decision["support_stop_sec"]),
        )
        if any(
            target not in lattice or target < support[0] or target > support[1]
            for target in targets
        ):
            raise SystemExit(f"selected exact targets escaped canonical support: {key}")
        attempts = metadata.get("attempts")
        if not isinstance(attempts, list):
            raise SystemExit(f"missing exact-decode attempts: {key}")
        primary_targets = [
            attempt.get("target_sec")
            for attempt in attempts
            if attempt.get("stage") == "primary"
        ]
        if primary_targets != decision["target_timestamps_sec"]:
            raise SystemExit(
                f"primary attempts do not reproduce frozen decision: {key}"
            )
        for attempt in attempts:
            index = attempt.get("canonical_index")
            target = attempt.get("target_sec")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(lattice)
                or target != lattice[index]
            ):
                raise SystemExit(f"attempt is not bound to its canonical tick: {key}")
            if attempt.get("stage") == "repair" and attempt.get("repair_source") != (
                "dynamic_frozen_residual_priority"
            ):
                raise SystemExit(
                    f"repair provenance is not frozen dynamic priority: {key}"
                )
        if key[-1] == "canonical_uniform":
            payload = (targets, actual, selected, trace.get("selected_pixel_hashes"))
            prior = uniform_payload.setdefault((key[1], key[2]), payload)
            if payload != prior:
                raise SystemExit(
                    f"uniform exact payload varies across origins: {key[:3]}"
                )
    paired_union_fields = (
        "timestamps_sec",
        "actual_pts_sec",
        "source_frame_indices",
        "pixel_hashes",
        "candidate_provenance",
    )
    for item_key in grid:
        for origin in range(5):
            base = (*item_key, origin)
            uniform = trace_map[(*base, "canonical_uniform")]
            rc12 = trace_map[(*base, "phasefuse_rc12")]
            if any(uniform.get(name) != rc12.get(name) for name in paired_union_fields):
                raise SystemExit(
                    f"paired candidate unions do not align exactly: {base}"
                )
    if len(grid) != 60 or any(value != set(range(5)) for value in grid.values()):
        raise SystemExit("trace cohort is not a 60-item x five-origin grid")
    expected = {f"videomme_{m}_origin{o:02d}.json" for m in methods for o in range(5)}
    keyframe_bundle = audit_keyframe_exports(args.keyframes, trace_map, expected)
    qwen_roundtrip = audit_qwen_index_roundtrip(decision_map, trace_map)
    result = {
        "schema_version": 1,
        "status": "validated",
        "num_rows": 600,
        "num_items": 60,
        "num_videos": 20,
        "methods": sorted(methods),
        "origins": [0, 1, 2, 3, 4],
        "frame_budget": 16,
        "repair_attempts": repair_attempts,
        "duplicate_rejections": duplicate_rejections,
        "distance_relaxations": relaxations,
        "decisions_sha256": sha(args.decisions),
        "traces_sha256": sha(args.traces),
        "fresh_source_decode": True,
        "scout_actual_pts_remap_used": False,
        "qwen_frame_index_pixel_hash_roundtrip": qwen_roundtrip,
        "keyframe_bundle": keyframe_bundle,
        "frozen_decisions": frozen_decisions,
        "source_video_bundle_sha256": sha(args.source_video_bundle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
