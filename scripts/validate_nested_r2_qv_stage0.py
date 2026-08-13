#!/usr/bin/env python3
"""Validate and seal the frozen nested-R2 QV Stage-0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import av
import numpy as np

from phase_stable.artifacts import read_signal_records
from phase_stable.canonical_nested_r2 import NestedR2Config
from phase_stable.sampling import read_manifests_jsonl

METHODS = ("canonical_uniform", "phasefuse_nested_r2")
ORIGINS = set(range(5))
EXPECTED_COHORT_SHA256 = (
    "349f450a169fc1160429738f30028f38b6401b5f33c3760d02d70b3111ccb3b4"
)
EXPECTED_PREREG_SHA256 = (
    "694522db81e5f9db5d244f713c03fd26844c5f680828f8fc3df0b4ed992c9cff"
)
EXPECTED_CONFIG_SHA256 = (
    "d3512cf41506b214c86cce72f607390184cecfa7b9bd4750adcc4122788e61e9"
)
EXPECTED_CHECKPOINT_REVISION = "520bf73fd0ef6ccce791cd3f06908040aef2d8cc"
EXPECTED_CHECKPOINT_TREE_SHA256 = (
    "93f537621bfb693fac5a52760a5b83bd93eb3dd1885292e680b62bc81d962b1f"
)
EXPECTED_CHECKPOINT_ROOT = Path(
    "/home/elicer/.cache/huggingface/hub/"
    "models--Salesforce--blip2-itm-vit-g/snapshots/"
    f"{EXPECTED_CHECKPOINT_REVISION}"
)
EXPECTED_VIDEOS_ROOT = Path(
    "/home/elicer/videounderstanding/outputs/local-data/qvhighlights/videos"
)
EXPECTED_CHECKPOINT_FILE_COUNT = 10
EXPECTED_CHECKPOINT_TOTAL_BYTES = 9473140308
RUNTIME_PACKAGES = ("numpy", "scipy", "av", "torch", "transformers", "Pillow")
EXPECTED_DECODER_CONTRACT = {
    "backend": "pyav_sequential_nearest_pts",
    "fresh_source_decode": True,
    "prohibit_scout_actual_pts_remap": True,
    "midpoint_tie": "earlier_pts",
    "deduplicate_by": "decoded_frame_index",
    "dynamic_repair_priority": True,
}
EXPECTED_NESTED_COVERAGE_CONTRACT = {
    "same_uniform_voronoi_region": True,
    "edge_anchors_immutable": True,
    "adjacent_donor_slots_forbidden": True,
    "minimum_unchanged_control_slots": 14,
    "distance_relaxation_allowed": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_tree_sha256(rows: list[dict[str, Any]]) -> str:
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def _expected_runtime_code_paths(repo: Path) -> list[Path]:
    return sorted(
        [
            *repo.joinpath("phase_stable").glob("*.py"),
            *repo.joinpath("scripts").glob("*nested_r2*qv*.py"),
            repo / "scripts/decode_rc12_exact.py",
            repo / "scripts/run_nested_r2_qv_stage0.sh",
        ]
    )


def _current_runtime_code_rows(repo: Path) -> list[list[str]]:
    return [
        [path.relative_to(repo).as_posix(), sha256_file(path)]
        for path in _expected_runtime_code_paths(repo)
    ]


def _audit_checkpoint_preflight(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshot_root = Path(str(payload.get("snapshot_path", "")))
    reported_rows = payload.get("files")
    if (
        payload.get("schema_version") != 1
        or payload.get("algorithm")
        != "sha256(canonical-json(sorted[{path,size_bytes,sha256}]))"
        or payload.get("revision") != EXPECTED_CHECKPOINT_REVISION
        or not snapshot_root.is_absolute()
        or snapshot_root.resolve() != EXPECTED_CHECKPOINT_ROOT
        or not isinstance(reported_rows, list)
    ):
        raise ValueError("checkpoint preflight schema/root/revision drift")
    actual_rows = _snapshot_rows(snapshot_root)
    if reported_rows != actual_rows:
        raise ValueError("checkpoint member path/size/content binding drift")
    tree_sha256 = _canonical_tree_sha256(actual_rows)
    if (
        len(actual_rows) != EXPECTED_CHECKPOINT_FILE_COUNT
        or sum(int(row["size_bytes"]) for row in actual_rows)
        != EXPECTED_CHECKPOINT_TOTAL_BYTES
        or tree_sha256 != EXPECTED_CHECKPOINT_TREE_SHA256
        or payload.get("num_files") != len(actual_rows)
        or payload.get("total_size_bytes")
        != sum(int(row["size_bytes"]) for row in actual_rows)
        or payload.get("content_tree_sha256") != tree_sha256
    ):
        raise ValueError("checkpoint independently recomputed identity drift")
    return payload


def _audit_runtime_manifest(
    path: Path,
    checkpoint_path: Path,
    expected_git_head: str,
    *,
    repo: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    root = (repo or Path.cwd()).resolve()
    expected_rows = _current_runtime_code_rows(root)
    reported_rows = payload.get("code_files")
    current_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    current_clean = not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
    ).strip()
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if (
        payload.get("schema_version") != 1
        or not _is_git_oid(expected_git_head)
        or payload.get("git_head") != expected_git_head
        or current_head != expected_git_head
        or payload.get("git_status_clean") is not True
        or not current_clean
        or payload.get("qwen_commands_present") is not False
        or payload.get("checkpoint_preflight_sha256") != sha256_file(checkpoint_path)
        or payload.get("python") != sys.version
        or payload.get("python_executable_raw") != sys.executable
        or payload.get("python_executable_resolved")
        != str(Path(sys.executable).resolve())
        or payload.get("packages") != packages
        or reported_rows != expected_rows
        or payload.get("code_tree_sha256")
        != hashlib.sha256(
            json.dumps(expected_rows, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("runtime manifest independently recomputed semantics drift")
    return payload


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"non-object JSONL row: {path}:{number}")
        rows.append(value)
    return rows


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(row["dataset"]),
        str(row["video_id"]),
        str(row["question_id"]),
        int(row["origin_id"]),
        str(row["method"]),
    )


def _int_list(payload: Mapping[str, Any], name: str) -> list[int]:
    values = payload.get(name)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError(f"{name} must be an exact integer list")
    return list(values)


def _cohort_rows(path: Path) -> list[tuple[int, str, int, str, str]]:
    if sha256_file(path) != EXPECTED_COHORT_SHA256:
        raise ValueError("frozen QV cohort hash drift")
    result = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"invalid cohort row schema: {line_number}")
        rank, source, annotation_index, vid, qid = fields
        row = (int(rank), source, int(annotation_index), vid, str(qid))
        if row[0] != len(result) or vid.rsplit("_", 2)[0] != source:
            raise ValueError(f"invalid cohort identity/order: {line_number}")
        result.append(row)
    if (
        len(result) != 100
        or len({row[1] for row in result}) != 100
        or len({(row[3], row[4]) for row in result}) != 100
    ):
        raise ValueError("QV holdout cohort is not 100 rows")
    return result


def _query_identity(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows = read_rows(path)
    forbidden = {"relevant_windows", "relevant_clip_ids", "saliency_scores"}
    expected_row_keys = {"video_id", "query_id", "query", "duration", "metadata"}
    expected_metadata_keys = {
        "annotation_row_index",
        "source_id",
        "label_firewall",
    }
    result = {}
    for index, row in enumerate(rows):
        metadata = row.get("metadata")
        if set(row) != expected_row_keys or not isinstance(metadata, Mapping):
            raise ValueError(f"query-only row schema drift: {index}")
        if set(metadata) != expected_metadata_keys:
            raise ValueError(f"query-only metadata schema drift: {index}")
        if _contains_mapping_key(row, forbidden):
            raise ValueError("pre-seal query manifest leaked evaluation labels")
        key = (str(row["video_id"]), str(row["query_id"]))
        if key in result:
            raise ValueError("duplicate query identity")
        source_id = key[0].rsplit("_", 2)[0]
        duration = row.get("duration")
        if (
            not isinstance(row.get("query"), str)
            or not row["query"].strip()
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0
            or metadata.get("source_id") != source_id
            or isinstance(metadata.get("annotation_row_index"), bool)
            or not isinstance(metadata.get("annotation_row_index"), int)
            or metadata.get("label_firewall") != "query_only_before_blind_seal"
        ):
            raise ValueError("query row is missing the label firewall marker")
        result[key] = row
    if len(result) != 100:
        raise ValueError("query-only manifest is not 100 rows")
    return result


def _contains_mapping_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        return bool(forbidden & set(value)) or any(
            _contains_mapping_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_mapping_key(child, forbidden) for child in value)
    return False


def _source_inventory(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "frozen_qv_source_inventory"
        or Path(str(payload.get("videos_root"))).resolve() != EXPECTED_VIDEOS_ROOT
        or payload.get("num_videos") != 100
    ):
        raise ValueError("source inventory schema/root/status drift")
    rows = payload.get("videos")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("source inventory is not 100 videos")
    expected_keys = {
        "video_id",
        "question_id",
        "source_id",
        "video_path",
        "size_bytes",
        "sha256",
    }
    if any(not isinstance(row, Mapping) or set(row) != expected_keys for row in rows):
        raise ValueError("source inventory member schema drift")
    result = {str(row["video_id"]): dict(row) for row in rows}
    if len(result) != 100:
        raise ValueError("source inventory video IDs are not unique")
    for video_id, row in result.items():
        source = str(video_id).rsplit("_", 2)[0]
        if source != row.get("source_id"):
            raise ValueError("source inventory source_id drift")
        target = Path(str(row["video_path"]))
        if target.resolve() != (EXPECTED_VIDEOS_ROOT / f"{video_id}.mp4"):
            raise ValueError(f"source path escaped frozen videos root: {video_id}")
        size = row.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not _is_sha256(row.get("sha256"))
            or not target.is_file()
            or target.stat().st_size != size
        ):
            raise ValueError(f"source file binding drift: {video_id}")
        if sha256_file(target) != row.get("sha256"):
            raise ValueError(f"source SHA-256 drift: {video_id}")
    return result


def _uniform_indices(lattice: list[float]) -> list[int]:
    values = np.asarray(lattice, dtype=float)
    available = np.ones(values.size, dtype=bool)
    selected = []
    centers = values[0] + (np.arange(16) + 0.5) / 16 * (values[-1] - values[0])
    for center in centers:
        index = int(np.argmin(np.where(available, np.abs(values - center), np.inf)))
        selected.append(index)
        available[index] = False
    return sorted(selected)


def audit_decisions(
    decisions: list[dict[str, Any]], expected_items: set[tuple[str, str]]
) -> dict[str, Any]:
    expected_keys = {
        ("qvhighlights", video_id, question_id, origin, method)
        for video_id, question_id in expected_items
        for origin in ORIGINS
        for method in METHODS
    }
    decision_map = {_key(row): row for row in decisions}
    if (
        len(decisions) != 1000
        or len(decision_map) != 1000
        or set(decision_map) != expected_keys
    ):
        raise ValueError("decision grid is not exact 100 x 5 x 2")
    expected_row_keys = {
        "schema_version",
        "dataset",
        "video_id",
        "question_id",
        "origin_id",
        "origin_sec",
        "video_path",
        "support_start_sec",
        "support_stop_sec",
        "lattice_timestamps_sec",
        "decoder_contract",
        "record_metadata",
        "method",
        "target_indices",
        "target_timestamps_sec",
        "quantized_scores",
        "decision_metadata",
    }
    relocations = defaultdict(int)
    for key, row in decision_map.items():
        if (
            set(row) != expected_row_keys
            or row.get("schema_version") != 1
            or row.get("dataset") != "qvhighlights"
            or row.get("decoder_contract") != EXPECTED_DECODER_CONTRACT
            or not Path(str(row.get("video_path", ""))).is_absolute()
            or _contains_mapping_key(
                row.get("record_metadata", {}),
                {"relevant_windows", "relevant_clip_ids", "saliency_scores"},
            )
        ):
            raise ValueError(f"decision schema/provenance drift: {key}")
        lattice = [float(value) for value in row["lattice_timestamps_sec"]]
        if len(lattice) < 16 or any(
            not math.isclose(right - left, 1.0, abs_tol=1e-9)
            for left, right in itertools.pairwise(lattice)
        ):
            raise ValueError(f"QV canonical lattice is not origin-zero 1 Hz: {key}")
        if any(
            not math.isclose(value, round(value), abs_tol=1e-9) for value in lattice
        ):
            raise ValueError(f"QV canonical lattice contains non-integer ticks: {key}")
        support_start = float(row["support_start_sec"])
        support_stop = float(row["support_stop_sec"])
        origin_sec = float(row["origin_sec"])
        if (
            not 0.0 <= origin_sec < 1.0
            or not support_start <= lattice[0] <= support_start + 1.0
            or not support_stop - 1.0 <= lattice[-1] <= support_stop
        ):
            raise ValueError(f"decision support/origin drift: {key}")
        indices = _int_list(row, "target_indices")
        scores = np.asarray(row["quantized_scores"], dtype=float)
        if (
            scores.shape != (len(lattice),)
            or not np.all(np.isfinite(scores))
            or np.any(scores < -1e-12)
            or np.any(scores > 1.0 + 1e-12)
            or not np.allclose(scores * 10.0, np.round(scores * 10.0), atol=1e-9)
        ):
            raise ValueError(f"decision quantized score contract drift: {key}")
        if len(indices) != 16 or len(set(indices)) != 16 or indices != sorted(indices):
            raise ValueError(f"decision is not exact K=16: {key}")
        if [lattice[index] for index in indices] != row["target_timestamps_sec"]:
            raise ValueError(f"decision target/index binding drift: {key}")
        uniform = _uniform_indices(lattice)
        if key[-1] == "canonical_uniform":
            if (
                indices != uniform
                or any(float(value) != 0.0 for value in row["quantized_scores"])
                or row.get("decision_metadata")
                != {
                    "decision_signal": "none_uniform_control",
                    "ti_dwt_role": "not_used",
                    "phase_effect_claim": False,
                }
            ):
                raise ValueError(f"canonical uniform decision drift: {key}")
            continue
        meta = row.get("decision_metadata", {})
        if not isinstance(meta, Mapping):
            raise TypeError(f"nested-R2 decision metadata is missing: {key}")
        base = _int_list(meta, "base_uniform_indices")
        anchors = _int_list(meta, "anchor_indices")
        donors = _int_list(meta, "donor_indices")
        residuals = _int_list(meta, "residual_indices")
        moves = meta.get("relocations", [])
        if (
            meta.get("schema_version") != 1
            or meta.get("method") != "nested_r2"
            or meta.get("config") != asdict(NestedR2Config())
            or not math.isclose(
                float(meta.get("support_start_sec", math.nan)), support_start
            )
            or not math.isclose(
                float(meta.get("support_stop_sec", math.nan)), support_stop
            )
            or not math.isclose(float(meta.get("scout_step_sec", math.nan)), 1.0)
            or not math.isclose(float(meta.get("decision_step_sec", math.nan)), 1.0)
        ):
            raise ValueError(f"nested-R2 frozen config drift: {key}")
        if base != uniform or len(donors) != len(residuals) or len(donors) > 2:
            raise ValueError(f"nested-R2 base/relocation cardinality drift: {key}")
        if sorted(anchors + residuals) != indices or sorted(anchors + donors) != base:
            raise ValueError(
                f"nested-R2 provenance does not reconstruct targets: {key}"
            )
        if (
            meta.get("preserved_base_indices") != anchors
            or meta.get("selected_indices") != indices
            or meta.get("base_uniform_timestamps_sec")
            != [lattice[index] for index in base]
            or meta.get("preserved_base_timestamps_sec")
            != [lattice[index] for index in anchors]
            or meta.get("residual_timestamps_sec")
            != [lattice[index] for index in residuals]
            or meta.get("target_timestamps_sec") != row["target_timestamps_sec"]
        ):
            raise ValueError(f"nested-R2 metadata array binding drift: {key}")
        donor_slots = [base.index(index) for index in donors]
        if any(slot in {0, 15} for slot in donor_slots) or any(
            right - left <= 1 for left, right in itertools.pairwise(donor_slots)
        ):
            raise ValueError(f"nested-R2 donor safety drift: {key}")
        if len(anchors) < 14 or base[0] not in anchors or base[-1] not in anchors:
            raise ValueError(f"nested-R2 retained-anchor floor drift: {key}")
        owners = np.argmin(
            np.abs(np.asarray(lattice)[:, None] - np.asarray(lattice)[base][None, :]),
            axis=1,
        )
        for donor, residual in zip(donors, residuals):
            if owners[residual] != base.index(donor):
                raise ValueError(f"nested-R2 residual escaped donor region: {key}")
        if not np.array_equal(
            np.bincount(owners[indices], minlength=16), np.ones(16, dtype=int)
        ):
            raise ValueError(
                f"nested-R2 is not one target per A16 Voronoi region: {key}"
            )
        if (
            meta.get("relocation_count") != len(donors)
            or meta.get("unchanged_base_slot_count") != len(anchors)
            or len(moves) != len(donors)
            or any(float(move["quantized_gain"]) <= 0 for move in moves)
        ):
            raise ValueError(f"nested-R2 relocation gain drift: {key}")
        for move, donor, residual in zip(moves, donors, residuals):
            donor_slot = base.index(donor)
            nearest = min(
                abs(lattice[residual] - lattice[index])
                for index in base
                if index != donor
            )
            if (
                int(move.get("donor_slot", -1)) != donor_slot
                or int(move.get("donor_index", -1)) != donor
                or int(move.get("residual_index", -1)) != residual
                or not math.isclose(
                    float(move.get("donor_timestamp_sec", math.nan)),
                    lattice[donor],
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(move.get("residual_timestamp_sec", math.nan)),
                    lattice[residual],
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(move["donor_score"]), float(scores[donor]), abs_tol=1e-9
                )
                or not math.isclose(
                    float(move["residual_score"]), float(scores[residual]), abs_tol=1e-9
                )
                or not math.isclose(
                    float(move["quantized_gain"]),
                    float(scores[residual] - scores[donor]),
                    abs_tol=1e-9,
                )
                or abs(lattice[residual] - lattice[donor]) < 2.0 - 1e-9
                or min(
                    abs(lattice[residual] - lattice[index])
                    for index in base
                    if index != donor
                )
                < 2.0 - 1e-9
                or not math.isclose(
                    float(move.get("nearest_retained_anchor_sec", math.nan)),
                    nearest,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    f"nested-R2 relocation provenance/distance drift: {key}"
                )
        if (
            meta.get("decision_signal") != "frozen_rc12_smoothed_query_relevance_only"
            or meta.get("coverage_contract") != EXPECTED_NESTED_COVERAGE_CONTRACT
        ):
            raise ValueError(f"nested-R2 scoring/coverage role drift: {key}")
        if (
            meta.get("ti_dwt_role") != "not_used"
            or meta.get("phase_effect_claim") is not False
        ):
            raise ValueError(f"nested-R2 scientific role drift: {key}")
        relocations[len(donors)] += 1
    paired_fields = {
        "schema_version",
        "dataset",
        "video_id",
        "question_id",
        "origin_id",
        "origin_sec",
        "video_path",
        "support_start_sec",
        "support_stop_sec",
        "lattice_timestamps_sec",
        "decoder_contract",
        "record_metadata",
    }
    for video_id, question_id in expected_items:
        uniform_targets = None
        for origin in ORIGINS:
            uniform = decision_map[
                ("qvhighlights", video_id, question_id, origin, "canonical_uniform")
            ]
            treatment = decision_map[
                ("qvhighlights", video_id, question_id, origin, "phasefuse_nested_r2")
            ]
            if any(uniform[field] != treatment[field] for field in paired_fields):
                raise ValueError("paired decision common-field drift")
            if uniform_targets is None:
                uniform_targets = uniform["target_timestamps_sec"]
            elif uniform["target_timestamps_sec"] != uniform_targets:
                raise ValueError("canonical uniform control varies across origins")
    return {
        "status": "validated",
        "num_rows": len(decisions),
        "relocation_count_cells": {str(key): relocations[key] for key in range(3)},
        "lattice": "origin_zero_1hz",
        "frame_budget": 16,
        "minimum_exact_a16_retained": 14,
    }


def audit_traces(
    traces: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    source_inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decision_map = {_key(row): row for row in decisions}
    trace_map = {_key(row): row for row in traces}
    if (
        len(traces) != 1000
        or len(trace_map) != 1000
        or set(trace_map) != set(decision_map)
    ):
        raise ValueError("trace grid does not match decisions exactly")
    repair = duplicate = relaxation = 0
    entries_by_path: dict[
        str, list[tuple[tuple[str, str, str, int, str], list[int], list[str]]]
    ] = defaultdict(list)
    for key, trace in trace_map.items():
        decision = decision_map[key]
        if (
            trace.get("schema_version") != 1
            or float(trace.get("origin_sec", math.nan)) != float(decision["origin_sec"])
            or _contains_mapping_key(
                trace.get("record_metadata", {}),
                {"relevant_windows", "relevant_clip_ids", "saliency_scores"},
            )
        ):
            raise ValueError(f"trace schema/origin/label-firewall drift: {key}")
        selected_union_indices = trace.get("selected_indices")
        indices = trace.get("selected_source_frame_indices")
        hashes = trace.get("selected_pixel_hashes")
        actual = trace.get("selected_actual_pts_sec")
        targets = trace.get("selected_timestamps_sec")
        union_targets = trace.get("timestamps_sec")
        union_actual = trace.get("actual_pts_sec")
        union_indices = trace.get("source_frame_indices")
        union_hashes = trace.get("pixel_hashes")
        provenance = trace.get("candidate_provenance")
        if not all(
            isinstance(value, list) and len(value) == 16
            for value in (indices, hashes, actual, targets)
        ):
            raise ValueError(f"trace is not exact K=16: {key}")
        if (
            not isinstance(selected_union_indices, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in selected_union_indices
            )
            or len(selected_union_indices) != 16
            or selected_union_indices != sorted(set(selected_union_indices))
            or not all(
                isinstance(value, list)
                for value in (
                    union_targets,
                    union_actual,
                    union_indices,
                    union_hashes,
                    provenance,
                )
            )
            or len(
                {
                    len(union_targets),
                    len(union_actual),
                    len(union_indices),
                    len(union_hashes),
                    len(provenance),
                }
            )
            != 1
            or not union_targets
            or any(
                index < 0 or index >= len(union_targets)
                for index in selected_union_indices
            )
            or targets != [union_targets[index] for index in selected_union_indices]
            or actual != [union_actual[index] for index in selected_union_indices]
            or indices != [union_indices[index] for index in selected_union_indices]
            or hashes != [union_hashes[index] for index in selected_union_indices]
        ):
            raise ValueError(f"trace selected/candidate-union binding drift: {key}")
        if any(
            not isinstance(item, Mapping)
            or item.get("target_sec") != union_targets[index]
            or item.get("actual_pts_sec") != union_actual[index]
            or item.get("decoded_frame_index") != union_indices[index]
            or item.get("pixel_hash") != union_hashes[index]
            or item.get("midpoint_tie_policy") != "earlier_pts"
            or not isinstance(item.get("arm_attempts"), list)
            for index, item in enumerate(provenance)
        ):
            raise ValueError(f"trace candidate provenance drift: {key}")
        if len(set(indices)) != 16 or indices != sorted(indices):
            raise ValueError(f"fresh decoded frames are not unique/increasing: {key}")
        decode = trace.get("canonical_decode", {})
        if (
            decode.get("schema_version") != 1
            or decode.get("fresh_source_decode") is not True
            or decode.get("prohibit_scout_frame_remap") is not True
            or decode.get("decoder_backend") != "pyav_sequential_nearest_pts"
            or decode.get("midpoint_tie_policy") != "earlier_pts"
            or decode.get("frame_budget") != 16
            or decode.get("candidate_union_size") != len(union_targets)
            or not isinstance(decode.get("decode_passes"), list)
            or decode.get("decode_pass_count") != len(decode["decode_passes"])
        ):
            raise ValueError(f"fresh exact decode contract drift: {key}")
        attempts = decode.get("attempts")
        if not isinstance(attempts, list) or len(attempts) < 16:
            raise ValueError(f"decode attempts are missing: {key}")
        primary = [
            item.get("target_sec")
            for item in attempts
            if item.get("stage") == "primary"
        ]
        if primary != decision["target_timestamps_sec"]:
            raise ValueError(
                f"primary decode attempts do not reproduce decisions: {key}"
            )
        lattice = decision["lattice_timestamps_sec"]
        for attempt_index, attempt in enumerate(attempts):
            canonical_index = attempt.get("canonical_index")
            target = attempt.get("target_sec")
            pixel_hash = attempt.get("pixel_hash")
            if (
                attempt.get("attempt_index") != attempt_index
                or attempt.get("method") != key[-1]
                or attempt.get("stage") not in {"primary", "repair"}
                or (
                    attempt.get("stage") == "repair"
                    and attempt.get("repair_source")
                    != "dynamic_frozen_residual_priority"
                )
                or (
                    attempt.get("stage") == "primary"
                    and attempt.get("repair_source") is not None
                )
                or isinstance(canonical_index, bool)
                or not isinstance(canonical_index, int)
                or canonical_index < 0
                or canonical_index >= len(lattice)
                or target != lattice[canonical_index]
                or not isinstance(pixel_hash, str)
                or len(pixel_hash) != 64
                or any(character not in "0123456789abcdef" for character in pixel_hash)
            ):
                raise ValueError(f"decode attempt lattice/hash drift: {key}")
        selected_attempts = sorted(
            (attempt for attempt in attempts if attempt.get("status") == "selected"),
            key=lambda attempt: float(attempt["target_sec"]),
        )
        if (
            [attempt["target_sec"] for attempt in selected_attempts] != targets
            or [attempt["actual_pts_sec"] for attempt in selected_attempts] != actual
            or [attempt["decoded_frame_index"] for attempt in selected_attempts]
            != indices
            or [attempt["pixel_hash"] for attempt in selected_attempts] != hashes
        ):
            raise ValueError(f"trace selected attempt provenance drift: {key}")
        duplicate_losses = [
            attempt
            for attempt in attempts
            if attempt.get("status")
            in {"rejected_duplicate", "replaced_by_lower_error"}
        ]
        initial_losses = sum(
            attempt.get("duplicate_resolution_stage") == "initial"
            for attempt in duplicate_losses
        )
        repair_losses = sum(
            attempt.get("duplicate_resolution_stage") == "repair"
            for attempt in duplicate_losses
        )
        repairs = sum(attempt.get("stage") == "repair" for attempt in attempts)
        replacements = sum(
            attempt.get("status") == "replaced_by_lower_error" for attempt in attempts
        )
        relaxations = sum(bool(attempt.get("distance_relaxed")) for attempt in attempts)
        if (
            decode.get("repair_attempt_count") != repairs
            or decode.get("duplicate_rejection_count") != initial_losses + repair_losses
            or decode.get("initial_duplicate_rejection_count") != initial_losses
            or decode.get("repair_duplicate_rejection_count") != repair_losses
            or decode.get("duplicate_replacement_count") != replacements
            or decode.get("distance_relaxation_count") != relaxations
        ):
            raise ValueError(f"trace repair/duplicate count provenance drift: {key}")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError(f"selected pixel hash drift: {key}")
        repair += int(decode.get("repair_attempt_count", 0))
        duplicate += int(decode.get("duplicate_rejection_count", 0))
        relaxation += int(decode.get("distance_relaxation_count", 0))
        path = str(Path(str(decision["video_path"])).resolve())
        if path != str(Path(str(source_inventory[key[1]]["video_path"])).resolve()):
            raise ValueError(f"decision/source inventory path drift: {key}")
        entries_by_path[path].append((key, indices, hashes))
    if len(entries_by_path) != 100:
        raise ValueError("trace source paths are not one-to-one with 100 clips")
    paired_fields = (
        "timestamps_sec",
        "actual_pts_sec",
        "source_frame_indices",
        "pixel_hashes",
        "candidate_provenance",
    )
    for dataset, video_id, question_id, origin_id, method in trace_map:
        if method != "canonical_uniform":
            continue
        uniform = trace_map[
            (dataset, video_id, question_id, origin_id, "canonical_uniform")
        ]
        treatment = trace_map[
            (dataset, video_id, question_id, origin_id, "phasefuse_nested_r2")
        ]
        if any(uniform.get(field) != treatment.get(field) for field in paired_fields):
            raise ValueError(
                "paired exact-decode candidate union drift: "
                f"{(video_id, question_id, origin_id)}"
            )

    logical = unique_total = 0
    for path_text, entries in sorted(entries_by_path.items()):
        wanted = {index for _, indices, _ in entries for index in indices}
        observed: dict[int, str] = {}
        with av.open(path_text) as container:
            stream = container.streams.video[0]
            for source_index, frame in enumerate(container.decode(stream)):
                if source_index in wanted:
                    rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    observed[source_index] = hashlib.sha256(
                        memoryview(rgb).cast("B")
                    ).hexdigest()
                    if len(observed) == len(wanted):
                        break
        if set(observed) != wanted:
            raise ValueError(f"source frame roundtrip incomplete: {path_text}")
        for key, indices, hashes in entries:
            if [observed[index] for index in indices] != hashes:
                raise ValueError(f"source RGB hash roundtrip mismatch: {key}")
            logical += 16
        unique_total += len(wanted)
    return {
        "status": "validated",
        "num_rows": 1000,
        "num_video_decode_roundtrips": 100,
        "logical_selected_rgb_hash_comparisons": logical,
        "unique_source_frames_reopened": unique_total,
        "repair_attempt_count": repair,
        "duplicate_rejection_count": duplicate,
        "distance_relaxation_count": relaxation,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--sampling-manifests", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--source-video-bundle", type=Path, required=True)
    parser.add_argument("--blind-materialization", type=Path, required=True)
    parser.add_argument("--sealed-labels", type=Path, required=True)
    parser.add_argument("--checkpoint-preflight", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.prereg) != EXPECTED_PREREG_SHA256:
        raise ValueError("QV preregistration hash drift")
    if sha256_file(args.config) != EXPECTED_CONFIG_SHA256:
        raise ValueError("nested-R2 frozen config hash drift")
    cohort = _cohort_rows(args.cohort)
    query_map = _query_identity(args.query_manifest)
    expected_items = {(vid, qid) for _, _, _, vid, qid in cohort}
    if set(query_map) != expected_items:
        raise ValueError("query-only manifest does not align with frozen cohort")
    if list(query_map) != [(vid, qid) for _, _, _, vid, qid in cohort]:
        raise ValueError("query-only manifest does not preserve frozen cohort order")
    for rank, source_id, annotation_index, video_id, question_id in cohort:
        row = query_map[(video_id, question_id)]
        if (
            rank < 0
            or row["metadata"]["source_id"] != source_id
            or row["metadata"]["annotation_row_index"] != annotation_index
        ):
            raise ValueError("query-only metadata does not bind frozen cohort identity")
    source_inventory = _source_inventory(args.source_video_bundle)
    if set(source_inventory) != {vid for vid, _ in expected_items}:
        raise ValueError("source inventory does not align with frozen cohort")
    if any(
        str(source_inventory[video_id]["question_id"]) != question_id
        for video_id, question_id in expected_items
    ):
        raise ValueError("source inventory question identity drift")
    signals = read_signal_records(args.signals)
    if len(signals) != 500:
        raise ValueError("signal grid is not exact 100 x 5")
    signal_keys = {(row.video_id, row.question_id, row.origin_id) for row in signals}
    expected_signal_keys = {
        (vid, qid, origin) for vid, qid in expected_items for origin in ORIGINS
    }
    if signal_keys != expected_signal_keys:
        raise ValueError("signal identities/origins do not match the frozen cohort")
    forbidden = {"relevant_windows", "relevant_clip_ids", "saliency_scores"}
    manifest_rows = read_manifests_jsonl(args.sampling_manifests)
    manifests = {manifest.video_id: manifest for manifest in manifest_rows}
    if len(manifest_rows) != 100 or len(manifests) != 100:
        raise ValueError("sampling manifest grid is not 100 videos")
    feature_rows = []
    for record in signals:
        if _contains_mapping_key(record.metadata, forbidden):
            raise ValueError("pre-seal signal metadata leaked evaluation labels")
        manifest = manifests.get(record.video_id)
        if manifest is None:
            raise ValueError("signal video is absent from sampling manifests")
        origin = manifest.origins[record.origin_id]
        query_row = query_map[(record.video_id, record.question_id)]
        if (
            record.dataset != "qvhighlights"
            or manifest.master_seed != 20260810
            or manifest.num_origins != 5
            or manifest.sample_fps != 1.0
            or len(manifest.origins) != 5
            or manifest.duration_sec != float(query_row["duration"])
            or record.origin_sec != origin.origin_sec
            or record.timestamps_sec != origin.target_timestamps_sec
            or record.metadata.get("master_seed") != 20260810
            or record.metadata.get("sample_fps") != 1.0
            or record.metadata.get("query") != query_row["query"]
        ):
            raise ValueError("signal scout sampling contract drift")
        query_metadata = record.metadata.get("query_metadata", {})
        if query_metadata != query_row["metadata"]:
            raise ValueError("signal query metadata/cohort binding drift")
        if query_metadata.get("label_firewall") != "query_only_before_blind_seal":
            raise ValueError("signal query metadata lacks firewall marker")
        revision = str(record.metadata.get("feature_model_revision", ""))
        if (
            record.metadata.get("feature_model") != "blip2"
            or Path(revision).name != EXPECTED_CHECKPOINT_REVISION
            or record.metadata.get("feature_checkpoint_content_tree_sha256")
            != EXPECTED_CHECKPOINT_TREE_SHA256
        ):
            raise ValueError("signal BLIP2 checkpoint provenance drift")
        if not (
            len(record.timestamps_sec)
            == len(record.actual_pts_sec)
            == len(record.source_frame_indices)
            == len(record.relevance_scores)
            == len(record.pixel_hashes)
        ):
            raise ValueError("signal frame-aligned array drift")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in record.pixel_hashes
        ):
            raise ValueError("signal pixel hash drift")
        feature = Path(str(record.visual_features_path))
        if not feature.is_file():
            raise ValueError("signal feature member is missing")
        feature_rows.append(
            {
                "path": str(feature.resolve()),
                "size_bytes": feature.stat().st_size,
                "sha256": sha256_file(feature),
            }
        )
    if len({row["path"] for row in feature_rows}) != 500:
        raise ValueError("signal feature bundle is not 500 unique members")
    feature_rows.sort(key=lambda row: row["path"])
    feature_bundle_sha256 = hashlib.sha256(
        json.dumps(feature_rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decisions = read_rows(args.decisions)
    traces = read_rows(args.traces)
    materialization = json.loads(args.blind_materialization.read_text(encoding="utf-8"))
    if materialization.get("status") != "labels_hash_committed_query_only_materialized":
        raise ValueError("blind materialization status drift")
    if (
        materialization.get("num_rows") != 100
        or materialization.get("cohort_sha256") != EXPECTED_COHORT_SHA256
        or materialization.get("preregistration_sha256") != EXPECTED_PREREG_SHA256
        or materialization.get("query_manifest_sha256")
        != sha256_file(args.query_manifest)
        or materialization.get("sealed_labels_sha256")
        != sha256_file(args.sealed_labels)
        or materialization.get("sealed_label_plaintext_sha256")
        != sha256_file(args.sealed_labels)
        or materialization.get("cryptographic_blinding") is not False
    ):
        raise ValueError("blind materialization internal commitment drift")
    _audit_checkpoint_preflight(args.checkpoint_preflight)
    _audit_runtime_manifest(
        args.runtime_manifest,
        args.checkpoint_preflight,
        args.expected_git_head,
    )
    payload = {
        "schema_version": 1,
        "status": "sealed_before_label_join",
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "query_manifest_sha256": sha256_file(args.query_manifest),
        "sampling_manifests_sha256": sha256_file(args.sampling_manifests),
        "signals_sha256": sha256_file(args.signals),
        "feature_bundle_sha256": feature_bundle_sha256,
        "feature_bundle_num_files": len(feature_rows),
        "decisions_sha256": sha256_file(args.decisions),
        "traces_sha256": sha256_file(args.traces),
        "source_video_bundle_sha256": sha256_file(args.source_video_bundle),
        "blind_materialization_sha256": sha256_file(args.blind_materialization),
        "sealed_labels_sha256": sha256_file(args.sealed_labels),
        "checkpoint_preflight_sha256": sha256_file(args.checkpoint_preflight),
        "runtime_manifest_sha256": sha256_file(args.runtime_manifest),
        "labels_opened": False,
        "label_join_allowed_after_this_seal": True,
        "decision_audit": audit_decisions(decisions, expected_items),
        "trace_audit": audit_traces(traces, decisions, source_inventory),
    }
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
