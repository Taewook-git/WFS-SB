"""Two-stage PhaseFuse-RC12 exact-resample experiment construction."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import OriginSignalRecord, write_jsonl
from .canonical_residual import (
    RC12Config,
    select_canonical_uniform_targets,
    select_rc12_targets,
)

METHODS = ("canonical_uniform", "phasefuse_rc12")


def _decision_record_metadata(record: OriginSignalRecord) -> dict[str, Any]:
    """Keep scout provenance while excluding every decoded scout-frame mapping."""

    metadata = record.metadata
    safe = {
        name: metadata[name]
        for name in (
            "feature_device",
            "feature_dtype",
            "feature_layout",
            "feature_model",
            "feature_model_requested",
            "feature_model_revision",
            "feature_shape",
            "phasefuse_config_sha256",
            "query",
            "query_metadata",
            "sample_fps",
        )
        if name in metadata
    }
    multiphase = metadata.get("multiphase")
    if isinstance(multiphase, Mapping):
        safe["multiphase_contract"] = {
            name: multiphase[name]
            for name in (
                "schema_version",
                "outer_origin_id",
                "outer_origin_sec",
                "base_sample_fps",
                "dense_sample_fps",
                "num_phases",
                "phase_order",
                "phase_ids",
                "candidate_count_per_phase",
                "dense_candidate_count",
                "common_valid_support_sec",
                "manifest_common_valid_support_sec",
                "no_extrapolation",
            )
            if name in multiphase
        }
    return safe


def load_video_contracts(
    catalog_path: str | Path, manifest_path: str | Path
) -> dict[str, dict[str, Any]]:
    def read(path: str | Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    catalog = {str(row["video_id"]): row for row in read(catalog_path)}
    manifests = {str(row["video_id"]): row for row in read(manifest_path)}
    if set(catalog) != set(manifests):
        raise ValueError("catalog and multiphase manifest video grids do not align")
    result = {}
    for video_id in sorted(catalog):
        item = catalog[video_id]
        manifest = manifests[video_id]
        support = manifest.get("common_valid_support_sec")
        if not isinstance(support, list) or len(support) != 2:
            raise ValueError(f"invalid shared support for video {video_id}")
        result[video_id] = {
            "video_path": str(Path(item["video_path"]).resolve()),
            "duration_sec": float(manifest["duration_sec"]),
            "support_start_sec": float(support[0]),
            "support_stop_sec": float(support[1]),
        }
    return result


def _group_records(
    records: Sequence[OriginSignalRecord],
) -> dict[tuple[str, str, str], list[OriginSignalRecord]]:
    grouped: dict[tuple[str, str, str], list[OriginSignalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.item_key].append(record)
    for key, values in grouped.items():
        values.sort(key=lambda row: row.origin_id)
        if [row.origin_id for row in values] != list(range(5)):
            raise ValueError(f"RC12 requires exact five-origin scout grid: {key}")
    return dict(grouped)


def build_decision_specs(
    records: Sequence[OriginSignalRecord],
    video_contracts: Mapping[str, Mapping[str, Any]],
    *,
    config: RC12Config | None = None,
) -> list[dict[str, Any]]:
    """Build target-only decisions; no scout actual PTS enter the output."""

    resolved = config or RC12Config()
    rows: list[dict[str, Any]] = []
    for item_key, origins in sorted(_group_records(records).items()):
        dataset, video_id, question_id = item_key
        contract = video_contracts[video_id]
        support_start = float(contract["support_start_sec"])
        support_stop = float(contract["support_stop_sec"])
        uniform_lattice, uniform_indices = select_canonical_uniform_targets(
            origins[0].timestamps_sec,
            support_start_sec=support_start,
            support_stop_sec=support_stop,
            config=resolved,
        )
        uniform_targets = uniform_lattice[uniform_indices]
        for record in origins:
            decision = select_rc12_targets(
                record.timestamps_sec,
                record.relevance_scores,
                support_start_sec=support_start,
                support_stop_sec=support_stop,
                config=resolved,
            )
            if not np.array_equal(decision.lattice_timestamps_sec, uniform_lattice):
                raise RuntimeError("RC12 and uniform canonical lattices do not align")
            common = {
                "schema_version": 1,
                "dataset": dataset,
                "video_id": video_id,
                "question_id": question_id,
                "origin_id": int(record.origin_id),
                "origin_sec": float(record.origin_sec),
                "video_path": contract["video_path"],
                "support_start_sec": support_start,
                "support_stop_sec": support_stop,
                "lattice_timestamps_sec": uniform_lattice.astype(float).tolist(),
                "decoder_contract": {
                    "backend": "pyav_sequential_nearest_pts",
                    "fresh_source_decode": True,
                    "prohibit_scout_actual_pts_remap": True,
                    "midpoint_tie": "earlier_pts",
                    "deduplicate_by": "decoded_frame_index",
                    "dynamic_repair_priority": True,
                },
                "record_metadata": _decision_record_metadata(record),
            }
            rows.append(
                {
                    **common,
                    "method": "canonical_uniform",
                    "target_indices": uniform_indices.astype(int).tolist(),
                    "target_timestamps_sec": uniform_targets.astype(float).tolist(),
                    "quantized_scores": np.zeros(uniform_lattice.size).tolist(),
                    "decision_metadata": {
                        "decision_signal": "none_uniform_control",
                        "ti_dwt_role": "not_used",
                        "phase_effect_claim": False,
                    },
                }
            )
            rows.append(
                {
                    **common,
                    "method": "phasefuse_rc12",
                    "target_indices": decision.selected_indices.astype(int).tolist(),
                    "target_timestamps_sec": decision.target_timestamps_sec.astype(
                        float
                    ).tolist(),
                    "quantized_scores": decision.quantized_scores.astype(
                        float
                    ).tolist(),
                    "decision_metadata": decision.to_dict(include_arrays=False),
                }
            )
    return rows


def save_decision_specs(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_jsonl(path, rows)


__all__ = [
    "METHODS",
    "build_decision_specs",
    "load_video_contracts",
    "save_decision_specs",
]
