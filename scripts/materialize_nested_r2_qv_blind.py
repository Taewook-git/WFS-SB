#!/usr/bin/env python3
"""Materialize the frozen QV holdout with a pre-outcome label firewall.

In one materialization stage, only ``vid``, ``qid``, ``query``, and
``duration`` are allowed into selector-facing artifacts. Ground-truth
highlight fields are stored in a hash-committed, non-selector-facing sidecar.
This is an audited sequencing and integrity firewall, not cryptographic
blinding. Plaintext labels are copied into the analysis area only by ``join``
after a valid decision/trace seal exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EXPECTED_ANNOTATION_SHA256 = (
    "f668a1eaea156ec5315e14718999cea043a8cf948d3cafbd8e8d655318c3cd02"
)
EXPECTED_COHORT_SHA256 = (
    "349f450a169fc1160429738f30028f38b6401b5f33c3760d02d70b3111ccb3b4"
)
EXPECTED_PREREG_SHA256 = (
    "694522db81e5f9db5d244f713c03fd26844c5f680828f8fc3df0b4ed992c9cff"
)
LABEL_FIELDS = ("relevant_windows", "relevant_clip_ids", "saliency_scores")
QUERY_FIELDS = ("vid", "qid", "query", "duration")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    _atomic_bytes(path, encoded)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_cohort(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != EXPECTED_COHORT_SHA256:
        raise ValueError("QV cohort SHA-256 does not match the frozen protocol")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError(f"invalid cohort TSV row {line_number}")
        rank, source_id, annotation_index, vid, qid = fields
        row = {
            "rank": int(rank),
            "source_id": source_id,
            "annotation_row_index": int(annotation_index),
            "vid": vid,
            "qid": str(qid),
        }
        if row["rank"] != len(rows) or len(source_id) != 11:
            raise ValueError(f"invalid frozen cohort identity at row {line_number}")
        if str(vid).rsplit("_", 2)[0] != source_id:
            raise ValueError(f"cohort source_id/vid mismatch at row {line_number}")
        rows.append(row)
    if len(rows) != 100 or len({row["source_id"] for row in rows}) != 100:
        raise ValueError("QV cohort must contain exactly 100 unique source IDs")
    return rows


def _load_annotation_row(
    path: Path, wanted: Mapping[int, Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    if sha256_file(path) != EXPECTED_ANNOTATION_SHA256:
        raise ValueError("QV annotation SHA-256 does not match the frozen protocol")
    selected: dict[int, dict[str, Any]] = {}
    physical_index = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            physical_index += 1
            if physical_index not in wanted:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"annotation row {physical_index} is not an object")
            identity = wanted[physical_index]
            if (
                str(value.get("vid")) != identity["vid"]
                or str(value.get("qid")) != identity["qid"]
            ):
                raise ValueError(
                    f"annotation/cohort identity drift at row {physical_index}"
                )
            selected[physical_index] = value
    if set(selected) != set(wanted):
        raise ValueError("frozen QV annotation row coverage is incomplete")
    return selected


def materialize(args: argparse.Namespace) -> None:
    paths = {
        path.resolve()
        for path in (
            args.query_manifest,
            args.sealed_labels,
            args.output_manifest,
        )
    }
    if len(paths) != 3:
        raise ValueError("materialization output paths must be distinct")
    if sha256_file(args.prereg) != EXPECTED_PREREG_SHA256:
        raise ValueError("QV preregistration SHA-256 is not the execution freeze")
    cohort = _read_cohort(args.cohort)
    wanted = {row["annotation_row_index"]: row for row in cohort}
    annotation = _load_annotation_row(args.annotation, wanted)
    query_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for identity in cohort:
        raw = annotation[identity["annotation_row_index"]]
        missing_query = [name for name in QUERY_FIELDS if name not in raw]
        missing_labels = [name for name in LABEL_FIELDS if name not in raw]
        if missing_query or missing_labels:
            raise ValueError(
                f"annotation row {identity['annotation_row_index']} missing fields: "
                f"{missing_query + missing_labels}"
            )
        query_rows.append(
            {
                "video_id": str(raw["vid"]),
                "query_id": str(raw["qid"]),
                "query": str(raw["query"]),
                "duration": float(raw["duration"]),
                "metadata": {
                    "annotation_row_index": identity["annotation_row_index"],
                    "source_id": identity["source_id"],
                    "label_firewall": "query_only_before_blind_seal",
                },
            }
        )
        label_rows.append(
            {
                "source_id": identity["source_id"],
                "annotation_row_index": identity["annotation_row_index"],
                "vid": str(raw["vid"]),
                "qid": str(raw["qid"]),
                **{name: raw[name] for name in LABEL_FIELDS},
            }
        )

    query_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in query_rows
    )
    label_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in label_rows
    )
    _atomic_bytes(args.query_manifest, query_bytes)
    _atomic_bytes(args.sealed_labels, label_bytes)
    manifest = {
        "schema_version": 1,
        "status": "labels_hash_committed_query_only_materialized",
        "query_role": "selector input; query text is required for BLIP2 relevance",
        "label_role": "non-selector evaluation-only; join forbidden before blind artifact seal",
        "num_rows": 100,
        "annotation_sha256": EXPECTED_ANNOTATION_SHA256,
        "cohort_sha256": EXPECTED_COHORT_SHA256,
        "preregistration_sha256": EXPECTED_PREREG_SHA256,
        "query_manifest_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "sealed_labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "sealed_label_plaintext_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "label_firewall": "hash_committed_sidecar_programmatic_sequencing",
        "cryptographic_blinding": False,
        "query_fields": list(QUERY_FIELDS),
        "prohibited_plaintext_fields_before_seal": list(LABEL_FIELDS),
    }
    _atomic_json(args.output_manifest, manifest)


def join(args: argparse.Namespace) -> None:
    paths = {
        path.resolve()
        for path in (
            args.sealed_labels,
            args.blind_seal,
            args.decisions,
            args.traces,
            args.source_video_bundle,
            args.materialization_manifest,
            args.query_manifest,
            args.cohort,
            args.output_labels,
            args.join_manifest,
        )
    }
    if len(paths) != 10:
        raise ValueError("join input/output paths must be distinct")
    seal = _read_json(args.blind_seal)
    materialization = _read_json(args.materialization_manifest)
    required = {
        "status": "sealed_before_label_join",
        "decisions_sha256": sha256_file(args.decisions),
        "traces_sha256": sha256_file(args.traces),
        "source_video_bundle_sha256": sha256_file(args.source_video_bundle),
        "blind_materialization_sha256": sha256_file(args.materialization_manifest),
        "query_manifest_sha256": sha256_file(args.query_manifest),
        "sealed_labels_sha256": sha256_file(args.sealed_labels),
    }
    for name, expected in required.items():
        if seal.get(name) != expected:
            raise ValueError(f"blind seal does not bind current {name}")
    if seal.get("preregistration_sha256") != EXPECTED_PREREG_SHA256:
        raise ValueError("blind seal preregistration binding drift")
    if seal.get("cohort_sha256") != EXPECTED_COHORT_SHA256:
        raise ValueError("blind seal cohort binding drift")
    if (
        materialization.get("query_manifest_sha256")
        != required["query_manifest_sha256"]
    ):
        raise ValueError("materialization/query manifest binding drift")
    if materialization.get("sealed_labels_sha256") != required["sealed_labels_sha256"]:
        raise ValueError("materialization/sealed-label binding drift")
    plaintext = args.sealed_labels.read_bytes()
    if not plaintext:
        raise ValueError("sealed label artifact is empty")
    rows = [json.loads(line) for line in plaintext.decode("utf-8").splitlines() if line]
    if len(rows) != 100 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("sealed label grid is not the frozen 100-source cohort")
    if hashlib.sha256(plaintext).hexdigest() != materialization.get(
        "sealed_label_plaintext_sha256"
    ):
        raise ValueError("sealed label plaintext hash drift")
    cohort = _read_cohort(args.cohort)
    observed = [
        (
            str(row.get("source_id")),
            int(row.get("annotation_row_index", -1)),
            str(row.get("vid")),
            str(row.get("qid")),
        )
        for row in rows
    ]
    expected = [
        (
            row["source_id"],
            row["annotation_row_index"],
            row["vid"],
            row["qid"],
        )
        for row in cohort
    ]
    if observed != expected:
        raise ValueError("sealed labels do not preserve exact frozen cohort order")
    _atomic_bytes(args.output_labels, plaintext)
    _atomic_json(
        args.join_manifest,
        {
            "schema_version": 1,
            "status": "labels_joined_after_blind_seal",
            "num_rows": 100,
            "blind_seal_sha256": sha256_file(args.blind_seal),
            "decisions_sha256": required["decisions_sha256"],
            "traces_sha256": required["traces_sha256"],
            "output_labels_sha256": hashlib.sha256(plaintext).hexdigest(),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    first = sub.add_parser("materialize")
    first.add_argument("--annotation", type=Path, required=True)
    first.add_argument("--cohort", type=Path, required=True)
    first.add_argument("--prereg", type=Path, required=True)
    first.add_argument("--query-manifest", type=Path, required=True)
    first.add_argument("--sealed-labels", type=Path, required=True)
    first.add_argument("--output-manifest", type=Path, required=True)
    first.set_defaults(handler=materialize)
    second = sub.add_parser("join")
    second.add_argument("--sealed-labels", type=Path, required=True)
    second.add_argument("--blind-seal", type=Path, required=True)
    second.add_argument("--decisions", type=Path, required=True)
    second.add_argument("--traces", type=Path, required=True)
    second.add_argument("--source-video-bundle", type=Path, required=True)
    second.add_argument("--materialization-manifest", type=Path, required=True)
    second.add_argument("--query-manifest", type=Path, required=True)
    second.add_argument("--cohort", type=Path, required=True)
    second.add_argument("--output-labels", type=Path, required=True)
    second.add_argument("--join-manifest", type=Path, required=True)
    second.set_defaults(handler=join)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
