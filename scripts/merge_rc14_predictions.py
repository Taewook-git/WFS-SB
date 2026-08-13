#!/usr/bin/env python3
"""Construct a strict RC14-vs-canonical grid with immutable uniform reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

FIELDS = (
    "dataset",
    "video_id",
    "question_id",
    "origin_id",
    "method",
    "prediction",
    "gold",
)


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row["dataset"]),
        str(row["video_id"]),
        str(row["question_id"]),
        int(row["origin_id"]),
        str(row["method"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--reference-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "validated":
        raise SystemExit("RC14 validation is not terminal PASS")
    treatment = rows(args.treatment)
    reference = [
        row for row in rows(args.reference) if row.get("method") == "canonical_uniform"
    ]
    if len(treatment) != 300 or {row.get("method") for row in treatment} != {
        "phasefuse_rc14"
    }:
        raise SystemExit("RC14 predictions are not one exact 300-row treatment grid")
    if len(reference) != 300:
        raise SystemExit("canonical-uniform reference is not exactly 300 rows")
    merged = reference + treatment
    if any(set(row) != set(FIELDS) for row in merged):
        raise SystemExit("prediction row schema drift")
    mapping = {identity(row): row for row in merged}
    if len(mapping) != 600:
        raise SystemExit("duplicate RC14/canonical prediction identity")
    expected = {
        (dataset, video, question, origin, method)
        for dataset, video, question, origin, _ in mapping
        for method in ("canonical_uniform", "phasefuse_rc14")
    }
    if set(mapping) != expected or len({key[:3] for key in mapping}) != 60:
        raise SystemExit("merged RC14/canonical prediction grid is not rectangular")
    for item in {key[:4] for key in mapping}:
        baseline = mapping[(*item, "canonical_uniform")]
        treatment_row = mapping[(*item, "phasefuse_rc14")]
        if baseline["gold"] != treatment_row["gold"]:
            raise SystemExit(f"paired gold-answer drift: {item}")
    merged.sort(key=identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged))
    os.replace(temporary, args.output)
    provenance = {
        "schema_version": 1,
        "status": "validated",
        "num_rows": 600,
        "methods": ["canonical_uniform", "phasefuse_rc14"],
        "canonical_uniform_rows_reused": 300,
        "phasefuse_rc14_rows_fresh": 300,
        "treatment_predictions": str(args.treatment.resolve()),
        "treatment_predictions_sha256": sha(args.treatment),
        "reference_predictions": str(args.reference.resolve()),
        "reference_predictions_sha256": sha(args.reference),
        "validation_sha256": sha(args.validation),
        "reference_validation_sha256": sha(args.reference_validation),
        "merged_predictions_sha256": sha(args.output),
    }
    tmp = args.provenance_output.with_suffix(args.provenance_output.suffix + ".tmp")
    tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, args.provenance_output)
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
