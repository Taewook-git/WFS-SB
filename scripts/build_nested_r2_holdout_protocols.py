#!/usr/bin/env python3
"""Deterministically reconstruct the frozen nested-R2 holdout cohort files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

N30_SALT = (
    "phasefuse-v2-holdout-v1|repo=3ec7dc57a21bcacec4eb40be6138bb2d5142c01a|"
    "annotation=cc02294caaebdc6f07d9a850c84d6cda5978609c1884c91adc3dbcec90deed44"
)
N30_DOMAIN_ORDER = (
    "Knowledge",
    "Film & Television",
    "Sports Competition",
    "Artistic Performance",
    "Life Record",
    "Multilingual",
)
N30_QUOTAS = {
    "short": (3, 2, 1, 1, 2, 1),
    "medium": (3, 1, 2, 2, 2, 0),
    "long": (3, 2, 1, 1, 2, 1),
}
N30_INDICES = (
    36, 65, 66, 108, 117, 164, 201, 263, 278, 293,
    309, 376, 379, 392, 443, 445, 480, 514, 521, 542,
    604, 622, 650, 705, 707, 743, 807, 850, 864, 890,
)
N30_SHA256 = "caddfd261119366048e1180420987d7b1f2b4b3327cc22fb1398d1a3c21f57cd"

QV_SALT = (
    "phasefuse-nested-r2-qv-holdout-v1|"
    "config=d3512cf41506b214c86cce72f607390184cecfa7b9bd4750adcc4122788e61e9|"
    "annotation=f668a1eaea156ec5315e14718999cea043a8cf948d3cafbd8e8d655318c3cd02|"
    "dev_manifest=5b89434552f242502a5f00636f8aabd9186e8d6ab5c5de98d4372fcbfe4469ad"
)
QV_COHORT_SHA256 = "349f450a169fc1160429738f30028f38b6401b5f33c3760d02d70b3111ccb3b4"
QV_EXCLUSIONS_SHA256 = "6fd1db129754094b10aef4375c1e1364150f501d7af1e1fa910ae0bafcd497c1"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"row {number} is not an object: {path}")
        rows.append(row)
    return rows


def build_n30(annotation: Path, data_root: Path) -> bytes:
    rows = json.loads(annotation.read_text(encoding="utf-8"))
    unique: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        video_id = str(row["video_id"])
        if video_id not in seen:
            seen.add(video_id)
            unique.append((len(unique), row))
    if len(unique) != 900:
        raise ValueError("VideoMME input is not exactly 900 unique videos")
    dev_ids = {f"{value:03d}" for value in range(1, 21)}
    candidates = [
        (index, row)
        for index, row in unique
        if str(row["video_id"]) not in dev_ids
        and (data_root / f"{row['videoID']}.mp4").is_file()
    ]
    selected: list[tuple[int, dict[str, Any]]] = []
    for duration in ("short", "medium", "long"):
        for domain, quota in zip(
            N30_DOMAIN_ORDER, N30_QUOTAS[duration], strict=True
        ):
            stratum = [
                item
                for item in candidates
                if item[1]["duration"] == duration and item[1]["domain"] == domain
            ]
            stratum.sort(
                key=lambda item: hashlib.sha256(
                    f"{N30_SALT}|{item[1]['video_id']}".encode("utf-8")
                ).hexdigest()
            )
            if len(stratum) < quota:
                raise ValueError(f"insufficient n30 candidates: {duration}/{domain}")
            selected.extend(stratum[:quota])
    selected.sort(key=lambda item: item[0])
    if tuple(index for index, _ in selected) != N30_INDICES:
        raise ValueError("n30 reconstruction differs from the frozen indices")
    payload = "".join(
        f"{index}\t{row['video_id']}\t{row['videoID']}\t{row['duration']}\t{row['domain']}\n"
        for index, row in selected
    ).encode("utf-8")
    if len(payload) != 1215 or _sha256(payload) != N30_SHA256:
        raise ValueError("n30 canonical serialization drift")
    return payload


def qv_source_id(video_id: Any) -> str:
    source_id = str(video_id).rsplit("_", 2)[0]
    if len(source_id) != 11:
        raise ValueError(f"invalid QV source ID: {video_id!r}")
    return source_id


def build_qv(annotation: Path, dev_manifest: Path) -> tuple[bytes, bytes]:
    rows = _read_jsonl(annotation)
    dev_rows = _read_jsonl(dev_manifest)
    if len(rows) != 1550 or len(dev_rows) != 100:
        raise ValueError("QV annotation/development manifest row-count drift")
    excluded = sorted({qv_source_id(row["video_id"]) for row in dev_rows})
    if len(excluded) != 88:
        raise ValueError("QV development exclusion count drift")
    excluded_set = set(excluded)
    best: dict[str, tuple[tuple[str, int], dict[str, Any]]] = {}
    for row_index, row in enumerate(rows):
        source_id = qv_source_id(row["vid"])
        if source_id in excluded_set:
            continue
        row_hash = hashlib.sha256(
            f"{QV_SALT}|row|{source_id}|{row['vid']}|{row['qid']}".encode("utf-8")
        ).hexdigest()
        candidate = ((row_hash, row_index), row)
        if source_id not in best or candidate[0] < best[source_id][0]:
            best[source_id] = candidate
    ranked = sorted(
        best.items(),
        key=lambda item: (
            hashlib.sha256(
                f"{QV_SALT}|source|{item[0]}".encode("utf-8")
            ).hexdigest(),
            item[0],
        ),
    )[:100]
    cohort = "".join(
        f"{rank}\t{source_id}\t{best[source_id][0][1]}\t{row['vid']}\t{row['qid']}\n"
        for rank, (source_id, (_, row)) in enumerate(ranked)
    ).encode("utf-8")
    exclusions = "".join(f"{source_id}\n" for source_id in excluded).encode("utf-8")
    if len(cohort) != 4777 or _sha256(cohort) != QV_COHORT_SHA256:
        raise ValueError("QV cohort canonical serialization drift")
    if len(exclusions) != 1056 or _sha256(exclusions) != QV_EXCLUSIONS_SHA256:
        raise ValueError("QV exclusion canonical serialization drift")
    return cohort, exclusions


def _write_or_check(path: Path, payload: bytes, *, check: bool) -> None:
    if check:
        if path.read_bytes() != payload:
            raise ValueError(f"committed protocol file differs from reconstruction: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videomme-annotation", type=Path, required=True)
    parser.add_argument("--videomme-data-root", type=Path, required=True)
    parser.add_argument("--qv-annotation", type=Path, required=True)
    parser.add_argument("--qv-dev-manifest", type=Path, required=True)
    parser.add_argument("--n30-output", type=Path, required=True)
    parser.add_argument("--qv-output", type=Path, required=True)
    parser.add_argument("--qv-exclusions-output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    n30 = build_n30(args.videomme_annotation, args.videomme_data_root)
    qv, exclusions = build_qv(args.qv_annotation, args.qv_dev_manifest)
    _write_or_check(args.n30_output, n30, check=args.check)
    _write_or_check(args.qv_output, qv, check=args.check)
    _write_or_check(args.qv_exclusions_output, exclusions, check=args.check)
    print(
        json.dumps(
            {
                "status": "validated" if args.check else "written",
                "n30": {"bytes": len(n30), "sha256": _sha256(n30)},
                "qv": {"bytes": len(qv), "sha256": _sha256(qv)},
                "qv_exclusions": {
                    "bytes": len(exclusions),
                    "sha256": _sha256(exclusions),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
