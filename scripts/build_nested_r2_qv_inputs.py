#!/usr/bin/env python3
"""Build QV scout manifests/catalog and a content-bound source inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from phase_stable.qv_blind import load_blind_qv_query_videos
from phase_stable.sampling import build_sampling_manifest, write_manifests_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--num-origins", type=int, required=True)
    parser.add_argument("--sample-fps", type=float, required=True)
    args = parser.parse_args()
    if (args.master_seed, args.num_origins, args.sample_fps) != (20260810, 5, 1.0):
        raise ValueError(
            "QV Stage-0 scout contract is frozen to seed20260810/5 origins/1Hz"
        )
    args.manifests.parent.mkdir(parents=True, exist_ok=True)
    videos = load_blind_qv_query_videos(args.query_manifest, args.dataset_root)
    manifests = [
        build_sampling_manifest(
            video.video_id,
            float(video.duration_sec),
            master_seed=args.master_seed,
            num_origins=args.num_origins,
            sample_fps=args.sample_fps,
        )
        for video in videos
    ]
    write_manifests_jsonl(args.manifests, manifests)
    catalog_rows = []
    source_rows = []
    for video, manifest in zip(videos, manifests):
        source = video.video_path.resolve()
        if source != (args.dataset_root.resolve() / "videos" / f"{video.video_id}.mp4"):
            raise ValueError("source path is outside the frozen QV videos root")
        if not source.is_file() or source.stat().st_size <= 0:
            raise FileNotFoundError(source)
        source_id = video.video_id.rsplit("_", 2)[0]
        catalog_rows.append(
            {
                "dataset": "qvhighlights",
                "video_id": video.video_id,
                "video_path": str(source),
                "duration_sec": manifest.duration_sec,
                "num_questions": 1,
                "question_ids": [video.queries[0].question_id],
            }
        )
        source_rows.append(
            {
                "video_id": video.video_id,
                "question_id": video.queries[0].question_id,
                "source_id": source_id,
                "video_path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    write_jsonl(args.catalog, catalog_rows)
    payload = {
        "schema_version": 1,
        "status": "frozen_qv_source_inventory",
        "videos_root": str(args.dataset_root.resolve() / "videos"),
        "num_videos": 100,
        "videos": source_rows,
    }
    temporary = args.source_inventory.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.source_inventory)
    print("Wrote frozen QV scout manifests/catalog/source inventory")


if __name__ == "__main__":
    main()
