#!/usr/bin/env python3
"""Preprocess the frozen query-only QV cohort without loading GT labels."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from phase_stable.benchmarks import preprocess_benchmark_to_jsonl
from phase_stable.cli import _load_feature_extractor
from phase_stable.qv_blind import load_blind_qv_query_videos
from phase_stable.sampling import read_manifests_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frame-buffer-size", type=int, default=256)
    parser.add_argument("--checkpoint-tree-sha256", required=True)
    args = parser.parse_args()
    videos = load_blind_qv_query_videos(args.query_manifest, args.dataset_root)
    manifests = read_manifests_jsonl(args.manifests)
    extractor, resolved_model, device = _load_feature_extractor(
        "blip2", args.model_path, args.device
    )
    image = importlib.import_module("PIL.Image")
    destination = preprocess_benchmark_to_jsonl(
        videos,
        manifests,
        extractor=extractor,
        output_dir=args.output_dir,
        signal_jsonl=args.signal_jsonl,
        batch_size=args.batch_size,
        frame_buffer_size=args.frame_buffer_size,
        frame_adapter=image.fromarray,
        record_metadata={
            "feature_model": "blip2",
            "feature_model_revision": resolved_model,
            "feature_checkpoint_content_tree_sha256": args.checkpoint_tree_sha256,
            "feature_device": device,
            "label_firewall": "query_only_before_blind_seal",
        },
    )
    print(f"Wrote blind QV signals to {destination}")


if __name__ == "__main__":
    main()
