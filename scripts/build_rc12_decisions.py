#!/usr/bin/env python3
"""Build frozen canonical-adaptive and uniform target specifications."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from phase_stable.artifacts import read_signal_records
from phase_stable.canonical_nested_r2 import NestedR2Config
from phase_stable.canonical_residual import RC12Config
from phase_stable.rc12_experiment import (
    build_decision_specs,
    load_video_contracts,
    save_decision_specs,
)
from phase_stable.repro import write_reproducibility_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--treatment-method",
        choices=("phasefuse_rc12", "phasefuse_rc14", "phasefuse_nested_r2"),
        default="phasefuse_rc12",
    )
    parser.add_argument("--anchor-count", type=int, default=12)
    args = parser.parse_args()
    config = RC12Config(anchor_count=args.anchor_count)
    records = read_signal_records(args.signals)
    contracts = load_video_contracts(args.catalog, args.manifests)
    rows = build_decision_specs(
        records,
        contracts,
        config=config,
        treatment_method=args.treatment_method,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "decisions.jsonl"
    save_decision_specs(output, rows)
    summary = {
        "schema_version": 1,
        "methods": ["canonical_uniform", args.treatment_method],
        "num_rows": len(rows),
        "num_items": len(rows) // 10,
        "num_origins": 5,
        "frame_budget": 16,
        "config": (
            asdict(NestedR2Config())
            if args.treatment_method == "phasefuse_nested_r2"
            else config.__dict__
        ),
        "fresh_exact_decode_required": True,
        "ti_dwt_role": (
            "not_used"
            if args.treatment_method == "phasefuse_nested_r2"
            else "diagnostic_only"
        ),
    }
    (args.output_dir / "decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_reproducibility_manifests(
        args.output_dir,
        command=f"build-{args.treatment_method}-decisions",
        config=summary,
        input_paths=(args.signals, args.catalog, args.manifests),
        extra={"decisions": str(output.resolve())},
        repo_root=Path(__file__).parents[1],
    )
    print(f"Wrote {len(rows)} frozen target decisions to {output}")


if __name__ == "__main__":
    main()
