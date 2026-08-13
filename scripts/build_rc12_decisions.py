#!/usr/bin/env python3
"""Build frozen PhaseFuse-RC12 and canonical-uniform target specifications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase_stable.artifacts import read_signal_records
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
    args = parser.parse_args()
    config = RC12Config()
    records = read_signal_records(args.signals)
    contracts = load_video_contracts(args.catalog, args.manifests)
    rows = build_decision_specs(records, contracts, config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "decisions.jsonl"
    save_decision_specs(output, rows)
    summary = {
        "schema_version": 1,
        "methods": ["canonical_uniform", "phasefuse_rc12"],
        "num_rows": len(rows),
        "num_items": len(rows) // 10,
        "num_origins": 5,
        "frame_budget": 16,
        "config": config.__dict__,
        "fresh_exact_decode_required": True,
        "ti_dwt_role": "diagnostic_only",
    }
    (args.output_dir / "decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_reproducibility_manifests(
        args.output_dir,
        command="build-rc12-decisions",
        config=summary,
        input_paths=(args.signals, args.catalog, args.manifests),
        extra={"decisions": str(output.resolve())},
        repo_root=Path(__file__).parents[1],
    )
    print(f"Wrote {len(rows)} frozen target decisions to {output}")


if __name__ == "__main__":
    main()
