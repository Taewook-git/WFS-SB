#!/usr/bin/env python3
"""Build frozen nested-R2 QV decisions from query-only scout signals."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from phase_stable.artifacts import read_signal_records
from phase_stable.rc12_experiment import build_decision_specs, save_decision_specs


def read_rows(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_signal_records(args.signals)
    catalog = {str(row["video_id"]): row for row in read_rows(args.catalog)}
    grouped = defaultdict(list)
    for record in records:
        grouped[record.video_id].append(record)
    if len(grouped) != 100 or any(len(rows) != 5 for rows in grouped.values()):
        raise ValueError("QV decision input must be exact 100 videos x five origins")
    contracts = {}
    for video_id, rows in grouped.items():
        support_start = max(float(row.timestamps_sec[0]) for row in rows)
        support_stop = min(float(row.timestamps_sec[-1]) for row in rows)
        if support_stop <= support_start:
            raise ValueError(f"empty common scout support: {video_id}")
        contracts[video_id] = {
            "video_path": str(Path(catalog[video_id]["video_path"]).resolve()),
            "duration_sec": float(catalog[video_id]["duration_sec"]),
            "support_start_sec": support_start,
            "support_stop_sec": support_stop,
        }
    rows = build_decision_specs(
        records,
        contracts,
        treatment_method="phasefuse_nested_r2",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_decision_specs(args.output, rows)
    print(f"Wrote {len(rows)} frozen QV nested-R2 decisions")


if __name__ == "__main__":
    main()
