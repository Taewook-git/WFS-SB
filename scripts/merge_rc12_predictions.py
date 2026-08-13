#!/usr/bin/env python3
"""Validate the 10-cell RC12 grid and add strictly aligned context arms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

FIELDS = {
    "dataset",
    "video_id",
    "question_id",
    "origin_id",
    "method",
    "prediction",
    "gold",
}
PRIMARY_METHODS = ("canonical_uniform", "phasefuse_rc12")


def read(path):
    result = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if set(row) != FIELDS:
            raise SystemExit(f"invalid prediction schema {path}:{n}")
        result.append(row)
    return result


def key(row):
    return (
        row["dataset"],
        row["video_id"],
        row["question_id"],
        int(row["origin_id"]),
    )


def validate_arm(rows, method):
    keys = [key(row) for row in rows]
    if (
        len(rows) != 300
        or {row["method"] for row in rows} != {method}
        or len(set(keys)) != 300
        or {item[3] for item in keys} != set(range(5))
    ):
        raise SystemExit(f"invalid exact prediction grid for {method}")
    item_origins = {}
    for dataset, video, question, origin in keys:
        item_origins.setdefault((dataset, video, question), set()).add(origin)
    if len(item_origins) != 60 or any(
        origins != set(range(5)) for origins in item_origins.values()
    ):
        raise SystemExit(f"non-rectangular prediction grid for {method}")


def write(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(temporary, output)


def context_arm(spec):
    if "=" not in spec or "@" not in spec:
        raise SystemExit("--context-arm must be OUTPUT_METHOD=SOURCE_METHOD@PATH")
    output_method, source = spec.split("=", 1)
    source_method, path = source.split("@", 1)
    if not output_method or not source_method or not path:
        raise SystemExit("--context-arm must be OUTPUT_METHOD=SOURCE_METHOD@PATH")
    arm = [
        {**row, "method": output_method}
        for row in read(path)
        if row["method"] == source_method
    ]
    return output_method, arm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--context-arm", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-output", required=True)
    args = parser.parse_args()

    raw_primary = read(args.primary)
    primary = []
    for method in PRIMARY_METHODS:
        arm = [row for row in raw_primary if row["method"] == method]
        validate_arm(arm, method)
        primary.extend(sorted(arm, key=key))
    if len(raw_primary) != 600 or len(primary) != 600:
        raise SystemExit("primary input must contain exactly two 300-row arms")

    expected_keys = {key(row) for row in primary if row["method"] == PRIMARY_METHODS[0]}
    other_keys = {key(row) for row in primary if row["method"] == PRIMARY_METHODS[1]}
    if expected_keys != other_keys:
        raise SystemExit("primary arm cohorts do not align exactly")
    gold = {
        (row["dataset"], row["video_id"], row["question_id"]): row["gold"]
        for row in primary
    }

    context = []
    context_methods = []
    for spec in args.context_arm:
        method, arm = context_arm(spec)
        if method in {*PRIMARY_METHODS, *context_methods}:
            raise SystemExit(f"duplicate context method: {method}")
        validate_arm(arm, method)
        if {key(row) for row in arm} != expected_keys:
            raise SystemExit(f"unaligned context arm: {method}")
        context_methods.append(method)
        context.extend(sorted(arm, key=key))

    merged = primary + context
    for row in merged:
        item = (row["dataset"], row["video_id"], row["question_id"])
        if gold[item] != row["gold"]:
            raise SystemExit("gold mismatch across primary/context arms")
    write(args.output, primary)
    write(args.context_output, merged)
    print(
        f"Strict RC12 merge wrote 600 primary rows and {len(merged)} total rows "
        f"with contexts={context_methods}"
    )


if __name__ == "__main__":
    main()
