#!/usr/bin/env python3
"""Materialize the frozen downstream RC12 safety-gate verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def evaluate_gate(summary: dict) -> dict:
    stability = summary.get("stability")
    if not isinstance(stability, dict):
        raise TypeError("downstream summary is missing stability")
    if (
        stability.get("baseline_method") != "canonical_uniform"
        or stability.get("treatment_method") != "phasefuse_rc12"
    ):
        raise ValueError("downstream arm direction is not RC12 minus canonical uniform")
    comparison = stability.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("effect_definition") != (
        "treatment - baseline"
    ):
        raise ValueError("downstream comparison effect direction is invalid")
    order = comparison.get("effect_order")
    low = comparison.get("ci_low")
    high = comparison.get("ci_high")
    estimate = comparison.get("estimate")
    if not all(isinstance(values, list) for values in (order, low, high, estimate)):
        raise ValueError("downstream comparison vectors are missing")
    if len({len(order), len(low), len(high), len(estimate)}) != 1:
        raise ValueError("downstream comparison vectors do not align")
    accuracy_index = order.index("delta_mean_accuracy")
    pad_index = order.index("delta_pairwise_answer_disagreement")
    accuracy_pass = float(low[accuracy_index]) > -0.03
    pad_pass = float(high[pad_index]) < 0.03
    return {
        "schema_version": 1,
        "status": "pass" if accuracy_pass and pad_pass else "fail",
        "effect_definition": "phasefuse_rc12 - canonical_uniform",
        "accuracy_noninferiority": {
            "metric": "delta_mean_accuracy",
            "estimate": float(estimate[accuracy_index]),
            "ci_low": float(low[accuracy_index]),
            "ci_high": float(high[accuracy_index]),
            "strict_threshold": -0.03,
            "criterion": "ci_low > threshold",
            "pass": accuracy_pass,
        },
        "pad_safety": {
            "metric": "delta_pairwise_answer_disagreement",
            "estimate": float(estimate[pad_index]),
            "ci_low": float(low[pad_index]),
            "ci_high": float(high[pad_index]),
            "strict_threshold": 0.03,
            "criterion": "ci_high < threshold",
            "pass": pad_pass,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = evaluate_gate(summary)
    result["downstream_summary_sha256"] = hashlib.sha256(
        args.summary.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("frozen RC12 downstream safety gate failed")


if __name__ == "__main__":
    main()
