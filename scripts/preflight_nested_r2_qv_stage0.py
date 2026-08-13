#!/usr/bin/env python3
"""Small fail-closed preflight helpers for the frozen QV Stage-0 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def numeric_gpu_pids(lines: list[str]) -> list[int]:
    """Return only numeric first CSV fields from nvidia-smi compute rows."""

    result = []
    for line in lines:
        first_field = line.split(",", 1)[0].strip()
        if first_field.isascii() and first_field.isdigit():
            result.append(int(first_field))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-process-file", type=Path, required=True)
    args = parser.parse_args()
    pids = numeric_gpu_pids(
        args.gpu_process_file.read_text(encoding="utf-8").splitlines()
    )
    print(json.dumps({"numeric_gpu_pids": pids}, sort_keys=True))
    raise SystemExit(0 if pids else 1)


if __name__ == "__main__":
    main()
