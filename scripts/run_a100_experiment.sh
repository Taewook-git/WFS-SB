#!/usr/bin/env bash
# One entry point: safe pull/setup -> optional data fetch -> Stage-0 -> MLLM grid -> evaluation.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CALLER_DIR="$(pwd -P)"

die() {
  printf 'run_a100_experiment: ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: bash scripts/run_a100_experiment.sh [options]

Default action: update this clean checkout, build/reuse the Python 3.10 venv,
prompt for Hugging Face login if needed, obtain the first 20 VideoMME videos,
run DWT/SWT Stage-0, run the 2 methods x 5 origins Qwen grid, merge official
parser outputs, and compute the paired downstream summary.

Core options:
  --benchmark NAME          videomme (default), mlvu, or lvb
  --dataset-root PATH       Benchmark root; defaults to repository datasets
  --questions-file PATH     Annotation override
  --run-dir PATH            Artifact root (default: artifacts/<benchmark>_stage0_20)
  --config PATH             Phase-stable YAML (default: configs/phase_stable_icassp.yaml)
  --seed N                  Sampling/bootstrap seed (default: 20260810)
  --full                    Run every annotated video (10,000 bootstraps)
  --video-count N           First N unique videos in pilot mode (default: 20)
  --no-download-data        Do not fetch missing VideoMME archive chunks
  --skip-bootstrap          Reuse an already prepared venv without Git pull/install
  --skip-mllm               Stop after keyframe export
  --include-baselines       Also run Uniform and Top-K through the MLLM
  --matched-only            Reuse completed Stage-0 and run only the equal-boundary
                            DWT/SWT counterfactual (10 new Qwen cells)

Runtime options:
  --venv-dir PATH           Venv (default: ~/.venvs/wfs-sb-a100)
  --python PATH             Host Python 3.10 for bootstrap (default: python3.10)
  --cuda-device ID          Physical GPU ID (default: 0)
  --feature-model NAME      BLIP/CLIP extractor (default: blip2)
  --feature-model-path P    Hub ID or local extractor checkpoint
  --feature-batch-size N    BLIP inference batch (default: 32 for A100 80GB)
  --frame-buffer-size N     Maximum decoded RGB frames in RAM (default: 256)
  --qwen-checkpoint P       Hub ID/local Qwen path
  --qwen-max-pixels N       Per-frame Qwen limit (default: 200704; 40GB-safe)
  --num-origins N           Sampling origins (default: 5)
  --sample-fps F            Candidate sampling FPS (default: 1.0)
  --frame-budget N          Selected frames per question (default: 16)
  --run-matched             Also compute fixed-cardinality boundary metrics
  --matched-count N         Boundary count for matched analysis/rerun (default: 4)
  --mllm-limit N            Smoke-test limit passed to lmms-eval
  --force-step NAME         Force a Stage-0 step; repeatable
  --force-mllm              Re-run every MLLM grid cell
  -h, --help                Show this help

HF_TOKEN may be exported before launch, or omitted for a hidden interactive
`hf auth login` prompt. It is never printed or forwarded to lmms-eval logs.
Every expensive stage has provenance-aware completion markers, so rerunning the
same command resumes completed work and rejects stale outputs.
EOF
}

BENCHMARK="videomme"
DATASET_ROOT=""
QUESTIONS_FILE=""
RUN_DIR=""
CONFIG_FILE="${REPO_ROOT}/configs/phase_stable_icassp.yaml"
CONFIG_EXPLICIT=0
SEED=20260810
FULL_RUN=0
VIDEO_COUNT=20
DOWNLOAD_DATA=1
RUN_BOOTSTRAP=1
RUN_MLLM=1
INCLUDE_BASELINES=0
MATCHED_ONLY=0
VENV_DIR="${WFS_VENV_DIR:-${HOME}/.venvs/wfs-sb-a100}"
HOST_PYTHON="${WFS_PYTHON:-python3.10}"
CUDA_DEVICE="0"
FEATURE_MODEL="blip2"
FEATURE_MODEL_PATH=""
FEATURE_BATCH_SIZE=32
FRAME_BUFFER_SIZE=256
QWEN_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_MAX_PIXELS=200704
NUM_ORIGINS=5
SAMPLE_FPS=1.0
FRAME_BUDGET=16
RUN_MATCHED=0
MATCHED_COUNT=4
MLLM_LIMIT=""
declare -a FORCE_STEPS=()
FORCE_MLLM=0

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    --benchmark) need_value "$@"; BENCHMARK="${2,,}"; shift 2 ;;
    --dataset-root) need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
    --questions-file) need_value "$@"; QUESTIONS_FILE="$2"; shift 2 ;;
    --run-dir) need_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG_FILE="$2"; CONFIG_EXPLICIT=1; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --full) FULL_RUN=1; shift ;;
    --video-count) need_value "$@"; VIDEO_COUNT="$2"; shift 2 ;;
    --no-download-data) DOWNLOAD_DATA=0; shift ;;
    --skip-bootstrap) RUN_BOOTSTRAP=0; shift ;;
    --skip-mllm) RUN_MLLM=0; shift ;;
    --include-baselines) INCLUDE_BASELINES=1; shift ;;
    --matched-only) MATCHED_ONLY=1; shift ;;
    --venv-dir) need_value "$@"; VENV_DIR="$2"; shift 2 ;;
    --python) need_value "$@"; HOST_PYTHON="$2"; shift 2 ;;
    --cuda-device) need_value "$@"; CUDA_DEVICE="$2"; shift 2 ;;
    --feature-model) need_value "$@"; FEATURE_MODEL="$2"; shift 2 ;;
    --feature-model-path) need_value "$@"; FEATURE_MODEL_PATH="$2"; shift 2 ;;
    --feature-batch-size) need_value "$@"; FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --frame-buffer-size) need_value "$@"; FRAME_BUFFER_SIZE="$2"; shift 2 ;;
    --qwen-checkpoint) need_value "$@"; QWEN_CHECKPOINT="$2"; shift 2 ;;
    --qwen-max-pixels) need_value "$@"; QWEN_MAX_PIXELS="$2"; shift 2 ;;
    --num-origins) need_value "$@"; NUM_ORIGINS="$2"; shift 2 ;;
    --sample-fps) need_value "$@"; SAMPLE_FPS="$2"; shift 2 ;;
    --frame-budget) need_value "$@"; FRAME_BUDGET="$2"; shift 2 ;;
    --run-matched) RUN_MATCHED=1; shift ;;
    --matched-count) need_value "$@"; MATCHED_COUNT="$2"; RUN_MATCHED=1; shift 2 ;;
    --mllm-limit) need_value "$@"; MLLM_LIMIT="$2"; shift 2 ;;
    --force-step) need_value "$@"; FORCE_STEPS+=("$2"); shift 2 ;;
    --force-mllm) FORCE_MLLM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

if ((MATCHED_ONLY)); then
  ((RUN_MLLM == 1)) || die "--matched-only cannot be combined with --skip-mllm"
  ((${#FORCE_STEPS[@]} == 0)) || \
    die "--matched-only skips Stage-0 and cannot use --force-step"
  ((INCLUDE_BASELINES == 0)) || \
    die "--matched-only compares only dwt_matched and swt_matched"
fi

case "$BENCHMARK" in
  videomme)
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/videomme}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/videomme_json_file.json}"
    RAW_SUBDIR="data"
    ;;
  mlvu)
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/mlvu}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/mlvu_dev.json}"
    RAW_SUBDIR="video"
    ;;
  lvb|longvideobench)
    BENCHMARK="lvb"
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/longvideobench}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/lvb_val.json}"
    RAW_SUBDIR="videos"
    ;;
  *) die "--benchmark must be videomme, mlvu, or lvb" ;;
esac

command -v realpath >/dev/null 2>&1 || die "realpath is required"
if [[ "$DATASET_ROOT" != /* ]]; then DATASET_ROOT="$CALLER_DIR/$DATASET_ROOT"; fi
if [[ "$QUESTIONS_FILE" != /* ]]; then QUESTIONS_FILE="$CALLER_DIR/$QUESTIONS_FILE"; fi
if [[ "$VENV_DIR" != /* ]]; then VENV_DIR="$CALLER_DIR/$VENV_DIR"; fi
if [[ "$CONFIG_FILE" != /* ]]; then CONFIG_FILE="$CALLER_DIR/$CONFIG_FILE"; fi
DATASET_ROOT="$(realpath -m -- "$DATASET_ROOT")"
QUESTIONS_FILE="$(realpath -m -- "$QUESTIONS_FILE")"
VENV_DIR="$(realpath -m -- "$VENV_DIR")"
CONFIG_FILE="$(realpath -m -- "$CONFIG_FILE")"
if [[ -n "$FEATURE_MODEL_PATH" && -e "$FEATURE_MODEL_PATH" ]]; then
  FEATURE_MODEL_PATH="$(realpath -- "$FEATURE_MODEL_PATH")"
fi
if [[ -e "$QWEN_CHECKPOINT" ]]; then
  QWEN_CHECKPOINT="$(realpath -- "$QWEN_CHECKPOINT")"
fi

[[ "$VIDEO_COUNT" =~ ^[1-9][0-9]*$ ]] || die "--video-count must be positive"
[[ "$SEED" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
for value in "$FEATURE_BATCH_SIZE" "$FRAME_BUFFER_SIZE" "$QWEN_MAX_PIXELS" "$NUM_ORIGINS" "$FRAME_BUDGET" "$MATCHED_COUNT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "integer runtime options must be positive"
done
[[ -f "$QUESTIONS_FILE" ]] || die "questions file not found: $QUESTIONS_FILE"
[[ -f "$CONFIG_FILE" ]] || die "config file not found: $CONFIG_FILE"

if [[ -z "$RUN_DIR" ]]; then
  if ((FULL_RUN)); then
    RUN_DIR="${REPO_ROOT}/artifacts/${BENCHMARK}_full"
  else
    RUN_DIR="${REPO_ROOT}/artifacts/${BENCHMARK}_stage0_${VIDEO_COUNT}"
  fi
fi
if [[ "$RUN_DIR" != /* ]]; then RUN_DIR="$CALLER_DIR/$RUN_DIR"; fi
RUN_DIR="$(realpath -m -- "$RUN_DIR")"

if ((RUN_BOOTSTRAP)); then
  bootstrap_args=(
    --python "$HOST_PYTHON"
    --venv-dir "$VENV_DIR"
    --datasets "$BENCHMARK"
    --dataset-check paths
  )
  case "$BENCHMARK" in
    videomme) bootstrap_args+=(--videomme-root "$DATASET_ROOT") ;;
    mlvu) bootstrap_args+=(--mlvu-root "$DATASET_ROOT") ;;
    lvb) bootstrap_args+=(--lvb-root "$DATASET_ROOT") ;;
  esac
  bash "${SCRIPT_DIR}/bootstrap_a100.sh" "${bootstrap_args[@]}"
fi

[[ -x "$VENV_DIR/bin/python" ]] || die "venv is not ready: $VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

if ((MATCHED_ONLY && CONFIG_EXPLICIT == 0)); then
  completed_stage_config="${RUN_DIR}/stage0_config.json"
  [[ -s "$completed_stage_config" ]] || \
    die "completed Stage-0 config not found for --matched-only: $completed_stage_config"
  CONFIG_FILE="$("$PYTHON_BIN" - "$completed_stage_config" <<'PY'
import json
import sys
from pathlib import Path

stage = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config_path = Path(stage.get("config_file", ""))
if not config_path.is_file():
    raise SystemExit(f"completed Stage-0 analysis config is missing: {config_path}")
print(config_path.resolve())
PY
)"
fi

# Keep the checked-in protocol immutable. For a K ablation, create a run-local
# config whose experiment.frame_budget matches the requested export/MLLM budget.
mkdir -p -- "$RUN_DIR"
EFFECTIVE_CONFIG="$("$PYTHON_BIN" - \
  "$CONFIG_FILE" "$RUN_DIR" "$FRAME_BUDGET" "$MATCHED_ONLY" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
budget = int(sys.argv[3])
matched_only = int(sys.argv[4]) == 1
payload = yaml.safe_load(source.read_text(encoding="utf-8"))
if not isinstance(payload, dict) or not isinstance(payload.get("experiment"), dict):
    raise SystemExit(f"invalid phase-stable config: {source}")
if payload["experiment"].get("frame_budget") == budget:
    print(source.resolve())
elif matched_only:
    raise SystemExit(
        "--matched-only cannot rewrite the completed Stage-0 config; "
        f"requested frame budget {budget} disagrees with {source}"
    )
else:
    payload["experiment"]["frame_budget"] = budget
    destination = run_dir / "effective_phase_stable_config.yaml"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(destination.resolve())
PY
)"

if ((MATCHED_ONLY == 0 && DOWNLOAD_DATA)) && [[ "$BENCHMARK" == "videomme" ]]; then
  fetch_count="$VIDEO_COUNT"
  ((FULL_RUN)) && fetch_count=0
  bash "${SCRIPT_DIR}/fetch_videomme.sh" \
    --dataset-root "$DATASET_ROOT" \
    --questions-file "$QUESTIONS_FILE" \
    --video-count "$fetch_count"
fi

# The pinned lmms-eval patch resolves raw videos under ./datasets/<benchmark>.
# Link only that ignored raw-data leaf when preprocessing uses an external mount.
actual_raw="$(cd -- "$DATASET_ROOT" && pwd -P)/$RAW_SUBDIR"
case "$BENCHMARK" in
  videomme) expected_raw="${REPO_ROOT}/datasets/videomme/data" ;;
  mlvu) expected_raw="${REPO_ROOT}/datasets/mlvu/video" ;;
  lvb) expected_raw="${REPO_ROOT}/datasets/longvideobench/videos" ;;
esac
[[ -d "$actual_raw" ]] || die "raw-video directory not found: $actual_raw"
if [[ "$actual_raw" != "$expected_raw" ]]; then
  if [[ -L "$expected_raw" ]]; then
    [[ "$(readlink -f -- "$expected_raw")" == "$(readlink -f -- "$actual_raw")" ]] || \
      die "existing raw-data symlink points elsewhere: $expected_raw"
  elif [[ -d "$expected_raw" ]]; then
    [[ -z "$(find "$expected_raw" -mindepth 1 -maxdepth 1 -print -quit)" ]] || \
      die "refusing to replace non-empty raw-data directory: $expected_raw"
    rmdir -- "$expected_raw"
    ln -s -- "$actual_raw" "$expected_raw"
  elif [[ ! -e "$expected_raw" ]]; then
    ln -s -- "$actual_raw" "$expected_raw"
  else
    die "raw-data compatibility path is not a directory/symlink: $expected_raw"
  fi
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

origins_csv=""
for ((origin = 0; origin < NUM_ORIGINS; origin++)); do
  origins_csv+="${origins_csv:+,}${origin}"
done
bootstrap_repetitions=1000
((FULL_RUN == 0)) || bootstrap_repetitions=10000
# The interaction bootstrap is CPU-only and its tail endpoint determines the
# primary downstream conclusion, so keep Monte Carlo error small even in the
# 20-video pilot.
interaction_bootstrap_repetitions=50000

run_matched_counterfactual() {
  local origin
  local signals_path="${RUN_DIR}/origin_signals.jsonl"
  local stage0_config="${RUN_DIR}/stage0_config.json"
  local preprocess_marker="${RUN_DIR}/.stage0_state/preprocess.done.json"
  local raw_inventory="${RUN_DIR}/raw_video_inventory.json"
  local matched_token=""
  printf -v matched_token 'b%02d' "$MATCHED_COUNT"
  local matched_root="${RUN_DIR}/matched_cardinality/${matched_token}"
  local matched_analysis="${matched_root}/analysis"
  local matched_keyframes="${matched_root}/keyframes"
  local matched_mllm="${matched_root}/mllm"
  local matched_predictions="${matched_root}/predictions.jsonl"
  local matched_summary="${matched_root}/mllm_stability_summary.json"
  local adaptive_predictions="${RUN_DIR}/predictions.jsonl"
  local interaction_summary="${matched_root}/adaptive_matched_interaction_summary.json"

  [[ -s "$signals_path" ]] || \
    die "completed Stage-0 signals not found for --matched-only: $signals_path"
  [[ -s "$stage0_config" ]] || \
    die "completed Stage-0 config not found for --matched-only: $stage0_config"
  [[ -s "$preprocess_marker" ]] || \
    die "completed preprocess marker not found for --matched-only: $preprocess_marker"
  [[ -s "$raw_inventory" ]] || \
    die "Stage-0 raw-video inventory not found for --matched-only: $raw_inventory"
  "$PYTHON_BIN" - \
    "$stage0_config" "$preprocess_marker" "$raw_inventory" "$signals_path" \
    "$EFFECTIVE_CONFIG" "$BENCHMARK" "$QUESTIONS_FILE" "$DATASET_ROOT" \
    "$NUM_ORIGINS" "$FRAME_BUDGET" "$FULL_RUN" <<'PY'
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

(
    stage_path,
    marker_path,
    inventory_path,
    signals_path,
    config_path,
    benchmark,
    questions,
    dataset_root,
    origins,
    budget,
    full,
) = sys.argv[1:]
stage = json.loads(Path(stage_path).read_text(encoding="utf-8"))
config_sha = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
expected = {
    "benchmark": benchmark,
    "questions": str(Path(questions).resolve()),
    "dataset_root": str(Path(dataset_root).resolve()),
    "origins": str(int(origins)),
    "k": str(int(budget)),
    "full": str(int(full)),
    "config_sha": config_sha,
}
for key, value in expected.items():
    if str(stage.get(key, "")) != value:
        raise SystemExit(
            f"matched-only setting {key}={value!r} disagrees with "
            f"completed Stage-0 value {stage.get(key)!r}"
        )

inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
videos = inventory.get("videos")
if not isinstance(videos, list) or not videos:
    raise SystemExit("raw-video inventory contains no videos")
for video in videos:
    path = Path(video.get("video_path", ""))
    if not path.is_file():
        raise SystemExit(f"Stage-0 raw video is missing: {path}")
    stat = path.stat()
    if stat.st_size != video.get("size") or stat.st_mtime_ns != video.get("mtime_ns"):
        raise SystemExit(f"Stage-0 raw video changed after preprocessing: {path}")

def file_digest(path: Path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"matched-only input is missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}

signal_source = Path(signals_path).resolve()
source_result = file_digest(signal_source)
references = []
origin_sets = defaultdict(set)
with signal_source.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("dataset", "")).lower() != benchmark:
            raise SystemExit(
                f"signal row {line_number} dataset does not match {benchmark}"
            )
        key = (str(row.get("video_id", "")), str(row.get("question_id", "")))
        if not all(key):
            raise SystemExit(f"signal row {line_number} has an empty item identity")
        origin_id = row.get("origin_id")
        if isinstance(origin_id, bool) or not isinstance(origin_id, int):
            raise SystemExit(f"signal row {line_number} has invalid origin_id")
        if origin_id in origin_sets[key]:
            raise SystemExit(f"duplicate signal origin {origin_id} for item {key}")
        origin_sets[key].add(origin_id)
        feature_path = row.get("visual_features_path")
        if feature_path:
            references.append(str(Path(feature_path).resolve()))
if not origin_sets:
    raise SystemExit("origin signal bundle contains no records")
expected_origins = set(range(int(origins)))
for key, observed in origin_sets.items():
    if observed != expected_origins:
        raise SystemExit(
            f"item {key} origins {sorted(observed)} do not match "
            f"expected {sorted(expected_origins)}"
        )

bundle_digest = hashlib.sha256()
bundle_digest.update(source_result["sha256"].encode())
unique_references = sorted(set(references))
for value in unique_references:
    result = file_digest(Path(value))
    bundle_digest.update(value.encode())
    bundle_digest.update(result["sha256"].encode())
actual_bundle = {
    "kind": "signal-bundle",
    "files": 1 + len(unique_references),
    "sha256": bundle_digest.hexdigest(),
}
marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
outputs = marker.get("outputs")
if not isinstance(outputs, dict):
    raise SystemExit("preprocess marker has no output checksums")
matching_specs = []
for spec, stored in outputs.items():
    if not spec.startswith("signal-bundle::"):
        continue
    if Path(spec.split("::", 1)[1]).resolve() == signal_source:
        matching_specs.append((spec, stored))
if len(matching_specs) != 1 or matching_specs[0][1] != actual_bundle:
    raise SystemExit(
        "origin signal/features no longer match the completed preprocess marker"
    )
print(
    "Matched-only preflight: Stage-0 config, signal checksum, features, and "
    "origin grid match"
)
PY
  mkdir -p -- "$matched_analysis" "$matched_keyframes" "$matched_mllm"

  "$PYTHON_BIN" -m phase_stable matched-selection \
    "$signals_path" "$matched_analysis" \
    --config "$EFFECTIVE_CONFIG" \
    --count "$MATCHED_COUNT" \
    --method-suffix _matched \
    --baseline-method dwt_matched \
    --treatment-method swt_matched \
    --n-bootstrap "$bootstrap_repetitions" \
    --confidence 0.95 \
    --seed "$SEED"

  local -a matched_export_args=(
    "$PYTHON_BIN" -m phase_stable export-keyframes
    --traces "${matched_analysis}/traces.jsonl"
    --benchmark "$BENCHMARK"
    --questions-file "$QUESTIONS_FILE"
    --dataset-root "$DATASET_ROOT"
    --output-dir "$matched_keyframes"
    --methods dwt_matched swt_matched
    --expected-budget "$FRAME_BUDGET"
  )
  for ((origin = 0; origin < NUM_ORIGINS; origin++)); do
    if ((origin == 0)); then
      matched_export_args+=(--origin-ids)
    fi
    matched_export_args+=("$origin")
  done
  ((FULL_RUN == 1)) || matched_export_args+=(--allow-partial)
  "${matched_export_args[@]}"

  local original_keyframes="${RUN_DIR}/keyframes/${BENCHMARK}_dwt_origin00.json"
  local matched_reference="${matched_keyframes}/${BENCHMARK}_dwt_matched_origin00.json"
  [[ -s "$original_keyframes" ]] || \
    die "original Stage-0 keyframes are missing: $original_keyframes"
  "$PYTHON_BIN" - "$original_keyframes" "$matched_reference" <<'PY'
import json
import sys
from pathlib import Path

def annotations_without_selection(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"keyframe annotation is empty or malformed: {path}")
    return [
        {key: value for key, value in row.items() if key != "keyframe_indices"}
        for row in rows
    ]

original = annotations_without_selection(sys.argv[1])
matched = annotations_without_selection(sys.argv[2])
if original != matched:
    raise SystemExit(
        "current questions/annotation subset differs from the completed Stage-0 export"
    )
print("Matched-only preflight: annotation subset matches completed Stage-0")
PY

  local -a matched_mllm_args=(
    --benchmark "$BENCHMARK"
    --keyframe-dir "$matched_keyframes"
    --output-root "$matched_mllm"
    --methods dwt_matched,swt_matched
    --origins "$origins_csv"
    --qwen-checkpoint "$QWEN_CHECKPOINT"
    --cuda-device "$CUDA_DEVICE"
    --max-num-frames "$FRAME_BUDGET"
    --max-pixels "$QWEN_MAX_PIXELS"
    --attention sdpa
    --batch-size 1
    --python-bin "$PYTHON_BIN"
    --converter-python "$PYTHON_BIN"
    --repo-root "$REPO_ROOT"
    --predictions-output "$matched_predictions"
  )
  [[ -z "$MLLM_LIMIT" ]] || matched_mllm_args+=(--limit "$MLLM_LIMIT")
  ((FORCE_MLLM == 0)) || matched_mllm_args+=(--force)
  bash "${SCRIPT_DIR}/run_mllm_grid.sh" "${matched_mllm_args[@]}"

  "$PYTHON_BIN" -m phase_stable evaluate-predictions \
    "$matched_predictions" "$matched_summary" \
    --baseline-method dwt_matched \
    --treatment-method swt_matched \
    --n-bootstrap "$bootstrap_repetitions" \
    --confidence 0.95 \
    --seed "$SEED"

  [[ -s "$adaptive_predictions" ]] || \
    die "original adaptive predictions are missing or empty: $adaptive_predictions"

  # MLLM_PROVENANCE_CHECK_BEGIN
  "$PYTHON_BIN" - \
    "${RUN_DIR}/mllm/${BENCHMARK}" "${matched_mllm}/${BENCHMARK}" \
    "$BENCHMARK" "$NUM_ORIGINS" <<'PY'
import json
import sys
from pathlib import Path

adaptive_root, matched_root, benchmark, raw_num_origins = sys.argv[1:]
num_origins = int(raw_num_origins)
signature_fields = (
    "config",
    "task_hashes",
    "versions",
    "model_name",
    "model_source",
    "chat_template_sha",
    "system_instruction_sha",
)
regimes = (
    ("adaptive", Path(adaptive_root), ("dwt", "swt")),
    ("matched", Path(matched_root), ("dwt_matched", "swt_matched")),
)


def read_marker(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or empty MLLM completion marker: {path}")
    fields: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or "=" not in line:
            raise SystemExit(f"malformed MLLM marker {path}:{line_number}")
        key, value = line.split("=", 1)
        if key in fields:
            raise SystemExit(f"duplicate MLLM marker field {key!r} in {path}")
        fields[key] = value
    return fields


def load_signature(
    marker_path: Path,
    *,
    method: str,
    origin: int,
) -> tuple[dict[str, object], Path]:
    marker = read_marker(marker_path)
    expected_marker_fields = {
        "benchmark": benchmark,
        "method": method,
        "origin_id": str(origin),
    }
    for field, expected in expected_marker_fields.items():
        if marker.get(field) != expected:
            raise SystemExit(
                f"MLLM marker {marker_path} has {field}={marker.get(field)!r}; "
                f"expected {expected!r}"
            )
    raw_result_path = marker.get("results_json")
    if not raw_result_path:
        raise SystemExit(f"MLLM marker has no results_json field: {marker_path}")
    result_path = Path(raw_result_path)
    if not result_path.is_absolute():
        result_path = marker_path.parent / result_path
    if not result_path.is_file() or result_path.stat().st_size <= 0:
        raise SystemExit(f"MLLM result JSON is missing or empty: {result_path}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid MLLM result JSON {result_path}: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise SystemExit(f"MLLM result JSON must contain an object: {result_path}")
    missing = [field for field in signature_fields if field not in result]
    if missing:
        raise SystemExit(
            f"MLLM result JSON {result_path} lacks provenance fields: "
            f"{', '.join(missing)}"
        )
    signature = {field: result[field] for field in signature_fields}
    return signature, result_path


regime_signatures: dict[str, dict[str, object]] = {}
cell_counts: dict[str, int] = {}
for regime_name, root, methods in regimes:
    reference_signature = None
    reference_result = None
    count = 0
    for method in methods:
        for origin in range(num_origins):
            marker_path = root / method / f"origin{origin:02d}" / ".complete"
            signature, result_path = load_signature(
                marker_path,
                method=method,
                origin=origin,
            )
            count += 1
            if reference_signature is None:
                reference_signature = signature
                reference_result = result_path
                continue
            if signature != reference_signature:
                changed = [
                    field
                    for field in signature_fields
                    if signature[field] != reference_signature[field]
                ]
                raise SystemExit(
                    f"{regime_name} MLLM provenance differs across cells: "
                    f"{reference_result} versus {result_path}; changed fields: "
                    f"{', '.join(changed)}"
                )
    if reference_signature is None:
        raise SystemExit(f"{regime_name} MLLM grid contains no expected cells")
    regime_signatures[regime_name] = reference_signature
    cell_counts[regime_name] = count

adaptive_signature = regime_signatures["adaptive"]
matched_signature = regime_signatures["matched"]
if adaptive_signature != matched_signature:
    changed = [
        field
        for field in signature_fields
        if adaptive_signature[field] != matched_signature[field]
    ]
    raise SystemExit(
        "adaptive and matched MLLM provenance do not match; changed fields: "
        + ", ".join(changed)
    )
print(
    "MLLM provenance preflight: "
    f"{cell_counts['adaptive']} adaptive and {cell_counts['matched']} matched "
    "cells share one signature"
)
PY
  # MLLM_PROVENANCE_CHECK_END

  "$PYTHON_BIN" -m phase_stable evaluate-prediction-interaction \
    "$adaptive_predictions" "$matched_predictions" "$interaction_summary" \
    --adaptive-baseline-method dwt \
    --adaptive-treatment-method swt \
    --matched-baseline-method dwt_matched \
    --matched-treatment-method swt_matched \
    --n-bootstrap "$interaction_bootstrap_repetitions" \
    --confidence 0.95 \
    --seed "$SEED"

  printf '\nMatched-cardinality experiment complete.\n'
  printf '  Boundary count:       %s\n' "$MATCHED_COUNT"
  printf '  Matched artifacts:    %s\n' "$matched_root"
  printf '  Matched predictions:  %s\n' "$matched_predictions"
  printf '  Matched MLLM summary: %s\n' "$matched_summary"
  printf '  Interaction summary:  %s\n' "$interaction_summary"
}

if ((MATCHED_ONLY)); then
  run_matched_counterfactual
  exit 0
fi

stage_args=(
  --benchmark "$BENCHMARK"
  --questions-file "$QUESTIONS_FILE"
  --dataset-root "$DATASET_ROOT"
  --run-dir "$RUN_DIR"
  --config "$EFFECTIVE_CONFIG"
  --seed "$SEED"
  --num-origins "$NUM_ORIGINS"
  --sample-fps "$SAMPLE_FPS"
  --frame-budget "$FRAME_BUDGET"
  --feature-model "$FEATURE_MODEL"
  --device cuda:0
  --batch-size "$FEATURE_BATCH_SIZE"
  --frame-buffer-size "$FRAME_BUFFER_SIZE"
  --python "$PYTHON_BIN"
)
if ((FULL_RUN)); then
  stage_args+=(--full --bootstrap 10000)
else
  stage_args+=(--bootstrap 1000 --video-indices)
  for ((index = 0; index < VIDEO_COUNT; index++)); do stage_args+=("$index"); done
fi
[[ -z "$FEATURE_MODEL_PATH" ]] || stage_args+=(--model-path "$FEATURE_MODEL_PATH")
((RUN_MATCHED == 0)) || stage_args+=(--matched-count "$MATCHED_COUNT")
for force_step in "${FORCE_STEPS[@]}"; do stage_args+=(--force-step "$force_step"); done

bash "${SCRIPT_DIR}/run_stage0.sh" "${stage_args[@]}"

if ((RUN_MLLM == 0)); then
  printf 'A100 experiment stopped after Stage-0 as requested: %s\n' "$RUN_DIR"
  exit 0
fi

predictions_path="${RUN_DIR}/predictions.jsonl"
mllm_args=(
  --benchmark "$BENCHMARK"
  --keyframe-dir "${RUN_DIR}/keyframes"
  --output-root "${RUN_DIR}/mllm"
  --methods dwt,swt
  --origins "$origins_csv"
  --qwen-checkpoint "$QWEN_CHECKPOINT"
  --cuda-device "$CUDA_DEVICE"
  --max-num-frames "$FRAME_BUDGET"
  --max-pixels "$QWEN_MAX_PIXELS"
  --attention sdpa
  --batch-size 1
  --python-bin "$PYTHON_BIN"
  --converter-python "$PYTHON_BIN"
  --repo-root "$REPO_ROOT"
  --predictions-output "$predictions_path"
)
((INCLUDE_BASELINES == 0)) || mllm_args+=(--include-baselines)
[[ -z "$MLLM_LIMIT" ]] || mllm_args+=(--limit "$MLLM_LIMIT")
((FORCE_MLLM == 0)) || mllm_args+=(--force)
bash "${SCRIPT_DIR}/run_mllm_grid.sh" "${mllm_args[@]}"

summary_path="${RUN_DIR}/mllm_stability_summary.json"
"$PYTHON_BIN" -m phase_stable evaluate-predictions \
  "$predictions_path" "$summary_path" \
  --baseline-method dwt \
  --treatment-method swt \
  --n-bootstrap "$bootstrap_repetitions" \
  --confidence 0.95 \
  --seed "$SEED"

printf '\nA100 experiment complete.\n'
printf '  Stage/trace artifacts: %s\n' "$RUN_DIR"
printf '  Predictions:          %s\n' "$predictions_path"
printf '  MLLM stability:       %s\n' "$summary_path"
