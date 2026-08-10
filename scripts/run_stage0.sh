#!/usr/bin/env bash
# Reproducible Linux/A100 runner for the phase-stable 20-video pilot.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_VERSION="phase-stable-stage0-v1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

die() {
  printf 'run_stage0: ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local exit_code=$?
  printf 'run_stage0: FAILED at line %s (exit=%s)\n' "${BASH_LINENO[0]}" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: scripts/run_stage0.sh [options]

Runs the implemented Phase-Stable WFS-SB stages in this order:
  make-benchmark-manifests -> preprocess-benchmark -> analyze-signals
  -> matched-boundaries (optional) -> selection-baselines -> export-keyframes

Core options (the uppercase name is the equivalent environment variable):
  --benchmark NAME          videomme, mlvu, or lvb [BENCHMARK=videomme]
  --questions-file PATH     official annotation JSON [QUESTIONS_FILE]
  --dataset-root PATH       benchmark root containing raw videos [DATASET_ROOT]
  --run-dir PATH            isolated artifact directory [RUN_DIR]
  --config PATH             analysis YAML [CONFIG_FILE]
  --video-indices N...      unique-video indices [VIDEO_INDICES="0 ... 19"]
  --full                    all unique videos; omits --video-indices [FULL_RUN]
  --seed N                  origin/bootstrap seed [SEED=20260810]
  --num-origins N           real sampling origins [NUM_ORIGINS=5]
  --sample-fps F            candidate FPS [SAMPLE_FPS=1.0]
  --frame-budget N          K, checked against config/candidates [FRAME_BUDGET=16]
  --feature-model NAME      blip2/blip1/clip/siglip [FEATURE_MODEL=blip2]
  --model-path PATH_OR_ID   optional fixed extractor checkpoint [MODEL_PATH]
  --device DEVICE           extractor device [DEVICE=cuda]
  --batch-size N            extractor batch size [BATCH_SIZE=32]
  --frame-buffer-size N     bounded decoded RGB buffer [FRAME_BUFFER_SIZE=256]
  --bootstrap N             paired bootstrap repetitions [N_BOOTSTRAP=1000]
  --python PATH             Python executable [PYTHON_BIN=python]

Optional matched-cardinality stage:
  --run-matched             run with B=4 unless another count is supplied
  --matched-count N         fixed top-B [MATCHED_COUNT]
  --matched-counts-json P   pre-registered per-video counts [MATCHED_COUNTS_JSON]
  --skip-matched            disable the optional stage [RUN_MATCHED=0]

Resume/control:
  --force-step NAME         rerun manifests, preprocess, analyze, matched,
                            baselines, export, or all (repeatable) [FORCE_STEP]
  --skip-device-preflight   do not require CUDA before preprocessing
                            [SKIP_DEVICE_PREFLIGHT]
  -h, --help                show this help

Defaults by benchmark:
  videomme: datasets/videomme/videomme_json_file.json
  mlvu:     datasets/mlvu/mlvu_dev.json
  lvb:      datasets/longvideobench/lvb_val.json

Markers under RUN_DIR/.stage0_state are reused only when the exact command and
input fingerprint match and every recorded output still matches its checksum.
EOF
}

# Environment defaults. Path defaults are resolved after parsing benchmark/full.
BENCHMARK="${BENCHMARK:-videomme}"
QUESTIONS_FILE="${QUESTIONS_FILE:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
RUN_DIR="${RUN_DIR:-}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/phase_stable_icassp.yaml}"
VIDEO_INDICES_RAW="${VIDEO_INDICES:-}"
FULL_RUN="${FULL_RUN:-0}"
SEED="${SEED:-20260810}"
NUM_ORIGINS="${NUM_ORIGINS:-5}"
SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
FRAME_BUDGET="${FRAME_BUDGET:-16}"
FEATURE_MODEL="${FEATURE_MODEL:-blip2}"
MODEL_PATH="${MODEL_PATH:-}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FRAME_BUFFER_SIZE="${FRAME_BUFFER_SIZE:-256}"
N_BOOTSTRAP="${N_BOOTSTRAP:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MATCHED="${RUN_MATCHED:-}"
MATCHED_COUNT="${MATCHED_COUNT:-}"
MATCHED_COUNTS_JSON="${MATCHED_COUNTS_JSON:-}"
SKIP_DEVICE_PREFLIGHT="${SKIP_DEVICE_PREFLIGHT:-0}"
declare -a FORCE_STEPS=()

if [[ -n "${FORCE_STEP:-}" ]]; then
  force_from_env="${FORCE_STEP//,/ }"
  old_ifs="${IFS}"
  IFS=' ' read -r -a env_force_steps <<< "${force_from_env}"
  IFS="${old_ifs}"
  FORCE_STEPS+=("${env_force_steps[@]}")
fi

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --benchmark) need_value "$@"; BENCHMARK="$2"; shift 2 ;;
    --questions-file) need_value "$@"; QUESTIONS_FILE="$2"; shift 2 ;;
    --dataset-root) need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
    --run-dir) need_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG_FILE="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --num-origins) need_value "$@"; NUM_ORIGINS="$2"; shift 2 ;;
    --sample-fps) need_value "$@"; SAMPLE_FPS="$2"; shift 2 ;;
    --frame-budget) need_value "$@"; FRAME_BUDGET="$2"; shift 2 ;;
    --feature-model) need_value "$@"; FEATURE_MODEL="$2"; shift 2 ;;
    --model-path) need_value "$@"; MODEL_PATH="$2"; shift 2 ;;
    --device) need_value "$@"; DEVICE="$2"; shift 2 ;;
    --batch-size) need_value "$@"; BATCH_SIZE="$2"; shift 2 ;;
    --frame-buffer-size) need_value "$@"; FRAME_BUFFER_SIZE="$2"; shift 2 ;;
    --bootstrap|--n-bootstrap) need_value "$@"; N_BOOTSTRAP="$2"; shift 2 ;;
    --python) need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --full) FULL_RUN=1; shift ;;
    --video-indices)
      shift
      [[ $# -gt 0 && "$1" != --* ]] || die "--video-indices requires one or more integers"
      VIDEO_INDICES_RAW=""
      while [[ $# -gt 0 && "$1" != --* ]]; do
        VIDEO_INDICES_RAW+="${VIDEO_INDICES_RAW:+ }$1"
        shift
      done
      ;;
    --run-matched) RUN_MATCHED=1; shift ;;
    --skip-matched) RUN_MATCHED=0; shift ;;
    --matched-count)
      need_value "$@"; MATCHED_COUNT="$2"; MATCHED_COUNTS_JSON=""; RUN_MATCHED=1; shift 2
      ;;
    --matched-counts-json)
      need_value "$@"; MATCHED_COUNTS_JSON="$2"; MATCHED_COUNT=""; RUN_MATCHED=1; shift 2
      ;;
    --force-step) need_value "$@"; FORCE_STEPS+=("$2"); shift 2 ;;
    --skip-device-preflight) SKIP_DEVICE_PREFLIGHT=1; shift ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

BENCHMARK="${BENCHMARK,,}"
case "${BENCHMARK}" in
  videomme)
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/videomme}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/videomme_json_file.json}"
    ;;
  mlvu)
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/mlvu}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/mlvu_dev.json}"
    ;;
  lvb)
    DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/longvideobench}"
    QUESTIONS_FILE="${QUESTIONS_FILE:-${DATASET_ROOT}/lvb_val.json}"
    ;;
  *) die "--benchmark must be videomme, mlvu, or lvb" ;;
esac

case "${FULL_RUN,,}" in
  1|true|yes|on) FULL_RUN=1 ;;
  0|false|no|off|'') FULL_RUN=0 ;;
  *) die "FULL_RUN must be boolean" ;;
esac
if [[ -z "${N_BOOTSTRAP}" ]]; then
  if [[ "${FULL_RUN}" == 1 ]]; then N_BOOTSTRAP=10000; else N_BOOTSTRAP=1000; fi
fi
if [[ -z "${RUN_DIR}" ]]; then
  if [[ "${FULL_RUN}" == 1 ]]; then
    RUN_DIR="${REPO_ROOT}/artifacts/${BENCHMARK}_full"
  else
    RUN_DIR="${REPO_ROOT}/artifacts/${BENCHMARK}_stage0_20"
  fi
fi

if [[ "${FULL_RUN}" == 0 && -z "${VIDEO_INDICES_RAW}" ]]; then
  for ((index = 0; index < 20; index++)); do
    VIDEO_INDICES_RAW+="${VIDEO_INDICES_RAW:+ }${index}"
  done
fi
declare -a VIDEO_INDEX_ARRAY=()
if [[ "${FULL_RUN}" == 0 ]]; then
  normalized_indices="${VIDEO_INDICES_RAW//,/ }"
  old_ifs="${IFS}"
  IFS=' ' read -r -a VIDEO_INDEX_ARRAY <<< "${normalized_indices}"
  IFS="${old_ifs}"
  [[ ${#VIDEO_INDEX_ARRAY[@]} -gt 0 ]] || die "video index list must not be empty"
  declare -A seen_indices=()
  for index in "${VIDEO_INDEX_ARRAY[@]}"; do
    [[ "${index}" =~ ^[0-9]+$ ]] || die "invalid video index: ${index}"
    [[ -z "${seen_indices[${index}]:-}" ]] || die "duplicate video index: ${index}"
    seen_indices["${index}"]=1
  done
fi

if [[ -z "${RUN_MATCHED}" ]]; then
  if [[ -n "${MATCHED_COUNT}" || -n "${MATCHED_COUNTS_JSON}" ]]; then
    RUN_MATCHED=1
  else
    RUN_MATCHED=0
  fi
fi
case "${RUN_MATCHED,,}" in
  1|true|yes|on) RUN_MATCHED=1 ;;
  0|false|no|off) RUN_MATCHED=0 ;;
  *) die "RUN_MATCHED must be boolean" ;;
esac
if [[ "${RUN_MATCHED}" == 1 && -z "${MATCHED_COUNT}" && -z "${MATCHED_COUNTS_JSON}" ]]; then
  MATCHED_COUNT=4
fi
[[ -z "${MATCHED_COUNT}" || -z "${MATCHED_COUNTS_JSON}" ]] \
  || die "MATCHED_COUNT and MATCHED_COUNTS_JSON are mutually exclusive"

valid_force_steps=" manifests preprocess analyze matched baselines export all "
for force_step in "${FORCE_STEPS[@]}"; do
  [[ "${valid_force_steps}" == *" ${force_step} "* ]] \
    || die "invalid --force-step ${force_step}"
done

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python executable not found: ${PYTHON_BIN}"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
[[ -f "${QUESTIONS_FILE}" ]] || die "questions file not found: ${QUESTIONS_FILE}"
[[ -d "${DATASET_ROOT}" ]] || die "dataset root not found: ${DATASET_ROOT}"
[[ -f "${CONFIG_FILE}" ]] || die "analysis config not found: ${CONFIG_FILE}"
if [[ -n "${MATCHED_COUNTS_JSON}" ]]; then
  [[ -f "${MATCHED_COUNTS_JSON}" ]] \
    || die "matched counts JSON not found: ${MATCHED_COUNTS_JSON}"
fi

"${PYTHON_BIN}" - \
  "${SEED}" "${NUM_ORIGINS}" "${SAMPLE_FPS}" "${FRAME_BUDGET}" \
  "${BATCH_SIZE}" "${FRAME_BUFFER_SIZE}" "${N_BOOTSTRAP}" \
  "${MATCHED_COUNT:-0}" "${FEATURE_MODEL}" "${CONFIG_FILE}" <<'PY'
import sys
from phase_stable.config import load_phase_stable_config

seed, origins, fps, budget, batch, buffer_size, bootstrap, matched = sys.argv[1:9]
feature_model, config_path = sys.argv[9:11]
int(seed)
positive_ints = {
    "num_origins": origins,
    "frame_budget": budget,
    "batch_size": batch,
    "frame_buffer_size": buffer_size,
    "n_bootstrap": bootstrap,
}
for name, raw in positive_ints.items():
    if int(raw) <= 0:
        raise SystemExit(f"{name} must be positive")
if float(fps) <= 0:
    raise SystemExit("sample_fps must be positive")
if int(matched) < 0:
    raise SystemExit("matched count must be non-negative")
if feature_model not in {"blip2", "blip1", "clip", "siglip"}:
    raise SystemExit(f"unsupported feature model: {feature_model}")
config = load_phase_stable_config(config_path)
if config.experiment.frame_budget != int(budget):
    raise SystemExit(
        f"FRAME_BUDGET={budget} disagrees with config experiment.frame_budget="
        f"{config.experiment.frame_budget}"
    )
required_methods = {"dwt", "swt"}
if not required_methods.issubset(config.experiment.methods):
    raise SystemExit("analysis config must include both dwt and swt")
PY

case "${SKIP_DEVICE_PREFLIGHT,,}" in
  1|true|yes|on) SKIP_DEVICE_PREFLIGHT=1 ;;
  0|false|no|off|'') SKIP_DEVICE_PREFLIGHT=0 ;;
  *) die "SKIP_DEVICE_PREFLIGHT must be boolean" ;;
esac
if [[ "${DEVICE}" == cuda* && "${SKIP_DEVICE_PREFLIGHT}" == 0 ]]; then
  "${PYTHON_BIN}" - "${DEVICE}" <<'PY'
import sys
import torch

device = sys.argv[1]
if not torch.cuda.is_available():
    raise SystemExit(f"DEVICE={device}, but torch.cuda.is_available() is false")
index = int(device.split(":", 1)[1]) if ":" in device else 0
if index >= torch.cuda.device_count():
    raise SystemExit(f"CUDA device index {index} is unavailable")
name = torch.cuda.get_device_name(index)
print(f"CUDA preflight: device={device}, name={name}")
if "A100" not in name.upper():
    print("run_stage0: warning: runner defaults were tuned for A100", file=sys.stderr)
PY
fi

"${PYTHON_BIN}" -m phase_stable --help >/dev/null

MANIFESTS_PATH="${MANIFESTS_PATH:-${RUN_DIR}/sampling_manifests.jsonl}"
CATALOG_PATH="${CATALOG_PATH:-${RUN_DIR}/video_catalog.jsonl}"
PREPROCESS_DIR="${PREPROCESS_DIR:-${RUN_DIR}/preprocess}"
SIGNALS_PATH="${SIGNALS_PATH:-${RUN_DIR}/origin_signals.jsonl}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_DIR}/analysis}"
BASELINE_DIR="${BASELINE_DIR:-${RUN_DIR}/baselines}"
KEYFRAME_DIR="${KEYFRAME_DIR:-${RUN_DIR}/keyframes}"
MATCHED_OUTPUT="${MATCHED_OUTPUT:-${ANALYSIS_DIR}/matched_boundary_metrics.jsonl}"
STATE_DIR="${RUN_DIR}/.stage0_state"
LOG_DIR="${RUN_DIR}/logs"
RAW_INVENTORY="${RUN_DIR}/raw_video_inventory.json"
MANIFEST_SUMMARY="${RUN_DIR}/manifest_preflight.json"
RUN_CONFIG="${RUN_DIR}/stage0_config.json"
mkdir -p "${RUN_DIR}" "${STATE_DIR}" "${LOG_DIR}" "${KEYFRAME_DIR}"

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

fingerprint_tokens() {
  { for token in "$@"; do printf '%s\0' "${token}"; done; } \
    | sha256sum | awk '{print $1}'
}

render_command() {
  local rendered=""
  local token
  local quoted
  for token in "$@"; do
    printf -v quoted '%q' "${token}"
    rendered+="${rendered:+ }${quoted}"
  done
  printf '%s' "${rendered}"
}

SCRIPT_HASH="$(sha256_file "${BASH_SOURCE[0]}")"
mapfile -d '' CODE_FILES < <(
  {
    find phase_stable -type f -name '*.py' -print0
    printf '%s\0' preprocess/extract.py
  } | sort -z
)
CODE_HASH="$({
  for code_file in "${CODE_FILES[@]}"; do
    printf '%s\0%s\0' "${code_file}" "$(sha256_file "${code_file}")"
  done
} | sha256sum | awk '{print $1}')"
QUESTIONS_HASH="$(sha256_file "${QUESTIONS_FILE}")"
ANALYSIS_CONFIG_HASH="$(sha256_file "${CONFIG_FILE}")"

MODEL_FINGERPRINT="$("${PYTHON_BIN}" - "${MODEL_PATH:-default:${FEATURE_MODEL}}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

value = sys.argv[1]
path = Path(value)
if not path.exists():
    payload = {"model_id": value}
elif path.is_file():
    stat = path.stat()
    payload = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
else:
    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        files.append((str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    payload = {"path": str(path.resolve()), "files": files}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
print(hashlib.sha256(encoded).hexdigest())
PY
)"

# Raw-video preflight uses the same public loader and unique-video indices as
# make-benchmark-manifests. The inventory is stable unless path/size/mtime changes.
"${PYTHON_BIN}" - \
  "${BENCHMARK}" "${QUESTIONS_FILE}" "${DATASET_ROOT}" "${RAW_INVENTORY}" \
  "${FULL_RUN}" "${VIDEO_INDEX_ARRAY[@]}" <<'PY'
import json
import sys
from pathlib import Path

from phase_stable.benchmarks import load_benchmark_videos

benchmark, questions, dataset_root, output, full = sys.argv[1:6]
indices = [int(value) for value in sys.argv[6:]]
videos = load_benchmark_videos(benchmark, questions, dataset_root)
if full == "1":
    selected = videos
else:
    if any(index < 0 or index >= len(videos) for index in indices):
        raise SystemExit(f"video indices must lie in [0, {len(videos)})")
    selected = [videos[index] for index in indices]
if not selected:
    raise SystemExit("raw-video preflight selected no videos")
rows = []
for video in selected:
    path = video.video_path.resolve()
    if not path.is_file():
        raise SystemExit(f"raw video is missing: {path}")
    stat = path.stat()
    if stat.st_size <= 0:
        raise SystemExit(f"raw video is empty: {path}")
    rows.append({
        "video_id": video.video_id,
        "video_path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "num_questions": len(video.queries),
    })
payload = {"benchmark": benchmark, "full": full == "1", "videos": rows}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"Raw-video preflight: {len(rows)} videos")
PY
RAW_INVENTORY_HASH="$(sha256_file "${RAW_INVENTORY}")"

indices_csv=""
if [[ "${FULL_RUN}" == 0 ]]; then
  old_ifs="${IFS}"; IFS=,; indices_csv="${VIDEO_INDEX_ARRAY[*]}"; IFS="${old_ifs}"
fi
CONFIG_HASH="$(STAGE0_SCRIPT_VERSION="${SCRIPT_VERSION}" \
  STAGE0_BENCHMARK="${BENCHMARK}" STAGE0_QUESTIONS="${QUESTIONS_FILE}" \
  STAGE0_DATASET_ROOT="${DATASET_ROOT}" STAGE0_RUN_DIR="${RUN_DIR}" \
  STAGE0_CONFIG_FILE="${CONFIG_FILE}" STAGE0_CONFIG_SHA="${ANALYSIS_CONFIG_HASH}" \
  STAGE0_FULL="${FULL_RUN}" STAGE0_INDICES="${indices_csv}" \
  STAGE0_SEED="${SEED}" STAGE0_ORIGINS="${NUM_ORIGINS}" \
  STAGE0_FPS="${SAMPLE_FPS}" STAGE0_K="${FRAME_BUDGET}" \
  STAGE0_MODEL="${FEATURE_MODEL}" STAGE0_MODEL_PATH="${MODEL_PATH}" \
  STAGE0_MODEL_FP="${MODEL_FINGERPRINT}" STAGE0_DEVICE="${DEVICE}" \
  STAGE0_BATCH="${BATCH_SIZE}" STAGE0_BUFFER="${FRAME_BUFFER_SIZE}" \
  STAGE0_BOOTSTRAP="${N_BOOTSTRAP}" STAGE0_RUN_MATCHED="${RUN_MATCHED}" \
  STAGE0_MATCHED_COUNT="${MATCHED_COUNT}" STAGE0_MATCHED_JSON="${MATCHED_COUNTS_JSON}" \
  STAGE0_SCRIPT_SHA="${SCRIPT_HASH}" STAGE0_CODE_SHA="${CODE_HASH}" \
  "${PYTHON_BIN}" - "${RUN_CONFIG}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

keys = sorted(key for key in os.environ if key.startswith("STAGE0_"))
payload = {key.removeprefix("STAGE0_").lower(): os.environ[key] for key in keys}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
document = {"schema_version": 1, "config_hash": config_hash, **payload}
destination = Path(sys.argv[1])
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(config_hash)
PY
)"

is_forced() {
  local step="$1"
  local forced
  for forced in "${FORCE_STEPS[@]}"; do
    [[ "${forced}" == all || "${forced}" == "${step}" ]] && return 0
  done
  return 1
}

# Output specs are either ordinary files, signal-bundle::<signals.jsonl>, or
# trace-bundle::<traces.jsonl>. Bundle checks include every referenced array.
marker_valid() {
  local marker="$1" fingerprint="$2"
  shift 2
  "${PYTHON_BIN}" - "${marker}" "${fingerprint}" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker, expected_fingerprint, *expected_specs = sys.argv[1:]

def file_digest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size <= 0:
        raise ValueError(f"output is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}

def bundle_digest(spec: str):
    kind, raw_path = spec.split("::", 1)
    source = Path(raw_path)
    source_digest = file_digest(source)
    references = []
    field = "visual_features_path" if kind == "signal-bundle" else "array_path"
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get(field)
            if value:
                references.append(str(Path(value).resolve()))
    references = sorted(set(references))
    digest = hashlib.sha256()
    digest.update(source_digest["sha256"].encode())
    for value in references:
        result = file_digest(Path(value))
        digest.update(value.encode())
        digest.update(result["sha256"].encode())
    return {"kind": kind, "files": 1 + len(references), "sha256": digest.hexdigest()}

def digest_spec(spec: str):
    if spec.startswith(("signal-bundle::", "trace-bundle::")):
        return bundle_digest(spec)
    return file_digest(Path(spec))

try:
    payload = json.loads(Path(marker).read_text(encoding="utf-8"))
    if payload.get("fingerprint") != expected_fingerprint:
        raise ValueError("fingerprint mismatch")
    stored = payload.get("outputs")
    if not isinstance(stored, dict) or set(stored) != set(expected_specs):
        raise ValueError("output specification mismatch")
    for spec in expected_specs:
        if digest_spec(spec) != stored[spec]:
            raise ValueError(f"output checksum mismatch: {spec}")
except Exception as exc:
    print(f"resume marker rejected: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

write_marker() {
  local marker="$1" step="$2" fingerprint="$3" command_text="$4"
  shift 4
  "${PYTHON_BIN}" - \
    "${marker}" "${step}" "${fingerprint}" "${CONFIG_HASH}" "${command_text}" "$@" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

marker, step, fingerprint, config_hash, command_text, *specs = sys.argv[1:]

def file_digest(path: Path):
    if not path.is_file():
        raise SystemExit(f"required output is missing: {path}")
    if path.stat().st_size <= 0:
        raise SystemExit(f"required output is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "size": path.stat().st_size, "sha256": digest.hexdigest()}

def bundle_digest(spec: str):
    kind, raw_path = spec.split("::", 1)
    source = Path(raw_path)
    source_result = file_digest(source)
    field = "visual_features_path" if kind == "signal-bundle" else "array_path"
    references = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL output {source}:{line_number}: {exc}")
            value = row.get(field)
            if value:
                references.append(str(Path(value).resolve()))
    references = sorted(set(references))
    digest = hashlib.sha256()
    digest.update(source_result["sha256"].encode())
    for value in references:
        result = file_digest(Path(value))
        digest.update(value.encode())
        digest.update(result["sha256"].encode())
    return {"kind": kind, "files": 1 + len(references), "sha256": digest.hexdigest()}

def digest_spec(spec: str):
    if spec.startswith(("signal-bundle::", "trace-bundle::")):
        return bundle_digest(spec)
    return file_digest(Path(spec))

outputs = {spec: digest_spec(spec) for spec in specs}
payload = {
    "schema_version": 1,
    "step": step,
    "fingerprint": fingerprint,
    "config_hash": config_hash,
    "command": command_text,
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "outputs": outputs,
}
destination = Path(marker)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PY
}

declare -a STEP_OUTPUTS=()
STEP_COMMAND_DISPLAY=""
run_step() {
  local step="$1" fingerprint="$2"
  shift 2
  local marker="${STATE_DIR}/${step}.done.json"
  local command_text="${STEP_COMMAND_DISPLAY:-$(render_command "$@") }"
  if ! is_forced "${step}" && marker_valid \
    "${marker}" "${fingerprint}" "${STEP_OUTPUTS[@]}" 2>/dev/null; then
    printf '[resume] %-10s fingerprint and output checksums match\n' "${step}"
    return 0
  fi
  rm -f -- "${marker}"
  printf '[run]    %-10s %s\n' "${step}" "${command_text}"
  "$@" 2>&1 | tee "${LOG_DIR}/${step}.log"
  write_marker \
    "${marker}" "${step}" "${fingerprint}" "${command_text}" "${STEP_OUTPUTS[@]}"
  printf '[done]   %-10s marker=%s\n' "${step}" "${marker}"
}

step_fingerprint() {
  local step="$1"
  shift
  fingerprint_tokens \
    "${SCRIPT_VERSION}" "step=${step}" "script=${SCRIPT_HASH}" "code=${CODE_HASH}" "$@"
}

declare -a MANIFEST_CMD=(
  "${PYTHON_BIN}" -m phase_stable make-benchmark-manifests
  --benchmark "${BENCHMARK}"
  --questions-file "${QUESTIONS_FILE}"
  --dataset-root "${DATASET_ROOT}"
  --output "${MANIFESTS_PATH}"
  --catalog-output "${CATALOG_PATH}"
  --seed "${SEED}"
  --num-origins "${NUM_ORIGINS}"
  --sample-fps "${SAMPLE_FPS}"
)
if [[ "${FULL_RUN}" == 0 ]]; then
  MANIFEST_CMD+=(--video-indices "${VIDEO_INDEX_ARRAY[@]}")
fi
MANIFEST_FP="$(step_fingerprint manifests \
  "questions=${QUESTIONS_HASH}" "raw_inventory=${RAW_INVENTORY_HASH}" \
  "$(render_command "${MANIFEST_CMD[@]}")")"
STEP_OUTPUTS=("${MANIFESTS_PATH}" "${CATALOG_PATH}")
STEP_COMMAND_DISPLAY=""
run_step manifests "${MANIFEST_FP}" "${MANIFEST_CMD[@]}"

# Manifest/K preflight always runs, including after a resumed manifest step.
"${PYTHON_BIN}" - \
  "${MANIFESTS_PATH}" "${RAW_INVENTORY}" "${MANIFEST_SUMMARY}" \
  "${SEED}" "${NUM_ORIGINS}" "${SAMPLE_FPS}" "${FRAME_BUDGET}" <<'PY'
import json
import math
import sys
from pathlib import Path

from phase_stable.sampling import read_manifests_jsonl

manifest_path, inventory_path, output, seed, origins, fps, budget = sys.argv[1:8]
manifests = read_manifests_jsonl(manifest_path)
inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
expected_ids = [row["video_id"] for row in inventory["videos"]]
actual_ids = [manifest.video_id for manifest in manifests]
if actual_ids != expected_ids:
    raise SystemExit("manifest video IDs/order do not match the raw-video preflight")
for manifest in manifests:
    if manifest.master_seed != int(seed):
        raise SystemExit(f"manifest seed mismatch: {manifest.video_id}")
    if manifest.num_origins != int(origins):
        raise SystemExit(f"manifest origin-count mismatch: {manifest.video_id}")
    if not math.isclose(manifest.sample_fps, float(fps), rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"manifest FPS mismatch: {manifest.video_id}")
minimum = min(manifest.candidate_count for manifest in manifests)
if minimum < int(budget):
    bad = [(manifest.video_id, manifest.candidate_count) for manifest in manifests if manifest.candidate_count < int(budget)]
    raise SystemExit(f"minimum candidate count {minimum} is below K={budget}: {bad[:10]}")
payload = {
    "num_videos": len(manifests),
    "num_origins": int(origins),
    "sample_fps": float(fps),
    "frame_budget": int(budget),
    "min_candidate_count": minimum,
    "max_candidate_count": max(manifest.candidate_count for manifest in manifests),
}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"Manifest preflight: {payload}")
PY

MANIFESTS_HASH="$(sha256_file "${MANIFESTS_PATH}")"
declare -a PREPROCESS_CMD=(
  "${PYTHON_BIN}" -m phase_stable preprocess-benchmark
  --benchmark "${BENCHMARK}"
  --questions-file "${QUESTIONS_FILE}"
  --dataset-root "${DATASET_ROOT}"
  --manifests "${MANIFESTS_PATH}"
  --output-dir "${PREPROCESS_DIR}"
  --signal-jsonl "${SIGNALS_PATH}"
  --feature-model "${FEATURE_MODEL}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --frame-buffer-size "${FRAME_BUFFER_SIZE}"
)
if [[ -n "${MODEL_PATH}" ]]; then PREPROCESS_CMD+=(--model-path "${MODEL_PATH}"); fi
PREPROCESS_FP="$(step_fingerprint preprocess \
  "questions=${QUESTIONS_HASH}" "raw_inventory=${RAW_INVENTORY_HASH}" \
  "manifests=${MANIFESTS_HASH}" "model=${MODEL_FINGERPRINT}" \
  "$(render_command "${PREPROCESS_CMD[@]}")")"
STEP_OUTPUTS=(
  "signal-bundle::${SIGNALS_PATH}"
  "${PREPROCESS_DIR}/manifest/run_manifest.json"
  "${PREPROCESS_DIR}/manifest/environment.json"
)
STEP_COMMAND_DISPLAY=""
run_step preprocess "${PREPROCESS_FP}" "${PREPROCESS_CMD[@]}"

SIGNAL_BUNDLE_HASH="$("${PYTHON_BIN}" - "${SIGNALS_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
paths = [source]
with source.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            value = json.loads(line).get("visual_features_path")
            if value:
                paths.append(Path(value))
digest = hashlib.sha256()
for path in sorted({str(Path(value).resolve()) for value in paths}):
    file_path = Path(path)
    if not file_path.is_file():
        raise SystemExit(f"signal bundle file missing: {file_path}")
    digest.update(path.encode())
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
print(digest.hexdigest())
PY
)"

declare -a ANALYZE_CMD=(
  "${PYTHON_BIN}" -m phase_stable analyze-signals
  "${SIGNALS_PATH}" "${ANALYSIS_DIR}"
  --config "${CONFIG_FILE}"
  --baseline-method dwt
  --treatment-method swt
  --n-bootstrap "${N_BOOTSTRAP}"
  --confidence 0.95
  --seed "${SEED}"
)
ANALYZE_FP="$(step_fingerprint analyze \
  "signals=${SIGNAL_BUNDLE_HASH}" "analysis_config=${ANALYSIS_CONFIG_HASH}" \
  "$(render_command "${ANALYZE_CMD[@]}")")"
TRACES_PATH="${ANALYSIS_DIR}/traces.jsonl"
STEP_OUTPUTS=(
  "trace-bundle::${TRACES_PATH}"
  "${ANALYSIS_DIR}/item_metrics.jsonl"
  "${ANALYSIS_DIR}/item_metrics.csv"
  "${ANALYSIS_DIR}/summary.json"
  "${ANALYSIS_DIR}/manifest/run_manifest.json"
  "${ANALYSIS_DIR}/manifest/environment.json"
)
STEP_COMMAND_DISPLAY=""
run_step analyze "${ANALYZE_FP}" "${ANALYZE_CMD[@]}"

TRACES_HASH="$(sha256_file "${TRACES_PATH}")"
if [[ "${RUN_MATCHED}" == 1 ]]; then
  declare -a MATCHED_CMD=(
    "${PYTHON_BIN}" -m phase_stable matched-boundaries
    --traces "${TRACES_PATH}"
    --output "${MATCHED_OUTPUT}"
  )
  matched_input_fingerprint=""
  if [[ -n "${MATCHED_COUNTS_JSON}" ]]; then
    MATCHED_CMD+=(--counts-json "${MATCHED_COUNTS_JSON}")
    matched_input_fingerprint="counts=$(sha256_file "${MATCHED_COUNTS_JSON}")"
  else
    MATCHED_CMD+=(--count "${MATCHED_COUNT}")
    matched_input_fingerprint="count=${MATCHED_COUNT}"
  fi
  MATCHED_CMD+=(--tolerance-sec 1.0 --edge-margin-sec 0.0)
  MATCHED_FP="$(step_fingerprint matched \
    "traces=${TRACES_HASH}" "${matched_input_fingerprint}" \
    "$(render_command "${MATCHED_CMD[@]}")")"
  STEP_OUTPUTS=("${MATCHED_OUTPUT}")
  STEP_COMMAND_DISPLAY=""
  run_step matched "${MATCHED_FP}" "${MATCHED_CMD[@]}"
else
  printf '[skip]   matched    optional stage disabled\n'
fi

declare -a BASELINE_CMD=(
  "${PYTHON_BIN}" -m phase_stable selection-baselines
  "${SIGNALS_PATH}" "${BASELINE_DIR}"
  --methods uniform topk
  --frame-budget "${FRAME_BUDGET}"
  --selected-tolerance-sec 1.0
)
BASELINE_FP="$(step_fingerprint baselines \
  "signals=${SIGNAL_BUNDLE_HASH}" "$(render_command "${BASELINE_CMD[@]}")")"
BASELINE_TRACES="${BASELINE_DIR}/baseline_traces.jsonl"
STEP_OUTPUTS=(
  "${BASELINE_TRACES}"
  "${BASELINE_DIR}/baseline_item_metrics.jsonl"
)
STEP_COMMAND_DISPLAY=""
run_step baselines "${BASELINE_FP}" "${BASELINE_CMD[@]}"

declare -a ORIGIN_IDS=()
for ((origin_id = 0; origin_id < NUM_ORIGINS; origin_id++)); do ORIGIN_IDS+=("${origin_id}"); done
declare -a EXPORT_MAIN_CMD=(
  "${PYTHON_BIN}" -m phase_stable export-keyframes
  --traces "${TRACES_PATH}"
  --benchmark "${BENCHMARK}"
  --questions-file "${QUESTIONS_FILE}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${KEYFRAME_DIR}"
  --methods dwt swt
  --origin-ids "${ORIGIN_IDS[@]}"
  --expected-budget "${FRAME_BUDGET}"
)
declare -a EXPORT_BASELINE_CMD=(
  "${PYTHON_BIN}" -m phase_stable export-keyframes
  --traces "${BASELINE_TRACES}"
  --benchmark "${BENCHMARK}"
  --questions-file "${QUESTIONS_FILE}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${KEYFRAME_DIR}"
  --methods uniform topk
  --origin-ids "${ORIGIN_IDS[@]}"
  --expected-budget "${FRAME_BUDGET}"
)
if [[ "${FULL_RUN}" == 0 ]]; then
  EXPORT_MAIN_CMD+=(--allow-partial)
  EXPORT_BASELINE_CMD+=(--allow-partial)
fi
run_exports() {
  "${EXPORT_MAIN_CMD[@]}"
  "${EXPORT_BASELINE_CMD[@]}"
}
BASELINE_TRACES_HASH="$(sha256_file "${BASELINE_TRACES}")"
EXPORT_COMMAND_TEXT="$(render_command "${EXPORT_MAIN_CMD[@]}") && $(render_command "${EXPORT_BASELINE_CMD[@]}")"
EXPORT_FP="$(step_fingerprint export \
  "questions=${QUESTIONS_HASH}" "traces=${TRACES_HASH}" \
  "baseline_traces=${BASELINE_TRACES_HASH}" "${EXPORT_COMMAND_TEXT}")"
STEP_OUTPUTS=()
for method in dwt swt uniform topk; do
  for origin_id in "${ORIGIN_IDS[@]}"; do
    printf -v origin_token '%02d' "${origin_id}"
    STEP_OUTPUTS+=("${KEYFRAME_DIR}/${BENCHMARK}_${method}_origin${origin_token}.json")
  done
done
STEP_COMMAND_DISPLAY="${EXPORT_COMMAND_TEXT}"
run_step export "${EXPORT_FP}" run_exports
STEP_COMMAND_DISPLAY=""

printf '\nStage-0 complete.\n'
printf '  benchmark:    %s\n' "${BENCHMARK}"
printf '  selection:    %s\n' "$([[ "${FULL_RUN}" == 1 ]] && printf full || printf '%s videos' "${#VIDEO_INDEX_ARRAY[@]}")"
printf '  config hash:  %s\n' "${CONFIG_HASH}"
printf '  run config:   %s\n' "${RUN_CONFIG}"
printf '  artifacts:    %s\n' "${RUN_DIR}"
