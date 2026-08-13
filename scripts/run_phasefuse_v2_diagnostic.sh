#!/usr/bin/env bash
# Resume-safe PhaseFuse-v2 diagnostic using the completed dev20 base run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_phasefuse_v2_diagnostic.sh [options]

Inputs:
  --base-run-dir DIR       Completed PhaseFuse dev20 root
                           (default: artifacts/phasefuse_videomme_dev20)
  --questions-file FILE    VideoMME annotation JSON
  --dataset-root DIR       VideoMME dataset root
  --config FILE            PhaseFuse-v2 diagnostic YAML
                           (default: configs/phasefuse_v2_dev20.yaml)

Runtime:
  --python-bin CMD         Python executable (default: active A100 venv)
  --cuda-device ID         CUDA_VISIBLE_DEVICES for Qwen (default: 0)
  --n-bootstrap N          Video-cluster resamples (default: 10000)
  --skip-mllm              Run selectors/export only; do not require base MLLM
  --force-selection        Recompute the two selector arms
  -h, --help               Show this help

Outputs are isolated under BASE_RUN_DIR/v2_diagnostic.  Dense signals and
feature arrays are read in place from the completed base preprocess stage.
Only phasefuse_v2 is sent to Qwen; the original, independently verified
dense_swt predictions are reused for the paired downstream comparison.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

absolute_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${value}"
  fi
}

require_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$1" | awk '{print $1}'
  else
    shasum -a 256 -- "$1" | awk '{print $1}'
  fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "$1" | sha256sum | awk '{print $1}'
  else
    printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
  fi
}

source_tree_hash() {
  local root="$1"
  shift
  (
    cd -- "${root}"
    find "$@" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \) -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

directory_content_hash() {
  local root="$1"
  (
    cd -- "${root}"
    find -L . -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

valid_marker() {
  local marker="$1" fingerprint="$2"
  shift 2
  [[ -s "${marker}" && "$(sed -n 's/^fingerprint=//p' "${marker}")" == "${fingerprint}" ]] || return 1
  local output expected output_count
  output_count="$(grep -c '^output=' "${marker}" || true)"
  [[ "${output_count}" == "$#" ]] || return 1
  for output in "$@"; do
    [[ -s "${output}" ]] || return 1
    expected="output=${output}|$(sha256_file "${output}")"
    grep -Fqx -- "${expected}" "${marker}" || return 1
  done
}

write_marker() {
  local marker="$1" fingerprint="$2"
  shift 2
  local temporary="${marker}.tmp" output
  {
    printf 'fingerprint=%s\n' "${fingerprint}"
    printf 'completed_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    for output in "$@"; do
      printf 'output=%s|%s\n' "${output}" "$(sha256_file "${output}")"
    done
  } >"${temporary}"
  mv -f -- "${temporary}" "${marker}"
}

validate_artifact_bundle() {
  local bundle="$1"
  [[ -s "${bundle}" ]] || return 1
  "${PYTHON_BIN}" - "${bundle}" <<'PY'
import hashlib
import json
import pathlib
import sys

bundle = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in bundle.read_text(encoding="utf-8").splitlines()]
if not rows:
    raise SystemExit("artifact bundle is empty")
seen = set()
for number, row in enumerate(rows, 1):
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes"}:
        raise SystemExit(f"invalid artifact bundle row {number}")
    path = pathlib.Path(row["path"]).resolve()
    if path in seen or not path.is_file() or path.stat().st_size != row["size_bytes"]:
        raise SystemExit(f"invalid artifact in bundle row {number}: {path}")
    seen.add(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise SystemExit(f"artifact checksum mismatch: {path}")
PY
}

validate_base_preprocess() {
  "${PYTHON_BIN}" - \
    "${BASE_PREPROCESS_MARKER}" "${BASE_SIGNALS}" \
    "${BASE_FEATURE_BUNDLE}" "${BASE_PREPROCESS_SUMMARY}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
from collections import defaultdict

marker_path, signals_path, bundle_path, summary_path = map(pathlib.Path, sys.argv[1:])

try:
    marker_lines = marker_path.read_text(encoding="utf-8").splitlines()
except FileNotFoundError as exc:
    raise SystemExit(f"base preprocess completion marker is missing: {marker_path}") from exc

fingerprints = [line.split("=", 1)[1] for line in marker_lines if line.startswith("fingerprint=")]
if len(fingerprints) != 1 or not re.fullmatch(r"[0-9a-f]{64}", fingerprints[0]):
    raise SystemExit("base preprocess marker has no valid fingerprint")

outputs = {}
for line in marker_lines:
    if not line.startswith("output="):
        continue
    value = line[len("output="):]
    path_text, separator, expected_hash = value.rpartition("|")
    if not separator or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise SystemExit("malformed output entry in base preprocess marker")
    path = pathlib.Path(path_text).resolve()
    if path in outputs:
        raise SystemExit(f"duplicate base marker output: {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing/empty base marker output: {path}")
    observed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise SystemExit(f"base marker checksum mismatch: {path}")
    outputs[path] = expected_hash

for required in (signals_path.resolve(), bundle_path.resolve(), summary_path.resolve()):
    if required not in outputs:
        raise SystemExit(f"base marker does not authenticate required artifact: {required}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_summary = {
    "command": "preprocess-phasefuse",
    "num_videos": 20,
    "num_signal_records": 300,
    "num_outer_origins": 5,
    "num_inner_phases": 4,
}
for name, expected in expected_summary.items():
    if summary.get(name) != expected:
        raise SystemExit(f"base preprocess summary {name} must be {expected!r}")

bundle_rows = [
    json.loads(line) for line in bundle_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
bundle_paths = {pathlib.Path(row["path"]).resolve() for row in bundle_rows}
if not bundle_paths:
    raise SystemExit("base feature bundle is empty")

rows = []
for number, line in enumerate(signals_path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        raise SystemExit(f"blank line in base signals at {number}")
    row = json.loads(line)
    if not isinstance(row, dict):
        raise SystemExit(f"base signal row {number} is not an object")
    rows.append(row)
if len(rows) != 300:
    raise SystemExit(f"base signals must contain exactly 300 rows, found {len(rows)}")

origins_by_item = defaultdict(set)
seen = set()
video_ids = set()
signal_features = set()
for row in rows:
    if row.get("dataset") != "videomme":
        raise SystemExit("base signal dataset must be videomme")
    origin = row.get("origin_id")
    if isinstance(origin, bool) or not isinstance(origin, int) or origin not in range(5):
        raise SystemExit("base signal origin grid must be exactly 0..4")
    item = (row.get("dataset"), str(row.get("video_id")), str(row.get("question_id")))
    key = (*item, origin)
    if key in seen:
        raise SystemExit(f"duplicate base signal key: {key}")
    seen.add(key)
    origins_by_item[item].add(origin)
    video_ids.add(item[1])
    feature_value = row.get("visual_features_path")
    if not isinstance(feature_value, str) or not feature_value.strip():
        raise SystemExit(f"base signal has no feature path: {key}")
    signal_features.add(pathlib.Path(feature_value).resolve())

if len(origins_by_item) != 60 or len(video_ids) != 20:
    raise SystemExit("base signal cohort must contain 60 questions from 20 videos")
if any(origins != set(range(5)) for origins in origins_by_item.values()):
    raise SystemExit("base signal item does not contain the complete 5-origin grid")
if signal_features != bundle_paths:
    raise SystemExit("base signal feature paths do not match feature bundle exactly")
PY
}

validate_trace_grid() {
  "${PYTHON_BIN}" - "${TRACES}" <<'PY'
import json
import pathlib
import sys
from collections import defaultdict

path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 600:
    raise SystemExit(f"selector trace must contain exactly 600 rows, found {len(rows)}")
methods = {"dense_swt", "phasefuse_v2"}
observed = {row.get("method") for row in rows}
if observed != methods:
    raise SystemExit(f"selector methods do not match exactly: {observed}")
grid = defaultdict(set)
seen = set()
for row in rows:
    key = (row.get("dataset"), str(row.get("video_id")), str(row.get("question_id")))
    origin = row.get("origin_id")
    method = row.get("method")
    full_key = (*key, origin, method)
    if key[0] != "videomme" or origin not in range(5) or full_key in seen:
        raise SystemExit(f"invalid/duplicate selector key: {full_key}")
    seen.add(full_key)
    grid[(key, method)].add(origin)
    selected = row.get("selected_indices")
    if (
        not isinstance(selected, list)
        or len(selected) != 16
        or len(set(selected)) != 16
        or selected != sorted(selected)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in selected)
    ):
        raise SystemExit(f"selector did not produce exact sorted K=16: {full_key}")
if len(grid) != 120 or any(origins != set(range(5)) for origins in grid.values()):
    raise SystemExit("selector traces are not a complete 60-item x 2-method x 5-origin grid")
PY
}

strict_merge_predictions() {
  "${PYTHON_BIN}" - "$@" <<'PY'
# STRICT_MERGE_PY_BEGIN
import hashlib
import json
import os
import pathlib
import sys
from collections import defaultdict

base_path, verified_path, new_path, summary_path, output_path = map(pathlib.Path, sys.argv[1:])
required = {
    "dataset", "video_id", "question_id", "origin_id",
    "method", "prediction", "gold",
}

def read_jsonl(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise SystemExit(f"blank prediction line at {path}:{number}")
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != required:
            raise SystemExit(f"prediction row must have exactly seven fields at {path}:{number}")
        if row["dataset"] != "videomme":
            raise SystemExit(f"unexpected prediction dataset at {path}:{number}")
        if not isinstance(row["video_id"], str) or not row["video_id"]:
            raise SystemExit(f"invalid video_id at {path}:{number}")
        if not isinstance(row["question_id"], str) or not row["question_id"]:
            raise SystemExit(f"invalid question_id at {path}:{number}")
        if isinstance(row["origin_id"], bool) or row["origin_id"] not in range(5):
            raise SystemExit(f"origin_id must be exactly one of 0..4 at {path}:{number}")
        if not isinstance(row["prediction"], str):
            raise SystemExit(f"prediction must retain the official parser string at {path}:{number}")
        if row["gold"] not in tuple("ABCDE"):
            raise SystemExit(f"gold must be a normalized choice at {path}:{number}")
        rows.append(row)
    return rows

def key(row):
    return row["dataset"], row["video_id"], row["question_id"], row["origin_id"]

def exact_arm(rows, method, expected_count=300):
    if len(rows) != expected_count or {row["method"] for row in rows} != {method}:
        raise SystemExit(f"{method} arm must contain exactly {expected_count} exclusive rows")
    result = {}
    origins = defaultdict(set)
    gold = {}
    for row in rows:
        row_key = key(row)
        if row_key in result:
            raise SystemExit(f"duplicate prediction key in {method}: {row_key}")
        result[row_key] = row
        item = row_key[:3]
        origins[item].add(row_key[3])
        previous = gold.setdefault(item, row["gold"])
        if previous != row["gold"]:
            raise SystemExit(f"gold changes across origins in {method}: {item}")
    if len(origins) != 60 or len({item[1] for item in origins}) != 20:
        raise SystemExit(f"{method} must contain 60 questions from 20 videos")
    if any(values != set(range(5)) for values in origins.values()):
        raise SystemExit(f"{method} does not contain an exact 5-origin grid")
    return result

base_rows = read_jsonl(base_path)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
if summary.get("predictions_sha256") != base_sha:
    raise SystemExit("base downstream summary does not authenticate base predictions")
if summary.get("num_prediction_rows") != 2400 or len(base_rows) != 2400:
    raise SystemExit("completed base predictions must contain exactly 2400 rows")

base_arm_rows = [row for row in base_rows if row["method"] == "dense_swt"]
base_arm = exact_arm(base_arm_rows, "dense_swt")
verified_arm = exact_arm(read_jsonl(verified_path), "dense_swt")
if base_arm != verified_arm:
    raise SystemExit("original dense_swt rows do not match reconstruction from completed cells")

new_arm = exact_arm(read_jsonl(new_path), "phasefuse_v2")
if set(base_arm) != set(new_arm):
    raise SystemExit("dense_swt and phasefuse_v2 prediction keys do not align exactly")
for row_key in base_arm:
    if base_arm[row_key]["gold"] != new_arm[row_key]["gold"]:
        raise SystemExit(f"gold mismatch between arms: {row_key}")

merged = []
for arm in (base_arm, new_arm):
    merged.extend(arm[row_key] for row_key in sorted(arm))
if len(merged) != 600:
    raise SystemExit("strict merge did not produce exactly 600 rows")

output_path.parent.mkdir(parents=True, exist_ok=True)
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("w", encoding="utf-8", newline="\n") as handle:
    for row in merged:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
os.replace(temporary, output_path)
print(f"Strict merge wrote {len(merged)} rows to {output_path}")
# STRICT_MERGE_PY_END
PY
}

BASE_RUN_DIR_RAW="artifacts/phasefuse_videomme_dev20"
QUESTIONS_FILE_RAW="datasets/videomme/videomme_json_file.json"
DATASET_ROOT_RAW="datasets/videomme"
CONFIG_RAW="configs/phasefuse_v2_dev20.yaml"
PYTHON_BIN="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.venvs/wfs-sb-a100/bin/python}"
CUDA_DEVICE="0"
N_BOOTSTRAP="10000"
SKIP_MLLM=0
FORCE_SELECTION=0

while (($#)); do
  case "$1" in
    --base-run-dir) BASE_RUN_DIR_RAW="$2"; shift 2 ;;
    --questions-file) QUESTIONS_FILE_RAW="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT_RAW="$2"; shift 2 ;;
    --config) CONFIG_RAW="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
    --n-bootstrap) N_BOOTSTRAP="$2"; shift 2 ;;
    --skip-mllm) SKIP_MLLM=1; shift ;;
    --force-selection) FORCE_SELECTION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

BASE_RUN_DIR="$(absolute_path "${BASE_RUN_DIR_RAW}")"
RUN_DIR="${BASE_RUN_DIR}/v2_diagnostic"
QUESTIONS_FILE="$(absolute_path "${QUESTIONS_FILE_RAW}")"
DATASET_ROOT="$(absolute_path "${DATASET_ROOT_RAW}")"
CONFIG_FILE="$(absolute_path "${CONFIG_RAW}")"

[[ -x "${PYTHON_BIN}" ]] || die "Python executable is missing: ${PYTHON_BIN}"
[[ -d "${BASE_RUN_DIR}" ]] || die "completed base run is missing: ${BASE_RUN_DIR}"
[[ -s "${QUESTIONS_FILE}" ]] || die "questions file is missing: ${QUESTIONS_FILE}"
[[ -d "${DATASET_ROOT}" ]] || die "dataset root is missing: ${DATASET_ROOT}"
[[ -s "${CONFIG_FILE}" ]] || die "PhaseFuse-v2 config is missing: ${CONFIG_FILE}"
require_positive_integer --n-bootstrap "${N_BOOTSTRAP}"

mapfile -t CONFIG_CONTRACT < <(
  cd -- "${REPO_ROOT}"
  "${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import sys
from phase_stable.phasefuse_experiment import load_phasefuse_config

loaded = load_phasefuse_config(sys.argv[1])
if loaded.experiment.frame_budget != 16:
    raise SystemExit("phasefuse_v2 diagnostic requires exact frame_budget=16")
if loaded.experiment.num_phases != 4:
    raise SystemExit("phasefuse_v2 diagnostic requires exactly four inner phases")
if int(loaded.sampling.get("num_outer_origins", -1)) != 5:
    raise SystemExit("phasefuse_v2 diagnostic requires exactly five outer origins")
if not {"dense_swt", "phasefuse_v2"}.issubset(loaded.experiment.methods):
    raise SystemExit("config must enable dense_swt and phasefuse_v2")
print(loaded.experiment.frame_budget)
print(loaded.sampling["num_outer_origins"])
PY
)
[[ "${CONFIG_CONTRACT[*]}" == "16 5" ]] || die "invalid PhaseFuse-v2 config contract"
FRAME_BUDGET=16
METHODS=(dense_swt phasefuse_v2)
ORIGINS=(0 1 2 3 4)

mkdir -p -- "${RUN_DIR}"
exec 8>"${BASE_RUN_DIR}/.phasefuse.lock"
flock -n 8 || die "the base PhaseFuse run is still active: ${BASE_RUN_DIR}"
exec 9>"${RUN_DIR}/.phasefuse_v2.lock"
flock -n 9 || die "another PhaseFuse-v2 diagnostic owns ${RUN_DIR}"

BASE_PREPROCESS_MARKER="${BASE_RUN_DIR}/preprocess/.complete"
BASE_SIGNALS="${BASE_RUN_DIR}/preprocess/dense_signals.jsonl"
BASE_FEATURE_BUNDLE="${BASE_RUN_DIR}/preprocess/feature_bundle.jsonl"
BASE_PREPROCESS_SUMMARY="${BASE_RUN_DIR}/preprocess/preprocess_summary.json"
BASE_PREDICTIONS="${BASE_RUN_DIR}/predictions.jsonl"
BASE_DOWNSTREAM_SUMMARY="${BASE_RUN_DIR}/phasefuse_downstream_summary.json"
BASE_RUNTIME_PROVENANCE="${BASE_RUN_DIR}/mllm_runtime_provenance.json"

printf 'VERIFY: completed base preprocess artifacts\n'
validate_base_preprocess
validate_artifact_bundle "${BASE_FEATURE_BUNDLE}" || die "base feature bundle failed checksum validation"

TRACES="${RUN_DIR}/selection/traces.jsonl"
SELECTION_SUMMARY="${RUN_DIR}/selection/summary.json"
ANALYSIS_SUMMARY="${RUN_DIR}/selection/analysis_summary.json"
TRACE_ARRAY_BUNDLE="${RUN_DIR}/selection/trace_array_bundle.jsonl"
SELECTION_RUN_MANIFEST="${RUN_DIR}/selection/manifest/run_manifest.json"
SELECTION_ENVIRONMENT="${RUN_DIR}/selection/manifest/environment.json"
KEYFRAME_DIR="${RUN_DIR}/keyframes"
NEW_PREDICTIONS="${RUN_DIR}/phasefuse_v2_predictions.jsonl"
VERIFIED_BASELINE="${RUN_DIR}/dense_swt_predictions.verified.jsonl"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
DOWNSTREAM_SUMMARY="${RUN_DIR}/phasefuse_v2_downstream_summary.json"

WFS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}" phase_stable wfs preprocess scripts configs)"
SELECTION_FP="$(sha256_text "phasefuse_v2_selection_v1|$(sha256_file "${BASE_PREPROCESS_MARKER}")|$(sha256_file "${BASE_SIGNALS}")|$(sha256_file "${BASE_FEATURE_BUNDLE}")|$(sha256_file "${CONFIG_FILE}")|${WFS_SOURCE_SHA}|${METHODS[*]}|${N_BOOTSTRAP}")"
SELECTION_MARKER="${RUN_DIR}/selection/.complete"

if ((FORCE_SELECTION == 0)) \
  && valid_marker "${SELECTION_MARKER}" "${SELECTION_FP}" \
    "${TRACES}" "${SELECTION_SUMMARY}" "${ANALYSIS_SUMMARY}" \
    "${TRACE_ARRAY_BUNDLE}" "${SELECTION_RUN_MANIFEST}" "${SELECTION_ENVIRONMENT}" \
  && validate_artifact_bundle "${TRACE_ARRAY_BUNDLE}" \
  && validate_trace_grid; then
  printf 'SKIP: PhaseFuse-v2 selectors already complete\n'
else
  printf 'RUN: dense_swt + phasefuse_v2 selectors on authenticated base signals\n'
  (cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable run-phasefuse \
    "${BASE_SIGNALS}" "${RUN_DIR}/selection" --config "${CONFIG_FILE}" \
    --methods dense_swt phasefuse_v2 --baseline-method dense_swt \
    --treatment-method phasefuse_v2 --n-bootstrap "${N_BOOTSTRAP}")
  validate_artifact_bundle "${TRACE_ARRAY_BUNDLE}" || die "trace-array bundle failed validation"
  validate_trace_grid || die "PhaseFuse-v2 selector grid failed exact-K/cohort validation"
  write_marker "${SELECTION_MARKER}" "${SELECTION_FP}" \
    "${TRACES}" "${SELECTION_SUMMARY}" "${ANALYSIS_SUMMARY}" \
    "${TRACE_ARRAY_BUNDLE}" "${SELECTION_RUN_MANIFEST}" "${SELECTION_ENVIRONMENT}"
fi

# Export is deterministic and deliberately rerun so stale/missing keyframe JSON
# cannot hide behind a selector completion marker.
mkdir -p -- "${KEYFRAME_DIR}"
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable export-keyframes \
  --traces "${TRACES}" --benchmark videomme \
  --questions-file "${QUESTIONS_FILE}" --dataset-root "${DATASET_ROOT}" \
  --output-dir "${KEYFRAME_DIR}" --methods dense_swt phasefuse_v2 \
  --origin-ids 0 1 2 3 4 --expected-budget 16 --allow-annotation-subset)

if ((SKIP_MLLM)); then
  printf 'PhaseFuse-v2 selector diagnostic complete (MLLM skipped): %s\n' "${RUN_DIR}"
  exit 0
fi

[[ -s "${BASE_PREDICTIONS}" ]] || die "completed base predictions are missing: ${BASE_PREDICTIONS}"
[[ -s "${BASE_DOWNSTREAM_SUMMARY}" ]] || die "base downstream summary is missing: ${BASE_DOWNSTREAM_SUMMARY}"
[[ -s "${BASE_RUNTIME_PROVENANCE}" ]] || die "base Qwen provenance is missing: ${BASE_RUNTIME_PROVENANCE}"
[[ -d "${BASE_RUN_DIR}/mllm/videomme/dense_swt" ]] || die "base dense_swt MLLM cells are missing"

printf 'VERIFY: reconstruct original dense_swt predictions from completed cells\n'
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" "${SCRIPT_DIR}/convert_lmms_logs.py" \
  --grid-root "${BASE_RUN_DIR}/mllm/videomme" --benchmark videomme \
  --methods dense_swt --origins 0,1,2,3,4 --output "${VERIFIED_BASELINE}")

mapfile -t QWEN_CONTRACT < <(
  "${PYTHON_BIN}" - "${BASE_RUNTIME_PROVENANCE}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
required = (
    "schema_version", "wfs_code_sha256", "lmms_source_sha256",
    "checkpoint_requested", "checkpoint_snapshot_path",
    "checkpoint_snapshot_revision", "checkpoint_signature", "runtime",
)
if any(name not in payload for name in required):
    raise SystemExit("base Qwen provenance is incomplete")
for name in ("wfs_code_sha256", "lmms_source_sha256"):
    if not re.fullmatch(r"[0-9a-f]{64}", str(payload[name])):
        raise SystemExit(f"base Qwen provenance has invalid {name}")
resolved = pathlib.Path(payload["checkpoint_snapshot_path"])
if not resolved.is_absolute() or not resolved.is_dir():
    raise SystemExit("resolved base Qwen snapshot is missing")
if resolved.name != payload["checkpoint_snapshot_revision"]:
    raise SystemExit("base Qwen snapshot revision/path mismatch")
signature = str(payload["checkpoint_signature"])
if signature != f"snapshot:{resolved.name}" and not signature.startswith("content:"):
    raise SystemExit("base Qwen checkpoint signature is invalid")
if not isinstance(payload["runtime"], dict):
    raise SystemExit("base Qwen runtime provenance is invalid")
print(resolved)
print(payload["checkpoint_requested"])
print(signature)
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)
((${#QWEN_CONTRACT[@]} == 4)) || die "failed to validate base Qwen provenance"
RESOLVED_QWEN="${QWEN_CONTRACT[0]}"
QWEN_REQUESTED="${QWEN_CONTRACT[1]}"
QWEN_CHECKPOINT_SIGNATURE="${QWEN_CONTRACT[2]}"
BASE_PROVENANCE_SHA="${QWEN_CONTRACT[3]}"
if [[ "${QWEN_CHECKPOINT_SIGNATURE}" == content:* ]]; then
  [[ "${QWEN_CHECKPOINT_SIGNATURE#content:}" == "$(directory_content_hash "${RESOLVED_QWEN}")" ]] || \
    die "resolved Qwen checkpoint content no longer matches base provenance"
fi

LMMS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}/lmms-eval" lmms_eval)"
PACKAGE_SIGNATURE="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import json
import platform

names = ("torch", "transformers", "lmms_eval", "accelerate", "qwen-vl-utils")
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}, sort_keys=True))
PY
)"
RUNTIME_PROVENANCE="${RUN_DIR}/mllm_runtime_provenance.json"
"${PYTHON_BIN}" - "${RUNTIME_PROVENANCE}" "${BASE_RUNTIME_PROVENANCE}" \
  "${BASE_PROVENANCE_SHA}" "${WFS_SOURCE_SHA}" "${LMMS_SOURCE_SHA}" \
  "${QWEN_REQUESTED}" "${RESOLVED_QWEN}" "${QWEN_CHECKPOINT_SIGNATURE}" \
  "${PACKAGE_SIGNATURE}" <<'PY'
import json
import os
import pathlib
import sys

(output, inherited, inherited_sha, wfs_sha, lmms_sha, requested,
 resolved, checkpoint_signature, packages) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "inherited_base_provenance": str(pathlib.Path(inherited).resolve()),
    "inherited_base_provenance_sha256": inherited_sha,
    "wfs_code_sha256": wfs_sha,
    "lmms_source_sha256": lmms_sha,
    "checkpoint_requested": requested,
    "checkpoint_snapshot_path": resolved,
    "checkpoint_snapshot_revision": pathlib.Path(resolved).name,
    "checkpoint_signature": checkpoint_signature,
    "runtime": json.loads(packages),
}
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
MLLM_RUNTIME_SIGNATURE="$(sha256_file "${RUNTIME_PROVENANCE}")"

printf 'RUN/RESUME: Qwen phasefuse_v2 x 5 outer origins\n'
bash "${SCRIPT_DIR}/run_mllm_grid.sh" \
  --benchmark videomme --keyframe-dir "${KEYFRAME_DIR}" \
  --output-root "${RUN_DIR}/mllm" --methods phasefuse_v2 --origins 0,1,2,3,4 \
  --cuda-device "${CUDA_DEVICE}" --max-num-frames 16 --max-pixels 200704 \
  --attention sdpa --python-bin "${PYTHON_BIN}" --converter-python "${PYTHON_BIN}" \
  --repo-root "${REPO_ROOT}" --predictions-output "${NEW_PREDICTIONS}" \
  --qwen-checkpoint "${RESOLVED_QWEN}" --runtime-signature "${MLLM_RUNTIME_SIGNATURE}"

printf 'MERGE: authenticated dense_swt 300 + phasefuse_v2 300 rows\n'
strict_merge_predictions \
  "${BASE_PREDICTIONS}" "${VERIFIED_BASELINE}" "${NEW_PREDICTIONS}" \
  "${BASE_DOWNSTREAM_SUMMARY}" "${PREDICTIONS}"

printf 'RUN: paired PhaseFuse-v2 downstream analysis\n'
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable evaluate-phasefuse-predictions \
  --predictions "${PREDICTIONS}" --traces "${TRACES}" \
  --output "${DOWNSTREAM_SUMMARY}" --baseline-method dense_swt \
  --treatment-method phasefuse_v2 --expected-methods dense_swt phasefuse_v2 \
  --n-bootstrap "${N_BOOTSTRAP}")

printf '\nPhaseFuse-v2 diagnostic complete.\n'
printf '  Reused signals: %s\n' "${BASE_SIGNALS}"
printf '  Selection:      %s\n' "${ANALYSIS_SUMMARY}"
printf '  Predictions:    %s\n' "${PREDICTIONS}"
printf '  Downstream:     %s\n' "${DOWNSTREAM_SUMMARY}"
