#!/usr/bin/env bash
# Resume-safe PhaseFuse development/confirmatory experiment launcher.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_phasefuse_experiment.sh [options]

Dataset/run:
  --benchmark NAME          videomme or qvhighlights (default: videomme)
  --questions-file FILE     Benchmark annotation (default: dataset-specific)
  --dataset-root DIR        Benchmark root (default: datasets/<benchmark>)
  --run-dir DIR             Isolated artifact root
  --config FILE             PhaseFuse YAML (default: configs/phasefuse_icassp.yaml)
  --video-indices LIST      Comma/space list (default: 0..19 development subset)

Runtime:
  --python-bin CMD          Python executable (default: active venv python)
  --feature-model NAME      blip2/blip1/clip/siglip (default: blip2)
  --model-path ID           Optional feature checkpoint override
  --device DEVICE           Feature device (default: cuda:0)
  --feature-batch-size N    Feature batch size (default: 8)
  --frame-buffer-size N     Buffered target frames (default: 256)
  --cuda-device ID          CUDA_VISIBLE_DEVICES for Qwen (default: 0)
  --qwen-checkpoint ID      Default: Qwen/Qwen2.5-VL-7B-Instruct
  --n-bootstrap N           Video-cluster resamples (default: 10000)
  --limit N                 Optional lmms-eval smoke limit
  --skip-mllm               Stop after dense preprocessing/selection/export
  --force-preprocess        Recompute dense signals even if marker validates
  --force-selection         Recompute selectors/export even if marker validates
  -h, --help                Show this help

The default development run uses the first 20 VideoMME videos. Confirmatory
runs must pass a frozen, held-out --video-indices list. Outer origins are
evaluation perturbations; four inner phases are fused independently inside
every outer-origin cell.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

absolute_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then printf '%s\n' "${value}"; else printf '%s/%s\n' "${REPO_ROOT}" "${value}"; fi
}

split_list() {
  local raw="${1//,/ }"
  IFS=' ' read -r -a SPLIT_VALUES <<<"${raw}"
  ((${#SPLIT_VALUES[@]})) || die "list must not be empty"
}

require_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -- "$1" | awk '{print $1}'; else shasum -a 256 -- "$1" | awk '{print $1}'; fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then printf '%s' "$1" | sha256sum | awk '{print $1}'; else printf '%s' "$1" | shasum -a 256 | awk '{print $1}'; fi
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
    raise SystemExit(1)
seen = set()
for row in rows:
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise SystemExit(1)
    path = pathlib.Path(row["path"])
    if path in seen or not path.is_file() or path.stat().st_size != row["size_bytes"]:
        raise SystemExit(1)
    seen.add(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        raise SystemExit(1)
PY
}

write_marker() {
  local marker="$1" fingerprint="$2"; shift 2
  local temporary="${marker}.tmp" output
  {
    printf 'fingerprint=%s\n' "${fingerprint}"
    printf 'completed_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    for output in "$@"; do printf 'output=%s|%s\n' "${output}" "$(sha256_file "${output}")"; done
  } >"${temporary}"
  mv -f -- "${temporary}" "${marker}"
}

BENCHMARK="videomme"
QUESTIONS_FILE_RAW=""
DATASET_ROOT_RAW=""
RUN_DIR_RAW=""
CONFIG_RAW="configs/phasefuse_icassp.yaml"
VIDEO_INDICES_RAW="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
PYTHON_BIN="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.venvs/wfs-sb-a100/bin/python}"
FEATURE_MODEL="blip2"
MODEL_PATH=""
DEVICE="cuda:0"
FEATURE_BATCH_SIZE="8"
FRAME_BUFFER_SIZE="256"
CUDA_DEVICE="0"
QWEN_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"
N_BOOTSTRAP="10000"
LIMIT=""
SKIP_MLLM=0
FORCE_PREPROCESS=0
FORCE_SELECTION=0

while (($#)); do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --questions-file) QUESTIONS_FILE_RAW="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT_RAW="$2"; shift 2 ;;
    --run-dir) RUN_DIR_RAW="$2"; shift 2 ;;
    --config) CONFIG_RAW="$2"; shift 2 ;;
    --video-indices) VIDEO_INDICES_RAW="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --feature-model) FEATURE_MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --feature-batch-size) FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --frame-buffer-size) FRAME_BUFFER_SIZE="$2"; shift 2 ;;
    --cuda-device) CUDA_DEVICE="$2"; shift 2 ;;
    --qwen-checkpoint) QWEN_CHECKPOINT="$2"; shift 2 ;;
    --n-bootstrap) N_BOOTSTRAP="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --skip-mllm) SKIP_MLLM=1; shift ;;
    --force-preprocess) FORCE_PREPROCESS=1; shift ;;
    --force-selection) FORCE_SELECTION=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

case "${BENCHMARK,,}" in
  videomme)
    BENCHMARK="videomme"
    QUESTIONS_FILE_RAW="${QUESTIONS_FILE_RAW:-datasets/videomme/videomme_json_file.json}"
    DATASET_ROOT_RAW="${DATASET_ROOT_RAW:-datasets/videomme}"
    ;;
  qvhighlights|qvh)
    BENCHMARK="qvhighlights"
    QUESTIONS_FILE_RAW="${QUESTIONS_FILE_RAW:-datasets/qvhighlights/highlight_val_release.jsonl}"
    DATASET_ROOT_RAW="${DATASET_ROOT_RAW:-datasets/qvhighlights}"
    ((SKIP_MLLM)) || die "QVHighlights has no lmms-eval QA task; pass --skip-mllm"
    ;;
  *) die "unsupported benchmark: ${BENCHMARK}" ;;
esac

RUN_DIR_RAW="${RUN_DIR_RAW:-artifacts/phasefuse_${BENCHMARK}_dev20}"

QUESTIONS_FILE="$(absolute_path "${QUESTIONS_FILE_RAW}")"
DATASET_ROOT="$(absolute_path "${DATASET_ROOT_RAW}")"
RUN_DIR="$(absolute_path "${RUN_DIR_RAW}")"
CONFIG_FILE="$(absolute_path "${CONFIG_RAW}")"
[[ -x "${PYTHON_BIN}" ]] || die "Python executable is missing: ${PYTHON_BIN}"
[[ -s "${QUESTIONS_FILE}" ]] || die "questions file is missing: ${QUESTIONS_FILE}"
[[ -d "${DATASET_ROOT}" ]] || die "dataset root is missing: ${DATASET_ROOT}"
[[ -s "${CONFIG_FILE}" ]] || die "PhaseFuse config is missing: ${CONFIG_FILE}"
require_positive_integer --feature-batch-size "${FEATURE_BATCH_SIZE}"
require_positive_integer --frame-buffer-size "${FRAME_BUFFER_SIZE}"
require_positive_integer --n-bootstrap "${N_BOOTSTRAP}"
split_list "${VIDEO_INDICES_RAW}"
VIDEO_INDICES=("${SPLIT_VALUES[@]}")
for value in "${VIDEO_INDICES[@]}"; do [[ "${value}" =~ ^[0-9]+$ ]] || die "video index must be non-negative: ${value}"; done

mapfile -t CONFIG_CONTRACT < <(
  cd -- "${REPO_ROOT}"
  "${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import math
import sys

from phase_stable.phasefuse_experiment import load_phasefuse_config

loaded = load_phasefuse_config(sys.argv[1])
sampling = loaded.sampling
num_outer = sampling.get("num_outer_origins", 5)
num_inner = sampling.get("num_inner_phases", 4)
base_fps = sampling.get("base_sample_fps", 1.0)
dense_fps = sampling.get("dense_sample_fps", float(base_fps) * int(num_inner))
for name, value in (("num_outer_origins", num_outer), ("num_inner_phases", num_inner)):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SystemExit(f"sampling.{name} must be a positive integer")
if num_inner != loaded.experiment.num_phases:
    raise SystemExit("sampling.num_inner_phases must equal phasefuse.num_phases")
if (
    isinstance(base_fps, bool)
    or not isinstance(base_fps, (int, float))
    or not math.isfinite(float(base_fps))
    or float(base_fps) <= 0
    or isinstance(dense_fps, bool)
    or not isinstance(dense_fps, (int, float))
    or not math.isclose(float(dense_fps), float(base_fps) * num_inner)
):
    raise SystemExit("sampling.dense_sample_fps must equal base_sample_fps*num_inner_phases")
if not {"dense_swt", "phasefuse"}.issubset(loaded.experiment.methods):
    raise SystemExit("phasefuse.methods must include dense_swt and phasefuse")
print(loaded.experiment.frame_budget)
print(num_outer)
print(",".join(loaded.experiment.methods))
PY
)
((${#CONFIG_CONTRACT[@]} == 3)) || die "failed to read PhaseFuse config contract"
FRAME_BUDGET="${CONFIG_CONTRACT[0]}"
NUM_OUTER_ORIGINS="${CONFIG_CONTRACT[1]}"
require_positive_integer phasefuse.frame_budget "${FRAME_BUDGET}"
require_positive_integer sampling.num_outer_origins "${NUM_OUTER_ORIGINS}"
split_list "${CONFIG_CONTRACT[2]}"
METHODS=("${SPLIT_VALUES[@]}")
ORIGINS=()
for ((origin_id = 0; origin_id < NUM_OUTER_ORIGINS; origin_id++)); do
  ORIGINS+=("${origin_id}")
done
mkdir -p -- "${RUN_DIR}"
exec 9>"${RUN_DIR}/.phasefuse.lock"
flock -n 9 || die "another PhaseFuse launcher owns ${RUN_DIR}"

SIGNALS="${RUN_DIR}/preprocess/dense_signals.jsonl"
PREPROCESS_SUMMARY="${RUN_DIR}/preprocess/preprocess_summary.json"
FEATURE_BUNDLE="${RUN_DIR}/preprocess/feature_bundle.jsonl"
SOURCE_BUNDLE="${RUN_DIR}/preprocess/source_video_bundle.jsonl"
MULTIPHASE_MANIFESTS="${RUN_DIR}/preprocess/multiphase_manifests.jsonl"
PREPROCESS_CATALOG="${RUN_DIR}/preprocess/catalog.jsonl"
PREPROCESS_RUN_MANIFEST="${RUN_DIR}/preprocess/manifest/run_manifest.json"
PREPROCESS_ENVIRONMENT="${RUN_DIR}/preprocess/manifest/environment.json"
TRACES="${RUN_DIR}/selection/traces.jsonl"
SELECTION_SUMMARY="${RUN_DIR}/selection/summary.json"
ANALYSIS_SUMMARY="${RUN_DIR}/selection/analysis_summary.json"
TRACE_ARRAY_BUNDLE="${RUN_DIR}/selection/trace_array_bundle.jsonl"
SELECTION_RUN_MANIFEST="${RUN_DIR}/selection/manifest/run_manifest.json"
SELECTION_ENVIRONMENT="${RUN_DIR}/selection/manifest/environment.json"
KEYFRAME_DIR="${RUN_DIR}/keyframes"
PREDICTIONS="${RUN_DIR}/predictions.jsonl"
DOWNSTREAM_SUMMARY="${RUN_DIR}/phasefuse_downstream_summary.json"

WFS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}" phase_stable wfs preprocess scripts configs)"
PACKAGE_SIGNATURE="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import json
import platform

names = ("torch", "transformers", "PyWavelets", "scipy", "numpy", "av", "lmms_eval", "accelerate", "qwen-vl-utils")
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python": platform.python_version(), "packages": versions}, sort_keys=True))
PY
)"
case "${FEATURE_MODEL}" in
  blip2) DEFAULT_FEATURE_CHECKPOINT="Salesforce/blip2-itm-vit-g" ;;
  blip1) DEFAULT_FEATURE_CHECKPOINT="Salesforce/blip-itm-base-coco" ;;
  clip) DEFAULT_FEATURE_CHECKPOINT="openai/clip-vit-base-patch32" ;;
  siglip) DEFAULT_FEATURE_CHECKPOINT="google/siglip-so400m-patch14-384" ;;
  *) die "unsupported feature model: ${FEATURE_MODEL}" ;;
esac
FEATURE_CHECKPOINT_REQUESTED="${MODEL_PATH:-${DEFAULT_FEATURE_CHECKPOINT}}"
if [[ -d "${FEATURE_CHECKPOINT_REQUESTED}" ]]; then
  RESOLVED_FEATURE_CHECKPOINT="$(cd -- "${FEATURE_CHECKPOINT_REQUESTED}" && pwd -P)"
else
  RESOLVED_FEATURE_CHECKPOINT="$(env -u HF_TOKEN "${PYTHON_BIN}" - "${FEATURE_CHECKPOINT_REQUESTED}" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

print(pathlib.Path(snapshot_download(repo_id=sys.argv[1])).resolve())
PY
)"
fi
[[ -d "${RESOLVED_FEATURE_CHECKPOINT}" ]] || \
  die "resolved feature checkpoint is missing: ${RESOLVED_FEATURE_CHECKPOINT}"
if [[ "${RESOLVED_FEATURE_CHECKPOINT}" == */snapshots/* ]]; then
  FEATURE_CHECKPOINT_SIGNATURE="snapshot:$(basename -- "${RESOLVED_FEATURE_CHECKPOINT}")"
else
  FEATURE_CHECKPOINT_SIGNATURE="content:$(directory_content_hash "${RESOLVED_FEATURE_CHECKPOINT}")"
fi
CODE_HASH="$(sha256_text "$(sha256_file "${CONFIG_FILE}")|$(sha256_file "${QUESTIONS_FILE}")|${WFS_SOURCE_SHA}|${PACKAGE_SIGNATURE}")"
INDEX_TOKEN="$(IFS=,; printf '%s' "${VIDEO_INDICES[*]}")"
PREPROCESS_FP="$(sha256_text "phasefuse_preprocess_v3|${CODE_HASH}|${BENCHMARK}|${DATASET_ROOT}|${INDEX_TOKEN}|${FEATURE_MODEL}|${FEATURE_CHECKPOINT_REQUESTED}|${RESOLVED_FEATURE_CHECKPOINT}|${FEATURE_CHECKPOINT_SIGNATURE}|${DEVICE}|${FEATURE_BATCH_SIZE}|${FRAME_BUFFER_SIZE}")"
PREPROCESS_MARKER="${RUN_DIR}/preprocess/.complete"

if ((FORCE_PREPROCESS == 0)) \
  && valid_marker "${PREPROCESS_MARKER}" "${PREPROCESS_FP}" \
    "${SIGNALS}" "${PREPROCESS_SUMMARY}" "${FEATURE_BUNDLE}" "${SOURCE_BUNDLE}" \
    "${MULTIPHASE_MANIFESTS}" "${PREPROCESS_CATALOG}" \
    "${PREPROCESS_RUN_MANIFEST}" "${PREPROCESS_ENVIRONMENT}" \
  && validate_artifact_bundle "${FEATURE_BUNDLE}" \
  && validate_artifact_bundle "${SOURCE_BUNDLE}"; then
  printf 'SKIP: dense PhaseFuse preprocessing already complete\n'
else
  preprocess_args=(
    "${PYTHON_BIN}" -m phase_stable preprocess-phasefuse
    --benchmark "${BENCHMARK}"
    --questions-file "${QUESTIONS_FILE}"
    --dataset-root "${DATASET_ROOT}"
    --output-dir "${RUN_DIR}/preprocess"
    --signal-jsonl "${SIGNALS}"
    --config "${CONFIG_FILE}"
    --video-indices "${VIDEO_INDICES[@]}"
    --feature-model "${FEATURE_MODEL}"
    --device "${DEVICE}"
    --batch-size "${FEATURE_BATCH_SIZE}"
    --frame-buffer-size "${FRAME_BUFFER_SIZE}"
  )
  preprocess_args+=(--model-path "${RESOLVED_FEATURE_CHECKPOINT}")
  printf 'RUN: dense physical-phase preprocessing\n'
  (cd -- "${REPO_ROOT}" && "${preprocess_args[@]}")
  validate_artifact_bundle "${FEATURE_BUNDLE}" || die "feature artifact bundle failed validation"
  validate_artifact_bundle "${SOURCE_BUNDLE}" || die "source-video bundle failed validation"
  write_marker "${PREPROCESS_MARKER}" "${PREPROCESS_FP}" \
    "${SIGNALS}" "${PREPROCESS_SUMMARY}" "${FEATURE_BUNDLE}" "${SOURCE_BUNDLE}" \
    "${MULTIPHASE_MANIFESTS}" "${PREPROCESS_CATALOG}" \
    "${PREPROCESS_RUN_MANIFEST}" "${PREPROCESS_ENVIRONMENT}"
fi

SELECTION_FP="$(sha256_text "phasefuse_selection_v2|${PREPROCESS_FP}|$(sha256_file "${SIGNALS}")|$(sha256_file "${FEATURE_BUNDLE}")|${METHODS[*]}|${N_BOOTSTRAP}")"
SELECTION_MARKER="${RUN_DIR}/selection/.complete"
if ((FORCE_SELECTION == 0)) \
  && valid_marker "${SELECTION_MARKER}" "${SELECTION_FP}" \
    "${TRACES}" "${SELECTION_SUMMARY}" "${ANALYSIS_SUMMARY}" "${TRACE_ARRAY_BUNDLE}" \
    "${SELECTION_RUN_MANIFEST}" "${SELECTION_ENVIRONMENT}" \
  && validate_artifact_bundle "${TRACE_ARRAY_BUNDLE}"; then
  printf 'SKIP: PhaseFuse selectors already complete\n'
else
  printf 'RUN: compute-matched PhaseFuse selectors\n'
  (cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable run-phasefuse \
    "${SIGNALS}" "${RUN_DIR}/selection" --config "${CONFIG_FILE}" \
    --methods "${METHODS[@]}" --baseline-method dense_swt \
    --treatment-method phasefuse --n-bootstrap "${N_BOOTSTRAP}")
  validate_artifact_bundle "${TRACE_ARRAY_BUNDLE}" || die "trace-array bundle failed validation"
  write_marker "${SELECTION_MARKER}" "${SELECTION_FP}" \
    "${TRACES}" "${SELECTION_SUMMARY}" "${ANALYSIS_SUMMARY}" "${TRACE_ARRAY_BUNDLE}" \
    "${SELECTION_RUN_MANIFEST}" "${SELECTION_ENVIRONMENT}"
fi

# Export is deterministic and cheap.  Re-run it even after a valid selection
# resume so deleted/stale annotation JSON cannot survive behind a stage marker.
if [[ "${BENCHMARK}" == "videomme" ]]; then
  mkdir -p -- "${KEYFRAME_DIR}"
  (cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable export-keyframes \
    --traces "${TRACES}" --benchmark videomme \
      --questions-file "${QUESTIONS_FILE}" --dataset-root "${DATASET_ROOT}" \
      --output-dir "${KEYFRAME_DIR}" --methods "${METHODS[@]}" \
      --origin-ids "${ORIGINS[@]}" --expected-budget "${FRAME_BUDGET}")
fi

if ((SKIP_MLLM)); then
  printf 'PhaseFuse selection experiment complete (MLLM skipped): %s\n' "${RUN_DIR}"
  exit 0
fi

methods_csv="$(IFS=,; printf '%s' "${METHODS[*]}")"
origins_csv="$(IFS=,; printf '%s' "${ORIGINS[*]}")"
mllm_args=(
  --benchmark videomme
  --keyframe-dir "${KEYFRAME_DIR}"
  --output-root "${RUN_DIR}/mllm"
  --methods "${methods_csv}"
  --origins "${origins_csv}"
  --cuda-device "${CUDA_DEVICE}"
  --max-num-frames "${FRAME_BUDGET}"
  --max-pixels 200704
  --attention sdpa
  --python-bin "${PYTHON_BIN}"
  --converter-python "${PYTHON_BIN}"
  --repo-root "${REPO_ROOT}"
  --predictions-output "${PREDICTIONS}"
)
[[ -z "${LIMIT}" ]] || mllm_args+=(--limit "${LIMIT}")
printf 'RUN/RESUME: Qwen method x outer-origin grid (%s cells)\n' "$((${#METHODS[@]} * ${#ORIGINS[@]}))"

if [[ -d "${QWEN_CHECKPOINT}" ]]; then
  RESOLVED_QWEN="$(cd -- "${QWEN_CHECKPOINT}" && pwd -P)"
  QWEN_CHECKPOINT_SIGNATURE="content:$(directory_content_hash "${RESOLVED_QWEN}")"
else
  RESOLVED_QWEN="$(env -u HF_TOKEN "${PYTHON_BIN}" - "${QWEN_CHECKPOINT}" <<'PY'
import pathlib
import sys
from huggingface_hub import snapshot_download

print(pathlib.Path(snapshot_download(repo_id=sys.argv[1])).resolve())
PY
)"
  QWEN_CHECKPOINT_SIGNATURE="snapshot:$(basename -- "${RESOLVED_QWEN}")"
fi
[[ -d "${RESOLVED_QWEN}" ]] || die "resolved Qwen snapshot is missing: ${RESOLVED_QWEN}"
LMMS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}/lmms-eval" lmms_eval)"
RUNTIME_PROVENANCE="${RUN_DIR}/mllm_runtime_provenance.json"
"${PYTHON_BIN}" - "${RUNTIME_PROVENANCE}" "${CODE_HASH}" "${LMMS_SOURCE_SHA}" \
  "${QWEN_CHECKPOINT}" "${RESOLVED_QWEN}" "${QWEN_CHECKPOINT_SIGNATURE}" \
  "${PACKAGE_SIGNATURE}" <<'PY'
import json
import pathlib
import sys

output, code_sha, lmms_sha, requested, resolved, checkpoint_signature, packages = sys.argv[1:]
payload = {
    "schema_version": 1,
    "wfs_code_sha256": code_sha,
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
temporary.replace(path)
PY
MLLM_RUNTIME_SIGNATURE="$(sha256_file "${RUNTIME_PROVENANCE}")"
mllm_args+=(--qwen-checkpoint "${RESOLVED_QWEN}" --runtime-signature "${MLLM_RUNTIME_SIGNATURE}")
bash "${SCRIPT_DIR}/run_mllm_grid.sh" "${mllm_args[@]}"

if [[ -n "${LIMIT}" ]]; then
  printf 'PhaseFuse MLLM smoke grid complete; strict full-cohort downstream analysis skipped because --limit=%s.\n' "${LIMIT}"
  exit 0
fi

printf 'RUN: stable-correct/stable-wrong and uncertainty analysis\n'
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable evaluate-phasefuse-predictions \
  --predictions "${PREDICTIONS}" --traces "${TRACES}" \
  --output "${DOWNSTREAM_SUMMARY}" --baseline-method dense_swt \
  --treatment-method phasefuse --expected-methods "${METHODS[@]}" \
  --n-bootstrap "${N_BOOTSTRAP}")

printf '\nPhaseFuse experiment complete.\n'
printf '  Dense signals: %s\n' "${SIGNALS}"
printf '  Selection:     %s\n' "${ANALYSIS_SUMMARY}"
printf '  Predictions:   %s\n' "${PREDICTIONS}"
printf '  Downstream:    %s\n' "${DOWNSTREAM_SUMMARY}"
