#!/usr/bin/env bash
# Run a benchmark's lmms-eval method x sampling-origin grid safely.
#
# The WFS-SB lmms-eval patch catches some evaluation exceptions internally, so
# a zero process exit status alone is not a completion signal.  A cell is marked
# complete only when this invocation also creates non-empty aggregated-results
# and per-sample log artifacts with the patch's data-file-derived prefix.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DEFAULT_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CALLER_DIR="$(pwd -P)"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mllm_grid.sh --benchmark NAME --keyframe-dir DIR --output-root DIR [options]

Required:
  --benchmark NAME       videomme, mlvu, lvb, or longvideobench
  --keyframe-dir DIR     Directory containing <prefix>_<method>_originNN.json
  --output-root DIR      Root for independent benchmark/method/origin cells

Grid options:
  --methods LIST         Comma/space-separated methods (default: dwt,swt)
  --include-baselines    Append uniform and topk to the method grid
  --origins LIST         Comma/space-separated non-negative IDs (default: 0,1,2,3,4)
  --prefix TEXT          Override benchmark filename prefix
  --task TEXT            Override lmms-eval task
  --split TEXT           Override data_files split

Model/runtime options:
  --qwen-checkpoint ID   Qwen checkpoint (default: Qwen/Qwen2.5-VL-7B-Instruct)
  --cuda-device ID       Physical CUDA_VISIBLE_DEVICES value (default: 0)
  --max-num-frames N     Must match export budget (default: 16)
  --max-pixels N         Qwen per-frame maximum (default: 200704 = 256 visual tokens)
  --attention NAME       sdpa, flash_attention_2, or eager (default: sdpa)
  --batch-size N         lmms-eval batch size (default: 1)
  --python-bin COMMAND   Python executable (default: python)
  --converter-python CMD Python used for log conversion (default: python)
  --repo-root DIR        WFS-SB root used as lmms-eval working directory
  --limit N              Optional lmms-eval smoke-test item limit
  --predictions-output F Merged 7-field JSONL output
                         (default: <output-root>/<benchmark>/predictions.jsonl)
  --runtime-signature S  Optional immutable caller provenance included in every
                         cell fingerprint (code/model/package signature)
  --no-convert           Keep verified sample logs but skip prediction merging
  --force                Re-run cells even when a valid completion marker exists
  --dry-run              Validate inputs and print commands without running or writing
  -h, --help             Show this help

Benchmark mapping:
  videomme        task=videomme                split=test        prefix=videomme
  mlvu            task=mlvu_dev                split=test        prefix=mlvu
  lvb             task=longvideobench_val_v    split=validation  prefix=lvb
  longvideobench  task=longvideobench_val_v    split=validation  prefix=lvb

Resume and logs:
  Each cell writes console.log and, only after verified lmms-eval artifacts,
  .complete. Existing valid .complete cells are skipped unless --force is used.
  Resume rechecks the keyframe/config fingerprint and exact artifact size/SHA-256;
  stale inputs, settings, paths, or logs therefore cause that cell to run again.
  Failed cells stop the grid immediately and never receive a marker.

Security and prediction conversion:
  The patched lmms-eval CLI logs its evaluation-tracker arguments, including
  HF_TOKEN when that variable is present. This runner removes HF_TOKEN only from
  child environments and never prints token values. Cached/public Qwen downloads
  remain usable. After the grid, the official parser result stored in each task's
  metric payload is joined to the exported annotation by doc_id and merged into
  a strict 7-field predictions JSONL. Raw free-form responses are never reparsed.
EOF
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

make_absolute() {
  local value="$1"
  local base="$2"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s/%s\n' "${base}" "${value}"
  fi
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

split_list() {
  local raw="$1"
  raw="${raw//,/ }"
  IFS=' ' read -r -a SPLIT_VALUES <<<"${raw}"
  ((${#SPLIT_VALUES[@]} > 0)) || die "list must not be empty"
}

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "${path}" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "${path}" | awk '{print $1}'
  else
    die "sha256sum or shasum is required for safe resume markers"
  fi
}

sha256_text() {
  local value="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    printf '%s' "${value}" | sha256sum | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    printf '%s' "${value}" | shasum -a 256 | awk '{print $1}'
  else
    die "sha256sum or shasum is required for safe resume markers"
  fi
}

file_size_bytes() {
  local path="$1"
  wc -c <"${path}" | tr -d '[:space:]'
}

path_for_marker() {
  local path="$1"
  # A Windows-native converter cannot open MSYS paths such as /tmp/... .
  # Store host paths on Git Bash and ordinary absolute paths on Linux.
  case "${OSTYPE:-}" in
    msys*|cygwin*|mingw*)
      if command -v cygpath >/dev/null 2>&1; then
        cygpath -m -- "${path}"
        return
      fi
      ;;
  esac
  printf '%s\n' "${path}"
}

marker_field() {
  local marker="$1"
  local field="$2"
  sed -n "s/^${field}=//p" "${marker}" | tail -n 1
}

assert_unique_safe_methods() {
  local -A seen=()
  local method
  for method in "$@"; do
    [[ "${method}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe method token: ${method}"
    [[ -z "${seen[${method}]:-}" ]] || die "duplicate method: ${method}"
    seen["${method}"]=1
  done
}

assert_unique_origins() {
  local -A seen=()
  local origin
  for origin in "$@"; do
    [[ "${origin}" =~ ^[0-9]+$ ]] || die "origin IDs must be non-negative integers: ${origin}"
    [[ -z "${seen[${origin}]:-}" ]] || die "duplicate origin ID: ${origin}"
    seen["${origin}"]=1
  done
}

print_command() {
  local argument
  printf 'DRY-RUN: cd %q && env -u HF_TOKEN CUDA_VISIBLE_DEVICES=%q' \
    "${REPO_ROOT}" "${CUDA_DEVICE}"
  for argument in "$@"; do
    printf ' %q' "${argument}"
  done
  printf '\n'
}

# Globals populated by collect_artifacts.
RESULT_ARTIFACTS=()
SAMPLE_ARTIFACTS=()

collect_artifacts() {
  local cell_dir="$1"
  local data_name="$2"
  local task="$3"
  local newer_than="${4:-}"
  local -a newer_args=()
  local candidate

  RESULT_ARTIFACTS=()
  SAMPLE_ARTIFACTS=()
  [[ -d "${cell_dir}" ]] || return 0
  if [[ -n "${newer_than}" ]]; then
    newer_args=(-newer "${newer_than}")
  fi

  while IFS= read -r -d '' candidate; do
    if grep -q '^[[:space:]]*{' "${candidate}"; then
      RESULT_ARTIFACTS+=("${candidate}")
    fi
  done < <(
    find "${cell_dir}" -type f "${newer_args[@]}" \
      -name "${data_name}+*_results.json" -size +0c -print0
  )
  while IFS= read -r -d '' candidate; do
    if grep -q '^[[:space:]]*{' "${candidate}"; then
      SAMPLE_ARTIFACTS+=("${candidate}")
    fi
  done < <(
    find "${cell_dir}" -type f "${newer_args[@]}" \
      -name "${data_name}+*_samples_${task}.jsonl" -size +0c -print0
  )
}

valid_completion_marker() {
  local marker="$1"
  local expected_benchmark="$2"
  local expected_task="$3"
  local expected_method="$4"
  local expected_origin="$5"
  local expected_keyframe_path="$6"
  local expected_keyframe_sha="$7"
  local expected_fingerprint="$8"
  local result_path sample_path

  [[ -f "${marker}" ]] || return 1
  [[ "$(marker_field "${marker}" benchmark)" == "${expected_benchmark}" ]] || return 1
  [[ "$(marker_field "${marker}" task)" == "${expected_task}" ]] || return 1
  [[ "$(marker_field "${marker}" method)" == "${expected_method}" ]] || return 1
  [[ "$(marker_field "${marker}" origin_id)" == "${expected_origin}" ]] || return 1
  [[ "$(marker_field "${marker}" keyframe_json)" == "${expected_keyframe_path}" ]] || return 1
  [[ "$(marker_field "${marker}" keyframe_sha256)" == "${expected_keyframe_sha}" ]] || return 1
  [[ "$(marker_field "${marker}" config_fingerprint)" == "${expected_fingerprint}" ]] || return 1

  result_path="$(marker_field "${marker}" results_json)"
  sample_path="$(marker_field "${marker}" samples_jsonl)"
  [[ -s "${result_path}" && -s "${sample_path}" ]] || return 1
  grep -q '^[[:space:]]*{' "${result_path}" || return 1
  grep -q '^[[:space:]]*{' "${sample_path}" || return 1
  [[ "$(file_size_bytes "${result_path}")" == "$(marker_field "${marker}" results_size_bytes)" ]] || return 1
  [[ "$(file_size_bytes "${sample_path}")" == "$(marker_field "${marker}" samples_size_bytes)" ]] || return 1
  [[ "$(sha256_file "${result_path}")" == "$(marker_field "${marker}" results_sha256)" ]] || return 1
  [[ "$(sha256_file "${sample_path}")" == "$(marker_field "${marker}" samples_sha256)" ]] || return 1
  return 0
}

write_completion_marker() {
  local marker="$1"
  local benchmark="$2"
  local task="$3"
  local method="$4"
  local origin="$5"
  local keyframe_file="$6"
  local keyframe_sha="$7"
  local config_fingerprint="$8"
  local result_file="$9"
  local sample_file="${10}"
  local temporary="${marker}.tmp"
  local keyframe_marker result_marker sample_marker

  keyframe_marker="$(path_for_marker "${keyframe_file}")"
  result_marker="$(path_for_marker "${result_file}")"
  sample_marker="$(path_for_marker "${sample_file}")"

  {
    printf 'benchmark=%s\n' "${benchmark}"
    printf 'task=%s\n' "${task}"
    printf 'method=%s\n' "${method}"
    printf 'origin_id=%s\n' "${origin}"
    printf 'keyframe_json=%s\n' "${keyframe_marker}"
    printf 'keyframe_sha256=%s\n' "${keyframe_sha}"
    printf 'config_fingerprint=%s\n' "${config_fingerprint}"
    printf 'results_json=%s\n' "${result_marker}"
    printf 'results_size_bytes=%s\n' "$(file_size_bytes "${result_file}")"
    printf 'results_sha256=%s\n' "$(sha256_file "${result_file}")"
    printf 'samples_jsonl=%s\n' "${sample_marker}"
    printf 'samples_size_bytes=%s\n' "$(file_size_bytes "${sample_file}")"
    printf 'samples_sha256=%s\n' "$(sha256_file "${sample_file}")"
    printf 'completed_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } >"${temporary}"
  mv -f -- "${temporary}" "${marker}"
}

BENCHMARK=""
KEYFRAME_DIR_RAW=""
OUTPUT_ROOT_RAW=""
METHODS_RAW="dwt,swt"
ORIGINS_RAW="0,1,2,3,4"
INCLUDE_BASELINES=0
PREFIX_OVERRIDE=""
TASK_OVERRIDE=""
SPLIT_OVERRIDE=""
QWEN_CHECKPOINT="Qwen/Qwen2.5-VL-7B-Instruct"
CUDA_DEVICE="0"
MAX_NUM_FRAMES="16"
# 256 * 28 * 28 is Qwen's documented lower-cost visual-token setting.  It
# keeps 16-frame SDPA inference inside a 40 GiB A100 MIG slice; the checkpoint
# default can make Transformers 4.49 materialize an attention problem large
# enough to OOM before the first answer.
MAX_PIXELS="200704"
ATTENTION="sdpa"
BATCH_SIZE="1"
PYTHON_BIN="python"
CONVERTER_PYTHON="python"
REPO_ROOT_RAW="${DEFAULT_REPO_ROOT}"
LIMIT=""
PREDICTIONS_OUTPUT_RAW=""
RUNTIME_SIGNATURE=""
CONVERT_PREDICTIONS=1
FORCE=0
DRY_RUN=0

while (($#)); do
  case "$1" in
    --benchmark)
      (($# >= 2)) || die "--benchmark requires a value"
      BENCHMARK="$2"
      shift 2
      ;;
    --keyframe-dir)
      (($# >= 2)) || die "--keyframe-dir requires a value"
      KEYFRAME_DIR_RAW="$2"
      shift 2
      ;;
    --output-root)
      (($# >= 2)) || die "--output-root requires a value"
      OUTPUT_ROOT_RAW="$2"
      shift 2
      ;;
    --methods)
      (($# >= 2)) || die "--methods requires a value"
      METHODS_RAW="$2"
      shift 2
      ;;
    --include-baselines)
      INCLUDE_BASELINES=1
      shift
      ;;
    --origins)
      (($# >= 2)) || die "--origins requires a value"
      ORIGINS_RAW="$2"
      shift 2
      ;;
    --prefix)
      (($# >= 2)) || die "--prefix requires a value"
      PREFIX_OVERRIDE="$2"
      shift 2
      ;;
    --task)
      (($# >= 2)) || die "--task requires a value"
      TASK_OVERRIDE="$2"
      shift 2
      ;;
    --split)
      (($# >= 2)) || die "--split requires a value"
      SPLIT_OVERRIDE="$2"
      shift 2
      ;;
    --qwen-checkpoint)
      (($# >= 2)) || die "--qwen-checkpoint requires a value"
      QWEN_CHECKPOINT="$2"
      shift 2
      ;;
    --cuda-device)
      (($# >= 2)) || die "--cuda-device requires a value"
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --max-num-frames)
      (($# >= 2)) || die "--max-num-frames requires a value"
      MAX_NUM_FRAMES="$2"
      shift 2
      ;;
    --max-pixels)
      (($# >= 2)) || die "--max-pixels requires a value"
      MAX_PIXELS="$2"
      shift 2
      ;;
    --attention)
      (($# >= 2)) || die "--attention requires a value"
      ATTENTION="$2"
      shift 2
      ;;
    --batch-size)
      (($# >= 2)) || die "--batch-size requires a value"
      BATCH_SIZE="$2"
      shift 2
      ;;
    --python-bin)
      (($# >= 2)) || die "--python-bin requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --converter-python)
      (($# >= 2)) || die "--converter-python requires a value"
      CONVERTER_PYTHON="$2"
      shift 2
      ;;
    --repo-root)
      (($# >= 2)) || die "--repo-root requires a value"
      REPO_ROOT_RAW="$2"
      shift 2
      ;;
    --limit)
      (($# >= 2)) || die "--limit requires a value"
      LIMIT="$2"
      shift 2
      ;;
    --predictions-output)
      (($# >= 2)) || die "--predictions-output requires a value"
      PREDICTIONS_OUTPUT_RAW="$2"
      shift 2
      ;;
    --runtime-signature)
      (($# >= 2)) || die "--runtime-signature requires a value"
      RUNTIME_SIGNATURE="$2"
      shift 2
      ;;
    --no-convert)
      CONVERT_PREDICTIONS=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "${BENCHMARK}" ]] || die "--benchmark is required"
[[ -n "${KEYFRAME_DIR_RAW}" ]] || die "--keyframe-dir is required"
[[ -n "${OUTPUT_ROOT_RAW}" ]] || die "--output-root is required"

case "${BENCHMARK,,}" in
  videomme)
    CANONICAL_BENCHMARK="videomme"
    DEFAULT_PREFIX="videomme"
    DEFAULT_TASK="videomme"
    DEFAULT_SPLIT="test"
    ;;
  mlvu)
    CANONICAL_BENCHMARK="mlvu"
    DEFAULT_PREFIX="mlvu"
    DEFAULT_TASK="mlvu_dev"
    DEFAULT_SPLIT="test"
    ;;
  lvb|longvideobench)
    CANONICAL_BENCHMARK="lvb"
    DEFAULT_PREFIX="lvb"
    DEFAULT_TASK="longvideobench_val_v"
    DEFAULT_SPLIT="validation"
    ;;
  *)
    die "unsupported benchmark: ${BENCHMARK}"
    ;;
esac

PREFIX="${PREFIX_OVERRIDE:-${DEFAULT_PREFIX}}"
TASK="${TASK_OVERRIDE:-${DEFAULT_TASK}}"
SPLIT="${SPLIT_OVERRIDE:-${DEFAULT_SPLIT}}"
[[ "${PREFIX}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe filename prefix: ${PREFIX}"
[[ "${TASK}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe task token: ${TASK}"
[[ "${SPLIT}" =~ ^[A-Za-z0-9_.-]+$ ]] || die "unsafe split token: ${SPLIT}"
[[ "${ATTENTION}" =~ ^(sdpa|flash_attention_2|eager)$ ]] || \
  die "--attention must be sdpa, flash_attention_2, or eager"
[[ "${QWEN_CHECKPOINT}" != *$'\n'* && "${QWEN_CHECKPOINT}" != *,* ]] || \
  die "--qwen-checkpoint must not contain newlines or commas"
require_positive_integer "--max-num-frames" "${MAX_NUM_FRAMES}"
require_positive_integer "--max-pixels" "${MAX_PIXELS}"
require_positive_integer "--batch-size" "${BATCH_SIZE}"
if [[ -n "${LIMIT}" ]]; then
  [[ "${LIMIT}" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || die "--limit must be numeric"
fi

split_list "${METHODS_RAW}"
METHODS=("${SPLIT_VALUES[@]}")
if ((INCLUDE_BASELINES)); then
  METHODS+=(uniform topk)
fi
assert_unique_safe_methods "${METHODS[@]}"
split_list "${ORIGINS_RAW}"
RAW_ORIGINS=("${SPLIT_VALUES[@]}")
assert_unique_origins "${RAW_ORIGINS[@]}"
ORIGINS=()
for origin in "${RAW_ORIGINS[@]}"; do
  ORIGINS+=("$((10#${origin}))")
done
assert_unique_origins "${ORIGINS[@]}"

REPO_ROOT="$(make_absolute "${REPO_ROOT_RAW}" "${CALLER_DIR}")"
KEYFRAME_DIR="$(make_absolute "${KEYFRAME_DIR_RAW}" "${CALLER_DIR}")"
OUTPUT_ROOT="$(make_absolute "${OUTPUT_ROOT_RAW}" "${CALLER_DIR}")"
if [[ -n "${PREDICTIONS_OUTPUT_RAW}" ]]; then
  PREDICTIONS_OUTPUT="$(make_absolute "${PREDICTIONS_OUTPUT_RAW}" "${CALLER_DIR}")"
else
  PREDICTIONS_OUTPUT="${OUTPUT_ROOT}/${CANONICAL_BENCHMARK}/predictions.jsonl"
fi
[[ -d "${REPO_ROOT}" ]] || die "repository root does not exist: ${REPO_ROOT}"
[[ -d "${KEYFRAME_DIR}" ]] || die "keyframe directory does not exist: ${KEYFRAME_DIR}"
[[ "${KEYFRAME_DIR}" != *$'\n'* && "${KEYFRAME_DIR}" != *'"'* && "${KEYFRAME_DIR}" != *\\* ]] || \
  die "keyframe directory path must not contain newlines, quotes, or backslashes"

# Validate the entire input grid before starting an expensive GPU cell.
for method in "${METHODS[@]}"; do
  for origin in "${ORIGINS[@]}"; do
    printf -v origin_token '%02d' "$((10#${origin}))"
    keyframe_file="${KEYFRAME_DIR}/${PREFIX}_${method}_origin${origin_token}.json"
    [[ -s "${keyframe_file}" ]] || die "missing or empty keyframe JSON: ${keyframe_file}"
  done
done

MODEL_ARGS="max_num_frames=${MAX_NUM_FRAMES},use_keyframe=True,pretrained=${QWEN_CHECKPOINT},max_pixels=${MAX_PIXELS},attn_implementation=${ATTENTION},interleave_visuals=False"

printf 'Grid: benchmark=%s task=%s split=%s methods=%s origins=%s\n' \
  "${CANONICAL_BENCHMARK}" "${TASK}" "${SPLIT}" \
  "${METHODS[*]}" "${ORIGINS[*]}"

for method in "${METHODS[@]}"; do
  for origin in "${ORIGINS[@]}"; do
    printf -v origin_token '%02d' "$((10#${origin}))"
    data_name="${PREFIX}_${method}_origin${origin_token}"
    keyframe_file="${KEYFRAME_DIR}/${data_name}.json"
    data_files="{\"${SPLIT}\":\"${keyframe_file}\"}"
    cell_dir="${OUTPUT_ROOT}/${CANONICAL_BENCHMARK}/${method}/origin${origin_token}"
    completion_marker="${cell_dir}/.complete"
    console_log="${cell_dir}/console.log"

    command=(
      "${PYTHON_BIN}" -m lmms_eval
      --model qwen2_5_vl
      --tasks "${TASK}"
      --model_args "${MODEL_ARGS}"
      --batch_size "${BATCH_SIZE}"
      --device cuda:0
      --output_path "${cell_dir}"
      --log_samples
      --data_files "${data_files}"
      --seed 0
    )
    if [[ -n "${LIMIT}" ]]; then
      command+=(--limit "${LIMIT}")
    fi

    keyframe_sha256="$(sha256_file "${keyframe_file}")"
    keyframe_marker_path="$(path_for_marker "${keyframe_file}")"
    fingerprint_payload="$(printf '%s\n' \
      "schema=run_mllm_grid_v2" \
      "benchmark=${CANONICAL_BENCHMARK}" \
      "task=${TASK}" \
      "split=${SPLIT}" \
      "prefix=${PREFIX}" \
      "method=${method}" \
      "origin_id=${origin}" \
      "keyframe_json=${keyframe_marker_path}" \
      "keyframe_sha256=${keyframe_sha256}" \
      "model=qwen2_5_vl" \
      "model_args=${MODEL_ARGS}" \
      "batch_size=${BATCH_SIZE}" \
      "logical_device=cuda:0" \
      "cuda_visible_devices=${CUDA_DEVICE}" \
      "data_files=${data_files}" \
      "limit=${LIMIT}" \
      "seed=0" \
      "log_samples=true" \
      "python_bin=${PYTHON_BIN}")"
    if [[ -n "${RUNTIME_SIGNATURE}" ]]; then
      fingerprint_payload+=$'\n'"runtime_signature=${RUNTIME_SIGNATURE}"
    fi
    config_fingerprint="$(sha256_text "${fingerprint_payload}")"

    if ((DRY_RUN)); then
      print_command "${command[@]}"
      continue
    fi

    if [[ -f "${completion_marker}" && ${FORCE} -eq 0 ]]; then
      if valid_completion_marker \
        "${completion_marker}" "${CANONICAL_BENCHMARK}" "${TASK}" \
        "${method}" "${origin}" "${keyframe_marker_path}" \
        "${keyframe_sha256}" "${config_fingerprint}"; then
        printf 'SKIP: method=%s origin=%s already complete\n' "${method}" "${origin}"
        continue
      fi
      printf 'WARN: stale completion marker, rerunning %s/origin%s\n' \
        "${method}" "${origin_token}" >&2
      rm -f -- "${completion_marker}"
    fi

    mkdir -p -- "${cell_dir}"
    run_sentinel="${cell_dir}/.run_started.tmp"
    : >"${run_sentinel}"
    printf 'RUN: method=%s origin=%s keyframes=%s\n' \
      "${method}" "${origin}" "${keyframe_file}"

    set +e
    (
      cd -- "${REPO_ROOT}"
      # The patched CLI interpolates HF_TOKEN into a logged argument mapping.
      # Remove only that variable for this public checkpoint evaluation.
      env -u HF_TOKEN \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
        HF_HUB_DISABLE_TELEMETRY=1 \
        "${command[@]}"
    ) 2>&1 | tee "${console_log}"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    lmms_status="${pipeline_status[0]}"
    tee_status="${pipeline_status[1]}"

    if [[ "${lmms_status}" -ne 0 || "${tee_status}" -ne 0 ]]; then
      rm -f -- "${run_sentinel}"
      die "lmms-eval failed for method=${method} origin=${origin} (lmms=${lmms_status}, tee=${tee_status})"
    fi

    collect_artifacts "${cell_dir}" "${data_name}" "${TASK}" "${run_sentinel}"
    rm -f -- "${run_sentinel}"
    if ((${#RESULT_ARTIFACTS[@]} == 0 || ${#SAMPLE_ARTIFACTS[@]} == 0)); then
      die "lmms-eval returned success but expected new result/sample logs are missing for method=${method} origin=${origin}"
    fi

    write_completion_marker \
      "${completion_marker}" "${CANONICAL_BENCHMARK}" "${TASK}" \
      "${method}" "${origin}" "${keyframe_file}" \
      "${keyframe_sha256}" "${config_fingerprint}" \
      "${RESULT_ARTIFACTS[0]}" "${SAMPLE_ARTIFACTS[0]}"
    printf 'DONE: method=%s origin=%s\n' "${method}" "${origin}"
  done
done

methods_csv="$(join_by_comma "${METHODS[@]}")"
origins_csv="$(join_by_comma "${ORIGINS[@]}")"
converter_command=(
  "${CONVERTER_PYTHON}" "${SCRIPT_DIR}/convert_lmms_logs.py"
  --grid-root "${OUTPUT_ROOT}/${CANONICAL_BENCHMARK}"
  --benchmark "${CANONICAL_BENCHMARK}"
  --methods "${methods_csv}"
  --origins "${origins_csv}"
  --output "${PREDICTIONS_OUTPUT}"
)

if ((DRY_RUN)); then
  if ((CONVERT_PREDICTIONS)); then
    print_command "${converter_command[@]}"
  fi
  printf 'Dry run complete; no lmms-eval cells or markers were written.\n'
else
  if ((CONVERT_PREDICTIONS)); then
    printf 'MERGE: verified sample logs -> %s\n' "${PREDICTIONS_OUTPUT}"
    (
      cd -- "${REPO_ROOT}"
      env -u HF_TOKEN "${converter_command[@]}"
    ) || die "prediction log conversion failed"
  fi
  printf 'MLLM grid complete: %s\n' "${OUTPUT_ROOT}/${CANONICAL_BENCHMARK}"
fi
