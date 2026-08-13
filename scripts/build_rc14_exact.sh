#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

BASE_RUN_DIR=""
RC12_RUN_DIR=""
RUN_DIR=""
EVALUATION_ROOT=""
PYTHON_BIN="${HOME}/.venvs/wfs-sb-a100/bin/python"
while (($#)); do
  case "$1" in
    --base-run-dir) BASE_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --rc12-run-dir) RC12_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --run-dir) RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --evaluation-root) EVALUATION_ROOT="$(realpath -m -- "$2")"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${BASE_RUN_DIR}" && -n "${RC12_RUN_DIR}" && -n "${RUN_DIR}" && -n "${EVALUATION_ROOT}" ]] || {
  echo "base/RC12/new run/evaluation directories are required" >&2
  exit 2
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SIGNALS="${BASE_RUN_DIR}/preprocess/dense_signals.jsonl"
CATALOG="${BASE_RUN_DIR}/preprocess/catalog.jsonl"
MANIFESTS="${BASE_RUN_DIR}/preprocess/multiphase_manifests.jsonl"
SOURCE_BUNDLE="${BASE_RUN_DIR}/preprocess/source_video_bundle.jsonl"
QUESTIONS="${EVALUATION_ROOT}/datasets/videomme/videomme_json_file.json"
DATASET_ROOT="${EVALUATION_ROOT}/datasets/videomme"
RC12_DECISIONS="${RC12_RUN_DIR}/decisions/decisions.jsonl"
RC12_TRACES="${RC12_RUN_DIR}/exact_decode/traces.jsonl"
RC12_KEYFRAMES="${RC12_RUN_DIR}/keyframes"
RC12_PREDICTIONS="${RC12_RUN_DIR}/predictions.jsonl"
RC12_VALIDATION="${RC12_RUN_DIR}/validation.json"
DECISIONS="${RUN_DIR}/decisions/decisions.jsonl"
TRACES="${RUN_DIR}/exact_decode/traces.jsonl"
PAIRED_TRACES="${RUN_DIR}/exact_decode/paired_traces.jsonl"
DECODE_SUMMARY="${RUN_DIR}/exact_decode/decode_summary.json"
KEYFRAMES="${RUN_DIR}/keyframes"
VALIDATION="${RUN_DIR}/validation.json"
mkdir -p -- "${RUN_DIR}"
exec 9>"${RUN_DIR}/.rc14_build.lock"
flock -n 9 || { echo "another RC14 builder owns ${RUN_DIR}" >&2; exit 1; }
for input in "${SIGNALS}" "${CATALOG}" "${MANIFESTS}" "${SOURCE_BUNDLE}" "${QUESTIONS}" "${RC12_DECISIONS}" "${RC12_TRACES}" "${RC12_PREDICTIONS}" "${RC12_VALIDATION}"; do
  [[ -s "${input}" ]] || { echo "missing RC14 input: ${input}" >&2; exit 1; }
done
[[ "$(sha256sum -- "${SIGNALS}" | cut -d' ' -f1)" == "cca842810d063f714ce3c6655baf672e22255b6c8714f2b363a56a7a4b6dce66" ]] || exit 1
[[ "$(sha256sum -- "${MANIFESTS}" | cut -d' ' -f1)" == "d493f6763ef0196196a8014d1006a219426bc2e013c1c5b687410a0d4e2378c3" ]] || exit 1

cd -- "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" scripts/build_rc12_decisions.py \
  --signals "${SIGNALS}" --catalog "${CATALOG}" --manifests "${MANIFESTS}" \
  --output-dir "${RUN_DIR}/decisions" --treatment-method phasefuse_rc14 \
  --anchor-count 14
PYTHONPATH=. "${PYTHON_BIN}" scripts/decode_rc12_exact.py \
  --decisions "${DECISIONS}" --output-dir "${RUN_DIR}/exact_decode" \
  --treatment-method phasefuse_rc14 --treatment-only \
  --reference-uniform-traces "${RC12_TRACES}"
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable analyze-phasefuse \
  --traces "${PAIRED_TRACES}" \
  --output "${RUN_DIR}/selector_analysis_summary.json" \
  --baseline-method canonical_uniform --treatment-method phasefuse_rc14 \
  --n-bootstrap 10000 --seed 20260813
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable export-keyframes \
  --traces "${TRACES}" --benchmark videomme --questions-file "${QUESTIONS}" \
  --dataset-root "${DATASET_ROOT}" --output-dir "${KEYFRAMES}" \
  --methods phasefuse_rc14 --origin-ids 0 1 2 3 4 \
  --expected-budget 16 --allow-annotation-subset
PYTHONPATH=. "${PYTHON_BIN}" scripts/validate_rc14_exact_artifacts.py \
  --decisions "${DECISIONS}" --traces "${TRACES}" \
  --paired-traces "${PAIRED_TRACES}" \
  --decode-summary "${DECODE_SUMMARY}" --keyframes "${KEYFRAMES}" \
  --source-video-bundle "${SOURCE_BUNDLE}" \
  --reference-decisions "${RC12_DECISIONS}" \
  --reference-keyframes "${RC12_KEYFRAMES}" \
  --reference-predictions "${RC12_PREDICTIONS}" --output "${VALIDATION}"
echo "RC14 exact selector/decode validation complete: ${RUN_DIR}"
