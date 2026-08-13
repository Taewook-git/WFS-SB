#!/usr/bin/env bash
# Resume-safe selector-only phase marginalization ablation on authenticated dev20.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_phasefuse_v2_selector_ablation.sh [options]

Inputs:
  --base-run-dir DIR   Completed PhaseFuse dev20 root
                       (default: artifacts/phasefuse_videomme_dev20)
  --config FILE        Frozen selector-ablation YAML
                       (default: configs/phasefuse_v2_selector_ablation_dev20.yaml)

Runtime:
  --python-bin CMD     Python executable (default: active A100 venv)
  --n-bootstrap N      Video-cluster resamples (default: 10000)
  --force              Recompute selector arms despite a valid marker
  -h, --help           Show this help

This is selector-only: it never invokes lmms-eval or Qwen.  It reads the
authenticated dev20 dense signals and feature arrays, then compares phase-0
1 Hz SWT and direct-dense 4 Hz SWT against four-phase median fusion under the
exact same global-coverage selector.  Outputs are isolated below
BASE_RUN_DIR/v2_selector_ablation.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

absolute_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then printf '%s\n' "${value}"; else printf '%s/%s\n' "${REPO_ROOT}" "${value}"; fi
}

require_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer: $2"
}

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum -- "$1" | awk '{print $1}'; else shasum -a 256 -- "$1" | awk '{print $1}'; fi
}

sha256_text() {
  if command -v sha256sum >/dev/null 2>&1; then printf '%s' "$1" | sha256sum | awk '{print $1}'; else printf '%s' "$1" | shasum -a 256 -- "$1" | awk '{print $1}'; fi
}

source_tree_hash() {
  local root="$1"; shift
  (cd -- "${root}" && find "$@" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
}

valid_marker() {
  local marker="$1" fingerprint="$2"; shift 2
  [[ -s "${marker}" && "$(sed -n 's/^fingerprint=//p' "${marker}")" == "${fingerprint}" ]] || return 1
  [[ "$(grep -c '^output=' "${marker}" || true)" == "$#" ]] || return 1
  local output
  for output in "$@"; do
    [[ -s "${output}" ]] || return 1
    grep -Fqx -- "output=${output}|$(sha256_file "${output}")" "${marker}" || return 1
  done
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

BASE_RUN_DIR_RAW="artifacts/phasefuse_videomme_dev20"
CONFIG_RAW="configs/phasefuse_v2_selector_ablation_dev20.yaml"
PYTHON_BIN="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.venvs/wfs-sb-a100/bin/python}"
N_BOOTSTRAP="10000"
FORCE=0

while (($#)); do
  case "$1" in
    --base-run-dir) BASE_RUN_DIR_RAW="$2"; shift 2 ;;
    --config) CONFIG_RAW="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --n-bootstrap) N_BOOTSTRAP="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

BASE_RUN_DIR="$(absolute_path "${BASE_RUN_DIR_RAW}")"
RUN_DIR="${BASE_RUN_DIR}/v2_selector_ablation"
SELECTION_DIR="${RUN_DIR}/selection"
CONFIG_FILE="$(absolute_path "${CONFIG_RAW}")"
BASE_MARKER="${BASE_RUN_DIR}/preprocess/.complete"
BASE_SIGNALS="${BASE_RUN_DIR}/preprocess/dense_signals.jsonl"
BASE_FEATURE_BUNDLE="${BASE_RUN_DIR}/preprocess/feature_bundle.jsonl"
BASE_SUMMARY="${BASE_RUN_DIR}/preprocess/preprocess_summary.json"

[[ -x "${PYTHON_BIN}" ]] || die "Python executable is missing: ${PYTHON_BIN}"
[[ -s "${CONFIG_FILE}" ]] || die "ablation config is missing: ${CONFIG_FILE}"
[[ -s "${BASE_MARKER}" && -s "${BASE_SIGNALS}" && -s "${BASE_FEATURE_BUNDLE}" && -s "${BASE_SUMMARY}" ]] || die "authenticated dev20 preprocessing bundle is incomplete"
require_positive_integer --n-bootstrap "${N_BOOTSTRAP}"

mkdir -p -- "${RUN_DIR}"
exec 8>"${BASE_RUN_DIR}/.phasefuse.lock"
flock -n 8 || die "the base PhaseFuse run is active: ${BASE_RUN_DIR}"
exec 9>"${RUN_DIR}/.selector_ablation.lock"
flock -n 9 || die "another selector ablation owns ${RUN_DIR}"

mapfile -t CONTRACT < <(
  cd -- "${REPO_ROOT}"
  "${PYTHON_BIN}" - "${BASE_MARKER}" "${BASE_SIGNALS}" "${BASE_FEATURE_BUNDLE}" "${BASE_SUMMARY}" "${CONFIG_FILE}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
from collections import defaultdict

marker_path, signals_path, bundle_path, summary_path, config_path = map(pathlib.Path, sys.argv[1:])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

marker_lines = marker_path.read_text(encoding="utf-8").splitlines()
fingerprints = [line.split("=", 1)[1] for line in marker_lines if line.startswith("fingerprint=")]
if len(fingerprints) != 1 or not re.fullmatch(r"[0-9a-f]{64}", fingerprints[0]):
    raise SystemExit("base preprocess marker fingerprint is invalid")
authenticated = {}
for line in marker_lines:
    if not line.startswith("output="):
        continue
    raw_path, separator, expected = line[len("output="):].rpartition("|")
    path = pathlib.Path(raw_path).resolve()
    if not separator or not path.is_file() or digest(path) != expected:
        raise SystemExit(f"base marker output failed authentication: {path}")
    authenticated[path] = expected
for required in (signals_path.resolve(), bundle_path.resolve(), summary_path.resolve()):
    if required not in authenticated:
        raise SystemExit(f"base marker does not authenticate {required}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
expected_summary = {
    "command": "preprocess-phasefuse", "num_videos": 20,
    "num_signal_records": 300, "num_outer_origins": 5, "num_inner_phases": 4,
}
if any(summary.get(key) != value for key, value in expected_summary.items()):
    raise SystemExit("base preprocess summary is not the frozen dev20 grid")

bundle_rows = [json.loads(line) for line in bundle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
bundle_paths = set()
for row in bundle_rows:
    if set(row) != {"path", "sha256", "size_bytes"}:
        raise SystemExit("invalid feature-bundle row")
    path = pathlib.Path(row["path"]).resolve()
    if path in bundle_paths or not path.is_file() or path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
        raise SystemExit(f"feature artifact failed authentication: {path}")
    bundle_paths.add(path)

rows = [json.loads(line) for line in signals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 300:
    raise SystemExit("base signals must contain exactly 300 rows")
grid = defaultdict(set)
signal_features = set()
for row in rows:
    item = (row.get("dataset"), str(row.get("video_id")), str(row.get("question_id")))
    origin = row.get("origin_id")
    if item[0] != "videomme" or isinstance(origin, bool) or origin not in range(5):
        raise SystemExit("base signal grid is invalid")
    if origin in grid[item]:
        raise SystemExit("duplicate base signal grid cell")
    grid[item].add(origin)
    signal_features.add(pathlib.Path(row["visual_features_path"]).resolve())
if len(grid) != 60 or len({item[1] for item in grid}) != 20 or any(origins != set(range(5)) for origins in grid.values()):
    raise SystemExit("base signals are not the exact 20-video/60-item/5-origin grid")
if signal_features != bundle_paths:
    raise SystemExit("signal feature paths do not match the authenticated bundle")

from phase_stable.phasefuse_experiment import load_phasefuse_config
config = load_phasefuse_config(config_path)
if config.experiment.methods != (
    "phase0_swt_v2_selector", "dense_swt_v2_selector", "phasefuse_v2"
):
    raise SystemExit("ablation config must contain exactly the three frozen arms")
if config.experiment.frame_budget != 16 or config.experiment.num_phases != 4:
    raise SystemExit("ablation config must use K=16 and four physical phases")
if config.experiment.selection_strategy != "global_coverage":
    raise SystemExit("ablation config must use global_coverage")
if config.metadata.get("status") != "frozen_selector_only_ablation":
    raise SystemExit("ablation config metadata is not frozen")

print(fingerprints[0])
print(digest(marker_path))
print(digest(signals_path))
print(digest(bundle_path))
print(digest(config_path))
PY
)
((${#CONTRACT[@]} == 5)) || die "failed to validate ablation provenance contract"

SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}" phase_stable scripts configs)"
FINGERPRINT="$(sha256_text "phasefuse_v2_selector_ablation_v2|base_fp=${CONTRACT[0]}|base_marker=${CONTRACT[1]}|signals=${CONTRACT[2]}|features=${CONTRACT[3]}|config=${CONTRACT[4]}|source=${SOURCE_SHA}|methods=phase0_swt_v2_selector,dense_swt_v2_selector,phasefuse_v2|bootstrap=${N_BOOTSTRAP}")"
MARKER="${RUN_DIR}/.complete"
TRACES="${SELECTION_DIR}/traces.jsonl"
SUMMARY="${SELECTION_DIR}/summary.json"
PHASE0_ANALYSIS="${SELECTION_DIR}/analysis_summary.json"
DENSE_ANALYSIS="${SELECTION_DIR}/dense_vs_phasefuse_v2_analysis.json"
TRACE_BUNDLE="${SELECTION_DIR}/trace_array_bundle.jsonl"
RUN_MANIFEST="${SELECTION_DIR}/manifest/run_manifest.json"
ENVIRONMENT="${SELECTION_DIR}/manifest/environment.json"

if ((FORCE == 0)) && valid_marker "${MARKER}" "${FINGERPRINT}" "${TRACES}" "${SUMMARY}" "${PHASE0_ANALYSIS}" "${DENSE_ANALYSIS}" "${TRACE_BUNDLE}" "${RUN_MANIFEST}" "${ENVIRONMENT}"; then
  printf 'SKIP: matched-selector dev20 ablation already complete\n'
  exit 0
fi

printf 'RUN: matched phase0/dense SWT selector controls vs phasefuse_v2\n'
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable run-phasefuse \
  "${BASE_SIGNALS}" "${SELECTION_DIR}" --config "${CONFIG_FILE}" \
  --methods phase0_swt_v2_selector dense_swt_v2_selector phasefuse_v2 \
  --baseline-method phase0_swt_v2_selector \
  --treatment-method phasefuse_v2 --n-bootstrap "${N_BOOTSTRAP}")

printf 'RUN: dense SWT v2-selector paired video-cluster bootstrap\n'
(cd -- "${REPO_ROOT}" && "${PYTHON_BIN}" -m phase_stable analyze-phasefuse \
  --traces "${TRACES}" --output "${DENSE_ANALYSIS}" \
  --baseline-method dense_swt_v2_selector --treatment-method phasefuse_v2 \
  --n-bootstrap "${N_BOOTSTRAP}")

"${PYTHON_BIN}" - "${TRACES}" "${TRACE_BUNDLE}" "${PHASE0_ANALYSIS}" "${DENSE_ANALYSIS}" <<'PY'
import hashlib
import json
import pathlib
import sys
from collections import defaultdict

traces_path, bundle_path, phase0_analysis_path, dense_analysis_path = map(pathlib.Path, sys.argv[1:])
rows = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
methods = {"phase0_swt_v2_selector", "dense_swt_v2_selector", "phasefuse_v2"}
if len(rows) != 900 or {row.get("method") for row in rows} != methods:
    raise SystemExit("trace output must be the exact 900-row three-arm grid")
grid = defaultdict(set)
for row in rows:
    key = (row.get("dataset"), str(row.get("video_id")), str(row.get("question_id")), row.get("method"))
    origin = row.get("origin_id")
    selected = row.get("selected_indices")
    sources = row.get("selected_source_frame_indices")
    if origin in grid[key] or origin not in range(5):
        raise SystemExit("trace grid contains a duplicate/invalid origin")
    grid[key].add(origin)
    if not isinstance(selected, list) or len(selected) != 16 or len(set(selected)) != 16 or selected != sorted(selected):
        raise SystemExit("trace violates exact sorted K=16")
    if not isinstance(sources, list) or len(sources) != 16 or len(set(sources)) != 16 or sources != sorted(sources):
        raise SystemExit("trace violates exact source-frame K=16")
    metadata = row.get("method_metadata", {})
    config = metadata.get("phasefuse", {}).get("config", {})
    expected = {
        "selection_strategy": "global_coverage", "uniform_reserve": 8,
        "selection_event_weight": 0.25, "component_scaling": "percentile",
        "min_selection_distance_sec": 0.5, "frame_budget": 16,
        "uncertainty_penalty": 0.0, "relevance_weight": 0.0,
        "phase_vote_weight": 0.0,
    }
    if any(config.get(name) != value for name, value in expected.items()):
        raise SystemExit("selector settings are not matched to frozen v2")
    if row["method"] == "phase0_swt_v2_selector" and (
        metadata.get("ablation") != "phase_marginalization_only_control"
        or metadata.get("phase_marginalization") != "disabled_phase0_only"
        or metadata.get("physical_phase_ids_used") != [0]
        or config.get("num_phases") != 1
    ):
        raise SystemExit("phase-0 control metadata is invalid")
    if row["method"] == "dense_swt_v2_selector" and (
        metadata.get("ablation") != "phase_marginalization_only_control"
        or metadata.get("phase_marginalization") != "disabled_dense_single_stream"
        or metadata.get("physical_phase_ids_used") != [0, 1, 2, 3]
        or config.get("num_phases") != 1
    ):
        raise SystemExit("direct-dense control metadata is invalid")
    if row["method"] == "phasefuse_v2" and config.get("num_phases") != 4:
        raise SystemExit("phasefuse_v2 must retain four physical phases")
if len(grid) != 180 or any(origins != set(range(5)) for origins in grid.values()):
    raise SystemExit("trace output is not a complete paired grid")

bundle_rows = [json.loads(line) for line in bundle_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(bundle_rows) != 900:
    raise SystemExit("trace array bundle must authenticate all 900 arrays")
for row in bundle_rows:
    path = pathlib.Path(row["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != row["size_bytes"] or digest != row["sha256"]:
        raise SystemExit(f"trace array failed authentication: {path}")

for path, baseline in (
    (phase0_analysis_path, "phase0_swt_v2_selector"),
    (dense_analysis_path, "dense_swt_v2_selector"),
):
    analysis = json.loads(path.read_text(encoding="utf-8"))
    if analysis.get("baseline_method") != baseline or analysis.get("treatment_method") != "phasefuse_v2":
        raise SystemExit("analysis arms are invalid")
    if analysis.get("num_paired_items") != 60 or analysis.get("origin_ids") != [0, 1, 2, 3, 4]:
        raise SystemExit("analysis is not the complete paired dev20 grid")
    comparison = analysis.get("comparison", {})
    if comparison.get("effect_definition") != "treatment - baseline" or comparison.get("n_clusters") != 20:
        raise SystemExit("analysis does not use the required joint video-cluster bootstrap")
PY

write_marker "${MARKER}" "${FINGERPRINT}" "${TRACES}" "${SUMMARY}" "${PHASE0_ANALYSIS}" "${DENSE_ANALYSIS}" "${TRACE_BUNDLE}" "${RUN_MANIFEST}" "${ENVIRONMENT}"
printf 'PhaseFuse-v2 selector ablation complete: %s\n' "${RUN_DIR}"
