#!/usr/bin/env bash
# Frozen, foreground QV Stage-0 selector runner. Qwen is intentionally absent.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

EXPECTED_APPROVAL="NESTED_R2_QV_STAGE0"
EXPECTED_PREREG_SHA="694522db81e5f9db5d244f713c03fd26844c5f680828f8fc3df0b4ed992c9cff"
EXPECTED_COHORT_SHA="349f450a169fc1160429738f30028f38b6401b5f33c3760d02d70b3111ccb3b4"
EXPECTED_CONFIG_SHA="d3512cf41506b214c86cce72f607390184cecfa7b9bd4750adcc4122788e61e9"
CHECKPOINT_REVISION="520bf73fd0ef6ccce791cd3f06908040aef2d8cc"
CHECKPOINT_TREE_SHA="93f537621bfb693fac5a52760a5b83bd93eb3dd1885292e680b62bc81d962b1f"
MASTER_SEED=20260810
NUM_ORIGINS=5
SAMPLE_FPS=1.0
DATASET_ROOT="/home/elicer/videounderstanding/outputs/local-data/qvhighlights"
ANNOTATION="${DATASET_ROOT}/annotations/highlight_val_release.jsonl"
PYTHON_BIN="/home/elicer/.venvs/wfs-sb-a100/bin/python"
MODEL_PATH="/home/elicer/.cache/huggingface/hub/models--Salesforce--blip2-itm-vit-g/snapshots/${CHECKPOINT_REVISION}"
RUN_DIR="/home/elicer/WFS-SB/artifacts/phasefuse_nested_r2_qv_holdout_stage0"
GPU_LOCK="/home/elicer/WFS-SB/.phasefuse_gpu.lock"
APPROVED=""

usage() {
  cat <<EOF
Usage: bash scripts/run_nested_r2_qv_stage0.sh --approved ${EXPECTED_APPROVAL} [--run-dir PATH]

Runs only the frozen 100-source QV Stage-0 selector gate:
query-only 5-origin 1Hz BLIP2 scout -> paired A16/nested-R2 decisions ->
fresh exact K16 decode -> pre-label artifact seal -> label join -> hierarchical
source-cluster bootstrap. This runner contains no Qwen command.
EOF
}

die() { printf 'nested-r2-qv-stage0: ERROR: %s\n' "$*" >&2; exit 1; }
while (($#)); do
  case "$1" in
    --approved) [[ $# -ge 2 ]] || die "--approved requires a value"; APPROVED="$2"; shift 2 ;;
    --run-dir) [[ $# -ge 2 ]] || die "--run-dir requires a value"; RUN_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ "${APPROVED}" == "${EXPECTED_APPROVAL}" ]] || die "explicit Stage-0 approval token is required"
[[ -x "${PYTHON_BIN}" ]] || die "missing frozen Python: ${PYTHON_BIN}"
[[ -d "${MODEL_PATH}" ]] || die "missing frozen BLIP2 snapshot: ${MODEL_PATH}"
[[ -f "${ANNOTATION}" ]] || die "missing frozen QV annotation: ${ANNOTATION}"
[[ "$(sha256sum protocols/nested_r2_qv_holdout_preregistration.json | awk '{print $1}')" == "${EXPECTED_PREREG_SHA}" ]] || die "preregistration hash drift"
[[ "$(sha256sum protocols/nested_r2_qv_holdout_cohort.tsv | awk '{print $1}')" == "${EXPECTED_COHORT_SHA}" ]] || die "cohort hash drift"
[[ "$(sha256sum configs/nested_r2_frozen.yaml | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA}" ]] || die "config hash drift"
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || die "runner requires a clean committed worktree"
[[ "$(git merge-base --is-ancestor 258dad062e623f7571559add70658cc7f3ba2034 HEAD; echo $?)" == "0" ]] || die "frozen nested-R2 source is not an ancestor"

RUN_DIR="$("${PYTHON_BIN}" -c 'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve())' "${RUN_DIR}")"
case "${RUN_DIR}/" in
  "${REPO_ROOT}/"*) die "--run-dir must be outside the execution worktree" ;;
esac
if [[ -e "${RUN_DIR}" ]]; then
  die "--run-dir must not preexist; choose a new isolated output path"
fi
mkdir -p "${RUN_DIR}" "${RUN_DIR}/logs" "${RUN_DIR}/.state" "${RUN_DIR}/blind" "${RUN_DIR}/analysis"
exec 9>"${GPU_LOCK}"
flock -n 9 || die "another PhaseFuse run holds ${GPU_LOCK}"
exec 8>"${RUN_DIR}/.state/run.lock"
flock -n 8 || die "another process holds the QV Stage-0 run lock"

QUERY_MANIFEST="${RUN_DIR}/blind/query_only_manifest.jsonl"
SEALED_LABELS="${RUN_DIR}/blind/evaluation_labels.preseal.jsonl"
MATERIALIZATION="${RUN_DIR}/blind/materialization.json"
SAMPLING_MANIFESTS="${RUN_DIR}/sampling_manifests.jsonl"
CATALOG="${RUN_DIR}/catalog.jsonl"
SOURCE_INVENTORY="${RUN_DIR}/source_video_bundle.json"
CHECKPOINT_PREFLIGHT="${RUN_DIR}/checkpoint_preflight.json"
RUNTIME_MANIFEST="${RUN_DIR}/runtime_manifest.json"
SIGNALS="${RUN_DIR}/scout_signals.jsonl"
PREPROCESS_DIR="${RUN_DIR}/preprocess"
DECISIONS="${RUN_DIR}/decisions.jsonl"
EXACT_DIR="${RUN_DIR}/exact_decode"
TRACES="${EXACT_DIR}/traces.jsonl"
BLIND_SEAL="${RUN_DIR}/blind/artifact_seal.json"
JOINED_LABELS="${RUN_DIR}/analysis/joined_labels.jsonl"
JOIN_MANIFEST="${RUN_DIR}/analysis/label_join_manifest.json"
METRIC_ROWS="${RUN_DIR}/analysis/paired_metric_rows.jsonl"
SUMMARY="${RUN_DIR}/analysis/stage0_summary.json"

run_logged() {
  local name="$1"; shift
  printf '[run] %s\n' "${name}"
  "$@" 2>&1 | tee "${RUN_DIR}/logs/${name}.log"
}

run_logged materialize "${PYTHON_BIN}" -m scripts.materialize_nested_r2_qv_blind materialize \
  --annotation "${ANNOTATION}" \
  --cohort protocols/nested_r2_qv_holdout_cohort.tsv \
  --prereg protocols/nested_r2_qv_holdout_preregistration.json \
  --query-manifest "${QUERY_MANIFEST}" \
  --sealed-labels "${SEALED_LABELS}" \
  --output-manifest "${MATERIALIZATION}"

run_logged inputs "${PYTHON_BIN}" -m scripts.build_nested_r2_qv_inputs \
  --query-manifest "${QUERY_MANIFEST}" \
  --dataset-root "${DATASET_ROOT}" \
  --manifests "${SAMPLING_MANIFESTS}" \
  --catalog "${CATALOG}" \
  --source-inventory "${SOURCE_INVENTORY}" \
  --master-seed "${MASTER_SEED}" \
  --num-origins "${NUM_ORIGINS}" \
  --sample-fps "${SAMPLE_FPS}"

run_logged checkpoint "${PYTHON_BIN}" - "${MODEL_PATH}" "${CHECKPOINT_PREFLIGHT}" "${CHECKPOINT_REVISION}" "${CHECKPOINT_TREE_SHA}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
revision = sys.argv[3]
expected = sys.argv[4]
rows = []
for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    rows.append({"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()})
encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
tree = hashlib.sha256(encoded).hexdigest()
if root.name != revision or tree != expected:
    raise SystemExit(f"checkpoint content identity drift: revision={root.name}, tree={tree}")
payload = {"schema_version": 1, "algorithm": "sha256(canonical-json(sorted[{path,size_bytes,sha256}]))", "snapshot_path": str(root.resolve()), "revision": revision, "content_tree_sha256": tree, "num_files": len(rows), "total_size_bytes": sum(row["size_bytes"] for row in rows), "files": rows}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
os.replace(temporary, output)
print(json.dumps({key: payload[key] for key in ("revision", "content_tree_sha256", "num_files", "total_size_bytes")}, sort_keys=True))
PY

"${PYTHON_BIN}" - "${RUNTIME_MANIFEST}" "${CHECKPOINT_PREFLIGHT}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

output, checkpoint = map(Path, sys.argv[1:])
repo = Path.cwd()
files = sorted([*repo.joinpath("phase_stable").glob("*.py"), *repo.joinpath("scripts").glob("*nested_r2*qv*.py"), repo / "scripts/decode_rc12_exact.py", repo / "scripts/run_nested_r2_qv_stage0.sh"])
code = []
for path in files:
    code.append((str(path.relative_to(repo)), hashlib.sha256(path.read_bytes()).hexdigest()))
packages = {}
for name in ("numpy", "scipy", "av", "torch", "transformers", "Pillow"):
    try: packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name] = None
payload = {
    "schema_version": 1,
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_status_clean": not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], text=True).strip(),
    "code_tree_sha256": hashlib.sha256(json.dumps(code, separators=(",", ":")).encode()).hexdigest(),
    "code_files": code,
    "python": sys.version,
    "python_executable_raw": sys.executable,
    "python_executable_resolved": str(Path(sys.executable).resolve()),
    "platform": platform.platform(),
    "packages": packages,
    "checkpoint_preflight_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "qwen_commands_present": False,
}
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
os.replace(temporary, output)
PY

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
if ! nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >"${RUN_DIR}/gpu_processes_at_launch.txt" 2>&1; then
  printf '[preflight] nvidia-smi compute-process diagnostic unavailable; enforcing CUDA free-memory threshold\n'
fi
if "${PYTHON_BIN}" -m scripts.preflight_nested_r2_qv_stage0 \
  --gpu-process-file "${RUN_DIR}/gpu_processes_at_launch.txt"; then
  die "GPU compute process already exists despite obtaining the shared PhaseFuse lock"
fi
"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
if "A100" not in name.upper():
    raise SystemExit(f"frozen runner requires A100, found {name}")
free, total = torch.cuda.mem_get_info(0)
if free < 30 * 1024**3:
    raise SystemExit(f"insufficient free CUDA memory: {free / 1024**3:.1f} GiB")
print(name, f"free={free / 1024**3:.1f}GiB", f"total={total / 1024**3:.1f}GiB")
PY

run_logged preprocess "${PYTHON_BIN}" -m scripts.preprocess_nested_r2_qv_blind \
  --query-manifest "${QUERY_MANIFEST}" \
  --dataset-root "${DATASET_ROOT}" \
  --manifests "${SAMPLING_MANIFESTS}" \
  --output-dir "${PREPROCESS_DIR}" \
  --signal-jsonl "${SIGNALS}" \
  --model-path "${MODEL_PATH}" \
  --device cuda:0 \
  --batch-size 8 \
  --frame-buffer-size 256 \
  --checkpoint-tree-sha256 "${CHECKPOINT_TREE_SHA}"

run_logged decisions "${PYTHON_BIN}" -m scripts.build_nested_r2_qv_decisions \
  --signals "${SIGNALS}" --catalog "${CATALOG}" --output "${DECISIONS}"

run_logged exact_decode "${PYTHON_BIN}" -m scripts.decode_nested_r2_qv_exact \
  --decisions "${DECISIONS}" --output-dir "${EXACT_DIR}"

run_logged blind_seal "${PYTHON_BIN}" -m scripts.validate_nested_r2_qv_stage0 \
  --cohort protocols/nested_r2_qv_holdout_cohort.tsv \
  --prereg protocols/nested_r2_qv_holdout_preregistration.json \
  --config configs/nested_r2_frozen.yaml \
  --query-manifest "${QUERY_MANIFEST}" \
  --sampling-manifests "${SAMPLING_MANIFESTS}" \
  --signals "${SIGNALS}" \
  --decisions "${DECISIONS}" \
  --traces "${TRACES}" \
  --source-video-bundle "${SOURCE_INVENTORY}" \
  --blind-materialization "${MATERIALIZATION}" \
  --sealed-labels "${SEALED_LABELS}" \
  --checkpoint-preflight "${CHECKPOINT_PREFLIGHT}" \
  --runtime-manifest "${RUNTIME_MANIFEST}" \
  --expected-git-head "$(git rev-parse HEAD)" \
  --output "${BLIND_SEAL}"

run_logged label_join "${PYTHON_BIN}" -m scripts.materialize_nested_r2_qv_blind join \
  --sealed-labels "${SEALED_LABELS}" \
  --blind-seal "${BLIND_SEAL}" \
  --decisions "${DECISIONS}" \
  --traces "${TRACES}" \
  --source-video-bundle "${SOURCE_INVENTORY}" \
  --materialization-manifest "${MATERIALIZATION}" \
  --query-manifest "${QUERY_MANIFEST}" \
  --cohort protocols/nested_r2_qv_holdout_cohort.tsv \
  --output-labels "${JOINED_LABELS}" \
  --join-manifest "${JOIN_MANIFEST}"

run_logged analysis "${PYTHON_BIN}" -m scripts.analyze_nested_r2_qv_stage0 \
  --labels "${JOINED_LABELS}" \
  --label-join-manifest "${JOIN_MANIFEST}" \
  --blind-seal "${BLIND_SEAL}" \
  --decisions "${DECISIONS}" \
  --traces "${TRACES}" \
  --output-rows "${METRIC_ROWS}" \
  --output-summary "${SUMMARY}"

printf 'QV nested-R2 Stage-0 complete: %s\n' "${SUMMARY}"
