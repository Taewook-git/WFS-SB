#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

APPROVED=""
RUN_DIR=""
BASE_RUN_DIR=""
PYTHON_BIN="${HOME}/.venvs/wfs-sb-a100/bin/python"
while (($#)); do
  case "$1" in
    --approved) APPROVED="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --base-run-dir) BASE_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "${APPROVED}" == "RC12_DEV20_QWEN" ]] || { echo "explicit root approval token required" >&2; exit 1; }
[[ -n "${RUN_DIR}" && -n "${BASE_RUN_DIR}" ]] || { echo "run/base directories required" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
mkdir -p -- "${RUN_DIR}"
exec 9>"${RUN_DIR}/.rc12_qwen.lock"
flock -n 9 || { echo "another RC12 Qwen launcher owns ${RUN_DIR}" >&2; exit 1; }

sha256_file() { sha256sum -- "$1" | awk '{print $1}'; }
source_tree_hash() {
  local root="$1"; shift
  (cd -- "${root}" && find "$@" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' \) -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
}
directory_content_hash() {
  (cd -- "$1" && find -L . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')
}

VALIDATION="${RUN_DIR}/validation.json"
KEYFRAMES="${RUN_DIR}/keyframes"
DECISIONS="${RUN_DIR}/decisions/decisions.jsonl"
TRACES="${RUN_DIR}/exact_decode/traces.jsonl"
SOURCE_BUNDLE="${BASE_RUN_DIR}/preprocess/source_video_bundle.jsonl"
BASE_PROVENANCE="${BASE_RUN_DIR}/mllm_runtime_provenance.json"
for input in "${VALIDATION}" "${DECISIONS}" "${TRACES}" "${SOURCE_BUNDLE}" "${BASE_PROVENANCE}"; do
  [[ -s "${input}" ]] || { echo "missing pre-Qwen input: ${input}" >&2; exit 1; }
done
[[ -z "$(cd -- "${REPO_ROOT}" && git status --porcelain --untracked-files=all)" ]] || {
  echo "production worktree must be clean before Qwen" >&2; exit 1;
}

mapfile -t PREFLIGHT < <("${PYTHON_BIN}" - "${VALIDATION}" "${DECISIONS}" "${TRACES}" "${KEYFRAMES}" "${SOURCE_BUNDLE}" "${BASE_PROVENANCE}" <<'PY'
import hashlib,json,pathlib,re,sys
validation_path,decisions,traces,keyframes,source_bundle,base_path=map(pathlib.Path,sys.argv[1:])
p=json.loads(validation_path.read_text()); base=json.loads(base_path.read_text())
assert p["status"]=="validated" and p["num_rows"]==600 and p["fresh_source_decode"] is True
assert p["qwen_frame_index_pixel_hash_roundtrip"]["logical_selected_hash_comparisons"]==9600
assert p["qwen_frame_index_pixel_hash_roundtrip"]["num_video_decode_passes"]==20
bundle=p["keyframe_bundle"]; assert bundle["num_files"]==10 and bundle["num_rows"]==600 and bundle["trace_indices_exact_match"] is True
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
if sha(decisions)!=p["decisions_sha256"] or sha(traces)!=p["traces_sha256"] or sha(source_bundle)!=p["source_video_bundle_sha256"]: raise SystemExit("validated artifact drift")
expected=bundle["filenames"]
if sorted(x.name for x in keyframes.glob("*.json"))!=expected: raise SystemExit("keyframe grid drift")
h=hashlib.sha256()
for name in expected: h.update(name.encode()); h.update(b'\0'); h.update((keyframes/name).read_bytes()); h.update(b'\0')
if h.hexdigest()!=bundle["sha256_name_nul_bytes_nul"]: raise SystemExit("keyframe content drift")
members=[json.loads(x) for x in source_bundle.read_text().splitlines() if x]
if len(members)!=20: raise SystemExit("source bundle is not dev20")
for row in members:
    path=pathlib.Path(row["path"])
    if not path.is_file() or path.stat().st_size!=row["size_bytes"] or sha(path)!=row["sha256"]: raise SystemExit(f"source member drift: {path}")
required=("schema_version","checkpoint_requested","checkpoint_snapshot_path","checkpoint_snapshot_revision","checkpoint_signature","runtime")
if any(name not in base for name in required): raise SystemExit("base Qwen provenance incomplete")
snapshot=pathlib.Path(base["checkpoint_snapshot_path"])
if not snapshot.is_absolute() or not snapshot.is_dir() or snapshot.name!=base["checkpoint_snapshot_revision"]: raise SystemExit("Qwen snapshot drift")
signature=str(base["checkpoint_signature"])
if signature!=f"snapshot:{snapshot.name}" and not signature.startswith("content:"): raise SystemExit("invalid checkpoint signature")
print(snapshot); print(base["checkpoint_requested"]); print(signature); print(sha(base_path)); print(sha(validation_path))
PY
)
((${#PREFLIGHT[@]} == 5)) || { echo "pre-Qwen provenance validation failed" >&2; exit 1; }
QWEN_SNAPSHOT="${PREFLIGHT[0]}"
QWEN_REQUESTED="${PREFLIGHT[1]}"
QWEN_SIGNATURE="${PREFLIGHT[2]}"
BASE_PROVENANCE_SHA="${PREFLIGHT[3]}"
VALIDATION_SHA="${PREFLIGHT[4]}"
if [[ "${QWEN_SIGNATURE}" == content:* ]]; then
  [[ "${QWEN_SIGNATURE#content:}" == "$(directory_content_hash "${QWEN_SNAPSHOT}")" ]] || { echo "Qwen content signature drift" >&2; exit 1; }
fi

WFS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}" phase_stable wfs preprocess scripts configs lmms-eval-diff)"
LMMS_PACKAGE_DIR="$(${PYTHON_BIN} -c 'import lmms_eval,pathlib; print(pathlib.Path(lmms_eval.__file__).resolve().parent)')"
LMMS_SOURCE_SHA="$(source_tree_hash "${LMMS_PACKAGE_DIR}" .)"
PACKAGE_SIGNATURE="$(${PYTHON_BIN} - <<'PY'
import importlib.metadata,json,platform
names=("torch","transformers","lmms_eval","accelerate","qwen-vl-utils","av","numpy","scipy")
versions={}
for name in names:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
print(json.dumps({"python":platform.python_version(),"packages":versions},sort_keys=True))
PY
)"
RUNTIME_PROVENANCE="${RUN_DIR}/mllm_runtime_provenance.json"
"${PYTHON_BIN}" - "${RUNTIME_PROVENANCE}" "${BASE_PROVENANCE}" "${BASE_PROVENANCE_SHA}" "${VALIDATION_SHA}" "${WFS_SOURCE_SHA}" "${LMMS_SOURCE_SHA}" "${QWEN_REQUESTED}" "${QWEN_SNAPSHOT}" "${QWEN_SIGNATURE}" "${PACKAGE_SIGNATURE}" "$(cd -- "${REPO_ROOT}" && git rev-parse HEAD)" <<'PY'
import json,os,pathlib,sys
(output,inherited,inherited_sha,validation_sha,wfs_sha,lmms_sha,requested,snapshot,checkpoint_signature,packages,head)=sys.argv[1:]
payload={"schema_version":1,"inherited_base_provenance":str(pathlib.Path(inherited).resolve()),"inherited_base_provenance_sha256":inherited_sha,"validation_sha256":validation_sha,"wfs_git_head":head,"wfs_code_sha256":wfs_sha,"lmms_source_sha256":lmms_sha,"checkpoint_requested":requested,"checkpoint_snapshot_path":snapshot,"checkpoint_snapshot_revision":pathlib.Path(snapshot).name,"checkpoint_signature":checkpoint_signature,"runtime":json.loads(packages),"grid":{"methods":["canonical_uniform","phasefuse_rc12"],"origins":[0,1,2,3,4],"independent_cells":10,"uniform_inference_reused":False},"generation":{"max_num_frames":16,"max_pixels":200704,"attention":"sdpa","batch_size":1}}
path=pathlib.Path(output); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
PY
RUNTIME_SIGNATURE="$(sha256_file "${RUNTIME_PROVENANCE}")"

bash "${SCRIPT_DIR}/run_mllm_grid.sh" \
  --benchmark videomme --keyframe-dir "${KEYFRAMES}" --output-root "${RUN_DIR}/mllm" \
  --methods canonical_uniform,phasefuse_rc12 --origins 0,1,2,3,4 --cuda-device 0 \
  --max-num-frames 16 --max-pixels 200704 --attention sdpa --batch-size 1 \
  --python-bin "${PYTHON_BIN}" --converter-python "${PYTHON_BIN}" \
  --repo-root "${REPO_ROOT}" --predictions-output "${RUN_DIR}/primary_predictions.raw.jsonl" \
  --qwen-checkpoint "${QWEN_SNAPSHOT}" --runtime-signature "${RUNTIME_SIGNATURE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_rc12_predictions.py" \
  --primary "${RUN_DIR}/primary_predictions.raw.jsonl" \
  --context-arm "uniform_dense=uniform_dense@${BASE_RUN_DIR}/predictions.jsonl" \
  --context-arm "dense_swt_v2_selector=dense_swt@${BASE_RUN_DIR}/v2_diagnostic/predictions.jsonl" \
  --context-arm "phasefuse_v2=phasefuse_v2@${BASE_RUN_DIR}/v2_diagnostic/predictions.jsonl" \
  --output "${RUN_DIR}/predictions.jsonl" --context-output "${RUN_DIR}/predictions_with_context.jsonl"

cd -- "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable evaluate-phasefuse-predictions \
  --predictions "${RUN_DIR}/predictions.jsonl" --traces "${TRACES}" \
  --output "${RUN_DIR}/downstream_summary.json" --baseline-method canonical_uniform \
  --treatment-method phasefuse_rc12 --expected-methods canonical_uniform phasefuse_rc12 \
  --n-bootstrap 10000 --seed 20260813
PYTHONPATH=. "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_rc12_gate.py" \
  --summary "${RUN_DIR}/downstream_summary.json" --output "${RUN_DIR}/gate_summary.json"
