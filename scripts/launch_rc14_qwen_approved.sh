#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

APPROVED=""
RUN_DIR=""
BASE_RUN_DIR=""
RC12_RUN_DIR=""
EVALUATION_ROOT=""
PYTHON_BIN="${HOME}/.venvs/wfs-sb-a100/bin/python"
while (($#)); do
  case "$1" in
    --approved) APPROVED="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --base-run-dir) BASE_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --rc12-run-dir) RC12_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --evaluation-root) EVALUATION_ROOT="$(realpath -m -- "$2")"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "${APPROVED}" == "RC14_DEV20_QWEN" ]] || {
  echo "explicit RC14 approval token required" >&2; exit 1
}
[[ -n "${RUN_DIR}" && -n "${BASE_RUN_DIR}" && -n "${RC12_RUN_DIR}" && -n "${EVALUATION_ROOT}" ]] || {
  echo "run/base/RC12/evaluation directories required" >&2; exit 2
}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
mkdir -p -- "${RUN_DIR}"
exec 9>"${RUN_DIR}/.rc14_qwen.lock"
flock -n 9 || { echo "another RC14 Qwen launcher owns ${RUN_DIR}" >&2; exit 1; }

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
PAIRED_TRACES="${RUN_DIR}/paired_traces.jsonl"
SOURCE_BUNDLE="${BASE_RUN_DIR}/preprocess/source_video_bundle.jsonl"
BASE_PROVENANCE="${BASE_RUN_DIR}/mllm_runtime_provenance.json"
REFERENCE_PREDICTIONS="${RC12_RUN_DIR}/predictions.jsonl"
REFERENCE_VALIDATION="${RC12_RUN_DIR}/validation.json"
EVALUATION_DATA_ROOT="${EVALUATION_ROOT}/datasets/videomme/data"
for input in "${VALIDATION}" "${DECISIONS}" "${TRACES}" "${PAIRED_TRACES}" "${SOURCE_BUNDLE}" "${BASE_PROVENANCE}" "${REFERENCE_PREDICTIONS}" "${REFERENCE_VALIDATION}"; do
  [[ -s "${input}" ]] || { echo "missing RC14 pre-Qwen input: ${input}" >&2; exit 1; }
done
[[ -d "${EVALUATION_DATA_ROOT}" ]] || { echo "missing evaluation dataset root" >&2; exit 1; }
[[ -z "$(cd -- "${REPO_ROOT}" && git status --porcelain --untracked-files=all)" ]] || {
  echo "RC14 production worktree must be clean before Qwen" >&2; exit 1
}

mapfile -t PREFLIGHT < <("${PYTHON_BIN}" - "${VALIDATION}" "${DECISIONS}" "${TRACES}" "${PAIRED_TRACES}" "${KEYFRAMES}" "${SOURCE_BUNDLE}" "${BASE_PROVENANCE}" "${REFERENCE_PREDICTIONS}" "${REFERENCE_VALIDATION}" "${EVALUATION_DATA_ROOT}" <<'PY'
import hashlib,json,pathlib,sys
(validation_path,decisions,traces,paired_traces,keyframes,source_bundle,base_path,
 reference_predictions,reference_validation,evaluation_data_root)=map(pathlib.Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
p=json.loads(validation_path.read_text()); base=json.loads(base_path.read_text())
assert p['status']=='validated' and p['method']=='phasefuse_rc14' and p['num_rows']==300
assert p['fresh_source_decode'] is True
assert p['decisions_sha256']==sha(decisions) and p['traces_sha256']==sha(traces)
assert p['qwen_frame_index_pixel_hash_roundtrip']['logical_selected_hash_comparisons']==4800
assert p['qwen_frame_index_pixel_hash_roundtrip']['num_video_decode_passes']==20
bundle=p['keyframe_bundle']
assert bundle['num_files']==5 and bundle['num_rows']==300 and bundle['trace_indices_exact_match'] is True
expected=bundle['filenames']
if sorted(path.name for path in keyframes.glob('*.json')) != expected: raise SystemExit('RC14 keyframe grid drift')
h=hashlib.sha256()
for name in expected:
    h.update(name.encode()); h.update(b'\0'); h.update((keyframes/name).read_bytes()); h.update(b'\0')
if h.hexdigest()!=bundle['sha256_name_nul_bytes_nul']: raise SystemExit('RC14 keyframe content drift')
reuse=p['canonical_uniform_prediction_reuse']
if reuse['source_predictions_sha256']!=sha(reference_predictions): raise SystemExit('uniform prediction reuse drift')
if len([x for x in paired_traces.read_text().splitlines() if x])!=600: raise SystemExit('paired trace grid drift')
members=[json.loads(line) for line in source_bundle.read_text().splitlines() if line]
if len(members)!=20: raise SystemExit('source bundle is not dev20')
for row in members:
    path=pathlib.Path(row['path']).resolve(strict=True)
    expected_path=(evaluation_data_root/path.name).resolve(strict=True)
    if path!=expected_path or path.stat().st_size!=row['size_bytes'] or sha(path)!=row['sha256']:
        raise SystemExit(f'source member/evaluation root drift: {path}')
required=('checkpoint_requested','checkpoint_snapshot_path','checkpoint_snapshot_revision','checkpoint_signature','runtime')
if any(name not in base for name in required): raise SystemExit('base Qwen provenance incomplete')
snapshot=pathlib.Path(base['checkpoint_snapshot_path'])
if not snapshot.is_absolute() or not snapshot.is_dir() or snapshot.name!=base['checkpoint_snapshot_revision']:
    raise SystemExit('Qwen snapshot drift')
print(snapshot); print(base['checkpoint_requested']); print(base['checkpoint_signature'])
print(sha(base_path)); print(sha(validation_path)); print(sha(reference_predictions)); print(sha(reference_validation))
PY
)
((${#PREFLIGHT[@]} == 7)) || { echo "RC14 pre-Qwen validation failed" >&2; exit 1; }
QWEN_SNAPSHOT="${PREFLIGHT[0]}"
QWEN_REQUESTED="${PREFLIGHT[1]}"
QWEN_SIGNATURE="${PREFLIGHT[2]}"
BASE_PROVENANCE_SHA="${PREFLIGHT[3]}"
VALIDATION_SHA="${PREFLIGHT[4]}"
REFERENCE_PREDICTIONS_SHA="${PREFLIGHT[5]}"
REFERENCE_VALIDATION_SHA="${PREFLIGHT[6]}"
if [[ "${QWEN_SIGNATURE}" == content:* ]]; then
  [[ "${QWEN_SIGNATURE#content:}" == "$(directory_content_hash "${QWEN_SNAPSHOT}")" ]] || exit 1
fi

WFS_SOURCE_SHA="$(source_tree_hash "${REPO_ROOT}" phase_stable wfs preprocess scripts configs lmms-eval-diff)"
LMMS_PACKAGE_DIR="$(${PYTHON_BIN} -c 'import lmms_eval,pathlib; print(pathlib.Path(lmms_eval.__file__).resolve().parent)')"
LMMS_SOURCE_SHA="$(source_tree_hash "${LMMS_PACKAGE_DIR}" .)"
PACKAGE_SIGNATURE="$(${PYTHON_BIN} - <<'PY'
import importlib.metadata,json,platform
names=('torch','transformers','lmms_eval','accelerate','qwen-vl-utils','av','numpy','scipy')
versions={}
for name in names:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]=None
print(json.dumps({'python':platform.python_version(),'packages':versions},sort_keys=True))
PY
)"
RUNTIME_PROVENANCE="${RUN_DIR}/mllm_runtime_provenance.json"
"${PYTHON_BIN}" - "${RUNTIME_PROVENANCE}" "${BASE_PROVENANCE}" "${BASE_PROVENANCE_SHA}" "${VALIDATION_SHA}" "${REFERENCE_PREDICTIONS_SHA}" "${REFERENCE_VALIDATION_SHA}" "${WFS_SOURCE_SHA}" "${LMMS_SOURCE_SHA}" "${QWEN_REQUESTED}" "${QWEN_SNAPSHOT}" "${QWEN_SIGNATURE}" "${PACKAGE_SIGNATURE}" "$(cd -- "${REPO_ROOT}" && git rev-parse HEAD)" "${REPO_ROOT}" "${EVALUATION_ROOT}" <<'PY'
import json,os,pathlib,sys
(output,inherited,inherited_sha,validation_sha,reference_predictions_sha,reference_validation_sha,
 wfs_sha,lmms_sha,requested,snapshot,checkpoint_signature,packages,head,production_root,evaluation_root)=sys.argv[1:]
payload={'schema_version':1,'inherited_base_provenance':str(pathlib.Path(inherited).resolve()),
'inherited_base_provenance_sha256':inherited_sha,'validation_sha256':validation_sha,
'canonical_uniform_reference_predictions_sha256':reference_predictions_sha,
'canonical_uniform_reference_validation_sha256':reference_validation_sha,
'wfs_git_head':head,'wfs_code_sha256':wfs_sha,'lmms_source_sha256':lmms_sha,
'production_repo_root':production_root,'evaluation_root':evaluation_root,
'checkpoint_requested':requested,'checkpoint_snapshot_path':snapshot,
'checkpoint_snapshot_revision':pathlib.Path(snapshot).name,'checkpoint_signature':checkpoint_signature,
'runtime':json.loads(packages),'grid':{'methods':['phasefuse_rc14'],'origins':[0,1,2,3,4],
'independent_cells':5,'canonical_uniform_inference_reused':True,
'canonical_uniform_reuse_reason':'exact keyframe bytes and exact fresh-decode payload hashes'},
'generation':{'max_num_frames':16,'max_pixels':200704,'attention':'sdpa','batch_size':1}}
path=pathlib.Path(output); tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
PY
RUNTIME_SIGNATURE="$(sha256_file "${RUNTIME_PROVENANCE}")"

bash "${SCRIPT_DIR}/run_mllm_grid.sh" \
  --benchmark videomme --keyframe-dir "${KEYFRAMES}" --output-root "${RUN_DIR}/mllm" \
  --methods phasefuse_rc14 --origins 0,1,2,3,4 --cuda-device 0 \
  --max-num-frames 16 --max-pixels 200704 --attention sdpa --batch-size 1 \
  --python-bin "${PYTHON_BIN}" --converter-python "${PYTHON_BIN}" \
  --repo-root "${EVALUATION_ROOT}" --predictions-output "${RUN_DIR}/treatment_predictions.raw.jsonl" \
  --qwen-checkpoint "${QWEN_SNAPSHOT}" --runtime-signature "${RUNTIME_SIGNATURE}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/merge_rc14_predictions.py" \
  --treatment "${RUN_DIR}/treatment_predictions.raw.jsonl" \
  --reference "${REFERENCE_PREDICTIONS}" --validation "${VALIDATION}" \
  --reference-validation "${REFERENCE_VALIDATION}" \
  --output "${RUN_DIR}/predictions.jsonl" \
  --provenance-output "${RUN_DIR}/prediction_reuse_provenance.json"

cd -- "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable evaluate-phasefuse-predictions \
  --predictions "${RUN_DIR}/predictions.jsonl" --traces "${PAIRED_TRACES}" \
  --output "${RUN_DIR}/downstream_summary.json" --baseline-method canonical_uniform \
  --treatment-method phasefuse_rc14 --expected-methods canonical_uniform phasefuse_rc14 \
  --n-bootstrap 10000 --seed 20260813
PYTHONPATH=. "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_canonical_gate.py" \
  --summary "${RUN_DIR}/downstream_summary.json" --output "${RUN_DIR}/gate_summary.json" \
  --treatment-method phasefuse_rc14
