#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rc12_exact_diagnostic.sh [options]
  --base-run-dir DIR   authenticated dev20 base (default artifacts/phasefuse_videomme_dev20)
  --run-dir DIR        isolated output (default artifacts/phasefuse_rc12_exact_dev20)
  --python-bin FILE    Python executable
  --force-decisions    rebuild canonical target decisions
  --force-decode       rebuild fresh exact-decode traces
  --prepare-qwen       validate/export and print the Qwen command; never launches it

This runner stops before Qwen by design. The generated launch script requires
an explicit --approved token and runs only PhaseFuse-RC12 plus the deduplicated
canonical-uniform control.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_RUN_DIR="${REPO_ROOT}/artifacts/phasefuse_videomme_dev20"
RUN_DIR="${REPO_ROOT}/artifacts/phasefuse_rc12_exact_dev20"
PYTHON_BIN="${HOME}/.venvs/wfs-sb-a100/bin/python"
FORCE_DECISIONS=0
FORCE_DECODE=0
PREPARE_QWEN=0
while (($#)); do
  case "$1" in
    --base-run-dir) BASE_RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --run-dir) RUN_DIR="$(realpath -m -- "$2")"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --force-decisions) FORCE_DECISIONS=1; shift ;;
    --force-decode) FORCE_DECODE=1; shift ;;
    --prepare-qwen) PREPARE_QWEN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SIGNALS="${BASE_RUN_DIR}/preprocess/dense_signals.jsonl"
CATALOG="${BASE_RUN_DIR}/preprocess/catalog.jsonl"
MANIFESTS="${BASE_RUN_DIR}/preprocess/multiphase_manifests.jsonl"
SOURCE_VIDEO_BUNDLE="${BASE_RUN_DIR}/preprocess/source_video_bundle.jsonl"
BASE_COMPLETE="${BASE_RUN_DIR}/preprocess/.complete"
DECISION_DIR="${RUN_DIR}/decisions"
DECISIONS="${DECISION_DIR}/decisions.jsonl"
DECODE_DIR="${RUN_DIR}/exact_decode"
TRACES="${DECODE_DIR}/traces.jsonl"
KEYFRAMES="${RUN_DIR}/keyframes"
QUESTIONS="${REPO_ROOT}/datasets/videomme/videomme_json_file.json"
DATASET_ROOT="${REPO_ROOT}/datasets/videomme"
mkdir -p -- "${RUN_DIR}"
exec 9>"${RUN_DIR}/.rc12.lock"
flock -n 9 || { echo "another RC12 runner owns ${RUN_DIR}" >&2; exit 1; }
for input in "${SIGNALS}" "${CATALOG}" "${MANIFESTS}" "${SOURCE_VIDEO_BUNDLE}" "${BASE_COMPLETE}" "${QUESTIONS}"; do
  [[ -s "${input}" ]] || { echo "missing input: ${input}" >&2; exit 1; }
done

sha256_file() { "${PYTHON_BIN}" - "$1" <<'PY'
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
print(h.hexdigest())
PY
}
[[ "$(sha256_file "${SIGNALS}")" == "cca842810d063f714ce3c6655baf672e22255b6c8714f2b363a56a7a4b6dce66" ]] || { echo "frozen dev20 dense_signals SHA mismatch" >&2; exit 1; }
[[ "$(sha256_file "${MANIFESTS}")" == "d493f6763ef0196196a8014d1006a219426bc2e013c1c5b687410a0d4e2378c3" ]] || { echo "frozen dev20 manifest SHA mismatch" >&2; exit 1; }
"${PYTHON_BIN}" - "${SOURCE_VIDEO_BUNDLE}" "${CATALOG}" <<'PY'
import hashlib,json,pathlib,sys
bundle_path,catalog_path=map(pathlib.Path,sys.argv[1:])
bundle=[json.loads(x) for x in bundle_path.read_text().splitlines() if x]
catalog=[json.loads(x) for x in catalog_path.read_text().splitlines() if x]
if len(bundle)!=20 or len(catalog)!=20: raise SystemExit("expected exact dev20 source grid")
members={}
for row in bundle:
    if set(row)!={"path","sha256","size_bytes"}: raise SystemExit("invalid source bundle schema")
    path=pathlib.Path(row["path"]).resolve()
    if path in members or not path.is_file() or path.stat().st_size!=row["size_bytes"]: raise SystemExit(f"invalid source member {path}")
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    if h.hexdigest()!=row["sha256"]: raise SystemExit(f"source SHA mismatch {path}")
    members[path]=row
paths=[pathlib.Path(row["video_path"]).resolve() for row in catalog]
if len(set(paths))!=20 or set(paths)!=set(members): raise SystemExit("catalog/source bundle path mismatch")
if len({str(row["video_id"]) for row in catalog})!=20 or any(int(row.get("num_questions",-1))!=3 for row in catalog): raise SystemExit("catalog is not 20 videos x 3 questions")
print("Validated 20 source bundle members and exact catalog binding")
PY
SOURCE_SHA="$(cd -- "${REPO_ROOT}" && git ls-files -co --exclude-standard -- phase_stable scripts configs | sort | while IFS= read -r f; do [[ -f "$f" ]] && sha256sum -- "$f"; done | sha256sum | cut -d' ' -f1)"
DECISION_FP="$(printf '%s' "rc12_decision_v1|${SOURCE_SHA}|$(sha256_file "${SIGNALS}")|$(sha256_file "${CATALOG}")|$(sha256_file "${MANIFESTS}")|$(sha256_file "${SOURCE_VIDEO_BUNDLE}")|$(sha256_file "${BASE_COMPLETE}")" | sha256sum | cut -d' ' -f1)"
DECODE_FP="$(printf '%s' "rc12_exact_decode_v1|${SOURCE_SHA}|${DECISION_FP}" | sha256sum | cut -d' ' -f1)"
marker_valid() {
  local marker="$1" expected="$2" output="$3"
  [[ -s "$marker" && -s "$output" ]] || return 1
  "${PYTHON_BIN}" - "$marker" "$expected" "$output" <<'PY'
import hashlib,json,sys
m=json.load(open(sys.argv[1])); data=open(sys.argv[3],'rb').read()
raise SystemExit(0 if m.get('fingerprint')==sys.argv[2] and m.get('output_sha256')==hashlib.sha256(data).hexdigest() else 1)
PY
}
write_marker() {
  "${PYTHON_BIN}" - "$1" "$2" "$3" <<'PY'
import hashlib,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[3]); payload={'fingerprint':sys.argv[2],'output':str(out.resolve()),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

if ((FORCE_DECISIONS)) || ! marker_valid "${DECISION_DIR}/.complete" "${DECISION_FP}" "${DECISIONS}"; then
  cd -- "${REPO_ROOT}"
  PYTHONPATH=. "${PYTHON_BIN}" scripts/build_rc12_decisions.py \
    --signals "${SIGNALS}" --catalog "${CATALOG}" --manifests "${MANIFESTS}" \
    --output-dir "${DECISION_DIR}"
  write_marker "${DECISION_DIR}/.complete" "${DECISION_FP}" "${DECISIONS}"
fi

# The exact decoder owns dynamic decoded-frame duplicate repair. It decodes the
# video-level primary union, then only on-demand repair-round unions; the full
# lattice and frozen scores define repair priority but are never decoded eagerly.
if ((FORCE_DECODE)) || ! marker_valid "${DECODE_DIR}/.complete" "${DECODE_FP}" "${TRACES}"; then
  cd -- "${REPO_ROOT}"
  PYTHONPATH=. "${PYTHON_BIN}" scripts/decode_rc12_exact.py \
    --decisions "${DECISIONS}" --output-dir "${DECODE_DIR}"
  write_marker "${DECODE_DIR}/.complete" "${DECODE_FP}" "${TRACES}"
fi

cd -- "${REPO_ROOT}"
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable analyze-phasefuse \
  --traces "${TRACES}" --output "${DECODE_DIR}/analysis_summary.json" \
  --baseline-method canonical_uniform --treatment-method phasefuse_rc12 \
  --n-bootstrap 10000 --seed 20260813
PYTHONPATH=. "${PYTHON_BIN}" -m phase_stable export-keyframes \
  --traces "${TRACES}" --benchmark videomme --questions-file "${QUESTIONS}" \
  --dataset-root "${DATASET_ROOT}" --output-dir "${KEYFRAMES}" \
  --methods canonical_uniform phasefuse_rc12 --origin-ids 0 1 2 3 4 \
  --expected-budget 16 --allow-annotation-subset

PYTHONPATH=. "${PYTHON_BIN}" scripts/validate_rc12_exact_artifacts.py \
  --decisions "${DECISIONS}" --traces "${TRACES}" --keyframes "${KEYFRAMES}" \
  --source-video-bundle "${SOURCE_VIDEO_BUNDLE}" \
  --output "${RUN_DIR}/validation.json"

if ((PREPARE_QWEN)); then
  cat <<EOF
Selector/exact decode validated. Qwen remains NOT launched.
After explicit approval only:
  bash scripts/launch_rc12_qwen_approved.sh --approved RC12_DEV20_QWEN \\
    --run-dir '${RUN_DIR}' --base-run-dir '${BASE_RUN_DIR}' --python-bin '${PYTHON_BIN}'
EOF
fi
echo "RC12 exact selector diagnostic complete: ${RUN_DIR}"
