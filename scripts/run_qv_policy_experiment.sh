#!/usr/bin/env bash
# Foreground, provenance-aware QVHighlights boundary-policy experiment runner.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_VERSION="qv-policy-val100-v1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CALLER_DIR="$(pwd -P)"
cd "${REPO_ROOT}"

die() {
  printf 'run_qv_policy_experiment: ERROR: %s\n' "$*" >&2
  exit 1
}

on_error() {
  local status=$?
  printf 'run_qv_policy_experiment: FAILED at line %s (exit=%s)\n' \
    "${BASH_LINENO[0]}" "${status}" >&2
  exit "${status}"
}
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bash scripts/run_qv_policy_experiment.sh [options]

Runs the isolated QVHighlights val-100 policy experiment in the foreground:
  make-benchmark-manifests (100 videos x 5 origins)
  -> preprocess-benchmark (query-conditioned features/signals)
  -> policy-separation --b-values 8 15 20

Inputs and outputs:
  --manifest PATH           Transformed val100_clear_phase16.jsonl
  --expected-manifest-sha H Expected canonical manifest SHA-256
  --dataset-root PATH       QVHighlights root containing videos/<video_id>.mp4
  --run-dir PATH            Isolated output root
  --config PATH             WFS-SB phase-stable YAML
  --venv-dir PATH           Prepared WFS-SB virtual environment

Runtime:
  --cuda-device N           Logical CUDA device index (default: 0)
  --feature-model NAME      blip2, blip1, clip, or siglip (default: blip2)
  --feature-model-path P    Optional pinned Hugging Face ID/local checkpoint
  --feature-batch-size N    Feature inference batch size (default: 8)
  --frame-buffer-size N     Maximum decoded RGB frames in RAM (default: 256)
  --sample-fps F            Candidate sampling FPS (default: 1.0)
  --seed N                  Sampling/bootstrap seed (default: 20260810)
  --n-bootstrap N           Paired bootstrap repetitions (default: 10000)
  --confidence F            Bootstrap confidence level (default: 0.95)

Resume and safety:
  --resume                  Reuse checksum-valid stages (default)
  --no-resume               Rerun all three stages without deleting artifacts
  --force-step NAME         Rerun manifests, preprocess, policy, or all
                            (repeatable)
  --rehash-inputs           Ignore the raw-video checksum cache this invocation
  --skip-gpu-preflight      Skip NVIDIA/PyTorch CUDA checks (testing only)
  -h, --help                Show this help

Defaults target the existing A100 host layout:
  manifest:     ~/videounderstanding/outputs/local-data/qvhighlights/
                manifests/val100_clear_phase16.jsonl
  dataset root: ~/videounderstanding/outputs/local-data/qvhighlights
  venv:         ~/.venvs/wfs-sb-a100
  run dir:      WFS-SB/artifacts/qvhighlights_policy_val100

The script intentionally remains a foreground process. For a detached run,
the caller may use nohup/systemd and redirect stdout; per-stage logs and atomic
completion markers are always written below RUN_DIR.
EOF
}

DEFAULT_QV_ROOT="${HOME:?HOME must be set}/videounderstanding/outputs/local-data/qvhighlights"
SOURCE_MANIFEST="${QV_POLICY_MANIFEST:-${DEFAULT_QV_ROOT}/manifests/val100_clear_phase16.jsonl}"
EXPECTED_MANIFEST_SHA256="${QV_POLICY_EXPECTED_MANIFEST_SHA256:-5b89434552f242502a5f00636f8aabd9186e8d6ab5c5de98d4372fcbfe4469ad}"
DATASET_ROOT="${QV_POLICY_DATASET_ROOT:-${DEFAULT_QV_ROOT}}"
RUN_DIR="${QV_POLICY_RUN_DIR:-${REPO_ROOT}/artifacts/qvhighlights_policy_val100}"
CONFIG_FILE="${QV_POLICY_CONFIG:-${REPO_ROOT}/configs/phase_stable_icassp.yaml}"
VENV_DIR="${QV_POLICY_VENV_DIR:-${HOME}/.venvs/wfs-sb-a100}"
CUDA_DEVICE="${QV_POLICY_CUDA_DEVICE:-0}"
FEATURE_MODEL="${QV_POLICY_FEATURE_MODEL:-blip2}"
FEATURE_MODEL_PATH="${QV_POLICY_FEATURE_MODEL_PATH:-}"
FEATURE_BATCH_SIZE="${QV_POLICY_FEATURE_BATCH_SIZE:-8}"
FRAME_BUFFER_SIZE="${QV_POLICY_FRAME_BUFFER_SIZE:-256}"
SAMPLE_FPS="${QV_POLICY_SAMPLE_FPS:-1.0}"
SEED="${QV_POLICY_SEED:-20260810}"
N_BOOTSTRAP="${QV_POLICY_N_BOOTSTRAP:-10000}"
CONFIDENCE="${QV_POLICY_CONFIDENCE:-0.95}"
RESUME=1
REHASH_INPUTS=0
SKIP_GPU_PREFLIGHT=0
EXPECTED_VIDEOS=100
NUM_ORIGINS=5
B_VALUES=(8 15 20)
declare -a FORCE_STEPS=()

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while (($#)); do
  case "$1" in
    --manifest) need_value "$@"; SOURCE_MANIFEST="$2"; shift 2 ;;
    --expected-manifest-sha) need_value "$@"; EXPECTED_MANIFEST_SHA256="${2,,}"; shift 2 ;;
    --dataset-root) need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
    --run-dir) need_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG_FILE="$2"; shift 2 ;;
    --venv-dir) need_value "$@"; VENV_DIR="$2"; shift 2 ;;
    --cuda-device) need_value "$@"; CUDA_DEVICE="$2"; shift 2 ;;
    --feature-model) need_value "$@"; FEATURE_MODEL="${2,,}"; shift 2 ;;
    --feature-model-path) need_value "$@"; FEATURE_MODEL_PATH="$2"; shift 2 ;;
    --feature-batch-size) need_value "$@"; FEATURE_BATCH_SIZE="$2"; shift 2 ;;
    --frame-buffer-size) need_value "$@"; FRAME_BUFFER_SIZE="$2"; shift 2 ;;
    --sample-fps) need_value "$@"; SAMPLE_FPS="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --n-bootstrap) need_value "$@"; N_BOOTSTRAP="$2"; shift 2 ;;
    --confidence) need_value "$@"; CONFIDENCE="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-resume) RESUME=0; shift ;;
    --force-step) need_value "$@"; FORCE_STEPS+=("$2"); shift 2 ;;
    --rehash-inputs) REHASH_INPUTS=1; shift ;;
    --skip-gpu-preflight) SKIP_GPU_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

command -v realpath >/dev/null 2>&1 || die "realpath is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"
command -v flock >/dev/null 2>&1 || die "flock is required"

absolute_existing_file() {
  local value="$1"
  [[ "${value}" == /* ]] || value="${CALLER_DIR}/${value}"
  [[ -f "${value}" ]] || die "file not found: ${value}"
  realpath -- "${value}"
}

absolute_existing_dir() {
  local value="$1"
  [[ "${value}" == /* ]] || value="${CALLER_DIR}/${value}"
  [[ -d "${value}" ]] || die "directory not found: ${value}"
  realpath -- "${value}"
}

SOURCE_MANIFEST="$(absolute_existing_file "${SOURCE_MANIFEST}")"
DATASET_ROOT="$(absolute_existing_dir "${DATASET_ROOT}")"
CONFIG_FILE="$(absolute_existing_file "${CONFIG_FILE}")"
VENV_DIR="$(absolute_existing_dir "${VENV_DIR}")"
if [[ "${RUN_DIR}" != /* ]]; then RUN_DIR="${CALLER_DIR}/${RUN_DIR}"; fi
RUN_DIR="$(realpath -m -- "${RUN_DIR}")"
if [[ -n "${FEATURE_MODEL_PATH}" && -e "${FEATURE_MODEL_PATH}" ]]; then
  FEATURE_MODEL_PATH="$(realpath -- "${FEATURE_MODEL_PATH}")"
fi

[[ "${RUN_DIR}" != "${REPO_ROOT}" ]] || die "--run-dir must not be the repository root"
[[ "${RUN_DIR}" != "${DATASET_ROOT}" ]] || die "--run-dir must not be the dataset root"
case "${RUN_DIR}/" in
  "${DATASET_ROOT}/"*) die "--run-dir must be isolated from the dataset root" ;;
esac

[[ "${CUDA_DEVICE}" =~ ^[0-9]+$ ]] || die "--cuda-device must be a non-negative integer"
[[ "${FEATURE_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "--feature-batch-size must be positive"
[[ "${FRAME_BUFFER_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "--frame-buffer-size must be positive"
[[ "${SEED}" =~ ^[0-9]+$ ]] || die "--seed must be a non-negative integer"
[[ "${N_BOOTSTRAP}" =~ ^[1-9][0-9]*$ ]] || die "--n-bootstrap must be positive"
[[ "${EXPECTED_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "--expected-manifest-sha must be a SHA-256 digest"
case "${FEATURE_MODEL}" in
  blip2|blip1|clip|siglip) ;;
  *) die "--feature-model must be blip2, blip1, clip, or siglip" ;;
esac
for step in "${FORCE_STEPS[@]}"; do
  case "${step}" in
    manifests|preprocess|policy|all) ;;
    *) die "--force-step must be manifests, preprocess, policy, or all" ;;
  esac
done

PYTHON_BIN="${VENV_DIR}/bin/python"
[[ -x "${PYTHON_BIN}" ]] || die "WFS-SB venv Python is not executable: ${PYTHON_BIN}"
"${PYTHON_BIN}" - "${SAMPLE_FPS}" "${CONFIDENCE}" <<'PY'
import math
import sys

fps, confidence = map(float, sys.argv[1:])
if not math.isfinite(fps) or fps <= 0:
    raise SystemExit("sample_fps must be finite and positive")
if not math.isfinite(confidence) or not 0 < confidence < 1:
    raise SystemExit("confidence must lie strictly between zero and one")
PY
"${PYTHON_BIN}" -m phase_stable --help >/dev/null

# Resolve every Hugging Face model ID to one immutable local snapshot before
# fingerprinting or preprocessing.  This avoids silently following a moving
# repository branch on a resumed experiment.
FEATURE_MODEL_PATH="$("${PYTHON_BIN}" - "${FEATURE_MODEL}" "${FEATURE_MODEL_PATH}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

defaults = {
    "blip2": "Salesforce/blip2-itm-vit-g",
    "blip1": "Salesforce/blip-itm-base-coco",
    "clip": "openai/clip-vit-base-patch32",
    "siglip": "google/siglip-so400m-patch14-384",
}
model, requested = sys.argv[1:]
candidate = requested or defaults[model]
path = Path(candidate)
if path.exists():
    resolved = path.resolve()
else:
    resolved = Path(snapshot_download(repo_id=candidate)).resolve()
if not resolved.is_dir():
    raise SystemExit(f"resolved feature model is not a directory: {resolved}")
print(resolved)
PY
)"
printf 'Feature model snapshot: %s\n' "${FEATURE_MODEL_PATH}"

STATE_DIR="${RUN_DIR}/.qv_policy_state"
LOG_DIR="${RUN_DIR}/logs"
SAMPLING_MANIFESTS="${RUN_DIR}/sampling_manifests.jsonl"
VIDEO_CATALOG="${RUN_DIR}/video_catalog.jsonl"
RAW_INVENTORY="${RUN_DIR}/raw_video_inventory.json"
RAW_HASH_CACHE="${STATE_DIR}/raw_video_sha256_cache.json"
MANIFEST_PREFLIGHT="${RUN_DIR}/manifest_preflight.json"
PREPROCESS_DIR="${RUN_DIR}/preprocess"
SIGNALS_PATH="${RUN_DIR}/origin_signals.jsonl"
SIGNAL_PREFLIGHT="${RUN_DIR}/signal_preflight.json"
POLICY_DIR="${RUN_DIR}/policy_separation"
POLICY_PREFLIGHT="${RUN_DIR}/policy_preflight.json"
RUN_CONFIG="${RUN_DIR}/run_config.json"
RUNTIME_INFO="${RUN_DIR}/runtime_fingerprint.json"
GPU_INFO="${RUN_DIR}/gpu_preflight.json"
mkdir -p "${RUN_DIR}" "${STATE_DIR}" "${LOG_DIR}"

exec 9>"${STATE_DIR}/run.lock"
flock -n 9 || die "another process holds ${STATE_DIR}/run.lock"

sha256_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

fingerprint_tokens() {
  { for token in "$@"; do printf '%s\0' "${token}"; done; } \
    | sha256sum | awk '{print $1}'
}

render_command() {
  local rendered="" token quoted
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
    find wfs -type f -name '*.py' -print0
    printf '%s\0' preprocess/extract.py
  } | sort -z
)
CODE_HASH="$({
  for path in "${CODE_FILES[@]}"; do
    printf '%s\0%s\0' "${path}" "$(sha256_file "${path}")"
  done
} | sha256sum | awk '{print $1}')"
SOURCE_MANIFEST_HASH="$(sha256_file "${SOURCE_MANIFEST}")"
[[ "${SOURCE_MANIFEST_HASH}" == "${EXPECTED_MANIFEST_SHA256}" ]] || \
  die "manifest SHA-256 is not the locked val100 protocol input: ${SOURCE_MANIFEST_HASH}"
ANALYSIS_CONFIG_HASH="$(sha256_file "${CONFIG_FILE}")"

"${PYTHON_BIN}" - "${RUNTIME_INFO}" <<'PY'
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

packages = {}
for name in ("numpy", "scipy", "PyWavelets", "scikit-learn", "av", "torch", "transformers"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
payload = {
    "python": sys.version,
    "python_executable": str(Path(sys.executable).resolve()),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": packages,
}
destination = Path(sys.argv[1])
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PY
RUNTIME_HASH="$(sha256_file "${RUNTIME_INFO}")"

if ((SKIP_GPU_PREFLIGHT)); then
  printf '{"skipped":true}\n' >"${GPU_INFO}.tmp"
  mv -- "${GPU_INFO}.tmp" "${GPU_INFO}"
  printf 'GPU preflight: skipped by explicit request\n' >&2
else
  command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required"
  NVIDIA_SUMMARY="$(nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader)" \
    || die "nvidia-smi could not query the GPU/driver"
  "${PYTHON_BIN}" - "${CUDA_DEVICE}" "${GPU_INFO}" "${NVIDIA_SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

import torch

index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if index >= torch.cuda.device_count():
    raise SystemExit(
        f"logical CUDA device {index} is unavailable; count={torch.cuda.device_count()}"
    )
torch.cuda.set_device(index)
probe = torch.ones(1, device=f"cuda:{index}")
if float(probe.cpu().item()) != 1.0:
    raise SystemExit("CUDA allocation/copy smoke check failed")
torch.cuda.synchronize(index)
properties = torch.cuda.get_device_properties(index)
payload = {
    "logical_device": index,
    "name": properties.name,
    "total_memory_bytes": properties.total_memory,
    "compute_capability": [properties.major, properties.minor],
    "torch_version": torch.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    "nvidia_smi": sys.argv[3].splitlines(),
}
if "A100" not in properties.name.upper():
    print("warning: defaults were tuned for an A100", file=sys.stderr)
gib = properties.total_memory / 1024**3
if gib < 35:
    raise SystemExit(f"visible CUDA memory is {gib:.1f} GiB; at least 35 GiB is required")
destination = Path(sys.argv[2])
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"GPU preflight: cuda:{index}, {properties.name}, {gib:.1f} GiB")
PY
fi
GPU_HASH="$(sha256_file "${GPU_INFO}")"

MODEL_FINGERPRINT="$("${PYTHON_BIN}" - "${FEATURE_MODEL}" "${FEATURE_MODEL_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model, raw_path = sys.argv[1:]
value = raw_path or f"default:{model}"
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

# The transformed QV manifest is the annotation/query input. Build a content-
# hashed inventory of exactly its 100 unique source videos. Cached raw digests
# are reused only when path, size, and nanosecond mtime all still match.
"${PYTHON_BIN}" - \
  "${SOURCE_MANIFEST}" "${SOURCE_MANIFEST_HASH}" "${DATASET_ROOT}" \
  "${RAW_INVENTORY}" "${RAW_HASH_CACHE}" "${EXPECTED_VIDEOS}" \
  "${REHASH_INPUTS}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from phase_stable.benchmarks import load_benchmark_videos

manifest, manifest_sha, dataset_root, inventory_path, cache_path, expected, rehash = sys.argv[1:]
videos = load_benchmark_videos("qvhighlights", manifest, dataset_root)
expected_count = int(expected)
if len(videos) != expected_count:
    raise SystemExit(f"expected {expected_count} unique videos, found {len(videos)}")
if any(len(video.queries) != 1 for video in videos):
    raise SystemExit("val-100 contract requires exactly one query per video")

raw_rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
annotation_hashes = {row.get("metadata", {}).get("annotation_sha256") for row in raw_rows}
if any(row.get("split") != "validation" for row in raw_rows):
    raise SystemExit("locked QV protocol requires split=validation for every row")
if any(row.get("metadata", {}).get("selection_rule_version") != "qvhighlights_clear_phase_v1" for row in raw_rows):
    raise SystemExit("locked QV protocol selection-rule version mismatch")
if annotation_hashes != {"f668a1eaea156ec5315e14718999cea043a8cf948d3cafbd8e8d655318c3cd02"}:
    raise SystemExit(f"locked QV source annotation hash mismatch: {annotation_hashes}")

cache_file = Path(cache_path)
try:
    cache_payload = json.loads(cache_file.read_text(encoding="utf-8"))
    prior = cache_payload.get("files", {}) if isinstance(cache_payload, dict) else {}
except (FileNotFoundError, json.JSONDecodeError, OSError):
    prior = {}
if rehash == "1":
    prior = {}

def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

rows = []
cache_rows = {}
hashed = 0
for index, video in enumerate(videos, start=1):
    path = video.video_path.resolve()
    if not path.is_file():
        raise SystemExit(f"raw video is missing: {path}")
    stat = path.stat()
    if stat.st_size <= 0:
        raise SystemExit(f"raw video is empty: {path}")
    key = str(path)
    cached = prior.get(key) if isinstance(prior, dict) else None
    sha = None
    if isinstance(cached, dict):
        candidate = cached.get("sha256")
        if (
            cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(candidate, str)
            and len(candidate) == 64
        ):
            sha = candidate
    if sha is None:
        sha = digest_file(path)
        hashed += 1
        if hashed == 1 or hashed % 10 == 0:
            print(f"Raw checksum progress: hashed {hashed} changed/new file(s); row {index}/{len(videos)}", flush=True)
    cache_rows[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha}
    query = video.queries[0]
    rows.append({
        "video_id": video.video_id,
        "question_id": query.question_id,
        "video_path": key,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha,
    })

def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

atomic_json(cache_file, {"schema_version": 1, "files": cache_rows})
atomic_json(Path(inventory_path), {
    "schema_version": 1,
    "benchmark": "qvhighlights",
    "source_manifest": str(Path(manifest).resolve()),
    "source_manifest_sha256": manifest_sha,
    "source_annotation_sha256": next(iter(annotation_hashes)),
    "dataset_root": str(Path(dataset_root).resolve()),
    "num_videos": len(rows),
    "num_queries": len(rows),
    "videos": rows,
})
print(f"Raw-video preflight: {len(rows)} videos; newly hashed={hashed}")
PY
RAW_INVENTORY_HASH="$(sha256_file "${RAW_INVENTORY}")"

CONFIG_HASH="$("${PYTHON_BIN}" - \
  "${RUN_CONFIG}" "${SCRIPT_VERSION}" "${SCRIPT_HASH}" "${CODE_HASH}" \
  "${SOURCE_MANIFEST}" "${SOURCE_MANIFEST_HASH}" "${DATASET_ROOT}" \
  "${RAW_INVENTORY_HASH}" "${CONFIG_FILE}" "${ANALYSIS_CONFIG_HASH}" \
  "${VENV_DIR}" "${RUNTIME_HASH}" "${GPU_HASH}" "${FEATURE_MODEL}" \
  "${FEATURE_MODEL_PATH}" "${MODEL_FINGERPRINT}" "${FEATURE_BATCH_SIZE}" \
  "${FRAME_BUFFER_SIZE}" "${CUDA_DEVICE}" "${SAMPLE_FPS}" "${SEED}" \
  "${N_BOOTSTRAP}" "${CONFIDENCE}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

(
    output, script_version, script_sha, code_sha, source_manifest,
    source_manifest_sha, dataset_root, raw_inventory_sha, config_path,
    config_sha, venv_dir, runtime_sha, gpu_sha, feature_model,
    feature_model_path, model_fingerprint, feature_batch_size,
    frame_buffer_size, cuda_device, sample_fps, seed, n_bootstrap, confidence,
) = sys.argv[1:]
try:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    git_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip())
except (OSError, subprocess.SubprocessError):
    git_commit, git_dirty = None, None
payload = {
    "schema_version": 1,
    "protocol": "qvhighlights-policy-separation-val100",
    "script_version": script_version,
    "script_sha256": script_sha,
    "code_sha256": code_sha,
    "git_commit": git_commit,
    "git_dirty": git_dirty,
    "source_manifest": source_manifest,
    "source_manifest_sha256": source_manifest_sha,
    "dataset_root": dataset_root,
    "raw_inventory_sha256": raw_inventory_sha,
    "config": config_path,
    "config_sha256": config_sha,
    "venv_dir": venv_dir,
    "runtime_sha256": runtime_sha,
    "gpu_preflight_sha256": gpu_sha,
    "feature_model": feature_model,
    "feature_model_path": feature_model_path or None,
    "model_fingerprint": model_fingerprint,
    "feature_batch_size": int(feature_batch_size),
    "frame_buffer_size": int(frame_buffer_size),
    "cuda_device": int(cuda_device),
    "sample_fps": float(sample_fps),
    "sampling_origins": 5,
    "seed": int(seed),
    "b_values": [8, 15, 20],
    "n_bootstrap": int(n_bootstrap),
    "confidence": float(confidence),
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
config_hash = hashlib.sha256(canonical.encode()).hexdigest()
document = {"config_hash": config_hash, **payload}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(config_hash)
PY
)"

is_forced() {
  local requested="$1" forced
  ((RESUME)) || return 0
  for forced in "${FORCE_STEPS[@]}"; do
    [[ "${forced}" == all || "${forced}" == "${requested}" ]] && return 0
  done
  return 1
}

# Specs: file::PATH (nonempty), allow-empty::PATH, signal-bundle::JSONL, or
# trace-bundle::JSONL. Bundle digests bind every referenced feature/NPZ file.
marker_valid() {
  local marker="$1" expected_fingerprint="$2"
  shift 2
  "${PYTHON_BIN}" - "${marker}" "${expected_fingerprint}" "$@" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

marker, expected_fingerprint, *expected_specs = sys.argv[1:]

def file_digest(path: Path, *, allow_empty=False):
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0 and not allow_empty:
        raise ValueError(f"output is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "size": size, "sha256": digest.hexdigest()}

def digest_spec(spec):
    kind, raw_path = spec.split("::", 1)
    source = Path(raw_path)
    if kind == "file":
        return file_digest(source)
    if kind == "allow-empty":
        return file_digest(source, allow_empty=True)
    if kind not in {"signal-bundle", "trace-bundle"}:
        raise ValueError(f"unknown output spec: {kind}")
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
                raise ValueError(f"invalid JSONL {source}:{line_number}: {exc}") from exc
            value = row.get(field)
            if value:
                references.append(str(Path(value).resolve()))
    references = sorted(set(references))
    digest = hashlib.sha256(source_result["sha256"].encode())
    for value in references:
        result = file_digest(Path(value))
        digest.update(value.encode())
        digest.update(result["sha256"].encode())
    return {"kind": kind, "files": 1 + len(references), "sha256": digest.hexdigest()}

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
    "${marker}" "${step}" "${fingerprint}" "${CONFIG_HASH}" \
    "${command_text}" "$@" <<'PY'
import datetime
import hashlib
import json
import sys
from pathlib import Path

marker, step, fingerprint, config_hash, command_text, *specs = sys.argv[1:]

def file_digest(path: Path, *, allow_empty=False):
    if not path.is_file():
        raise SystemExit(f"required output is missing: {path}")
    size = path.stat().st_size
    if size <= 0 and not allow_empty:
        raise SystemExit(f"required output is empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"kind": "file", "size": size, "sha256": digest.hexdigest()}

def digest_spec(spec):
    kind, raw_path = spec.split("::", 1)
    source = Path(raw_path)
    if kind == "file":
        return file_digest(source)
    if kind == "allow-empty":
        return file_digest(source, allow_empty=True)
    if kind not in {"signal-bundle", "trace-bundle"}:
        raise SystemExit(f"unknown output spec: {kind}")
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
                raise SystemExit(f"invalid JSONL {source}:{line_number}: {exc}")
            value = row.get(field)
            if value:
                references.append(str(Path(value).resolve()))
    references = sorted(set(references))
    digest = hashlib.sha256(source_result["sha256"].encode())
    for value in references:
        result = file_digest(Path(value))
        digest.update(value.encode())
        digest.update(result["sha256"].encode())
    return {"kind": kind, "files": 1 + len(references), "sha256": digest.hexdigest()}

outputs = {spec: digest_spec(spec) for spec in specs}
payload = {
    "schema_version": 1,
    "step": step,
    "fingerprint": fingerprint,
    "run_config_hash": config_hash,
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

validate_manifests() {
  "${PYTHON_BIN}" - \
    "${SAMPLING_MANIFESTS}" "${RAW_INVENTORY}" "${MANIFEST_PREFLIGHT}" \
    "${EXPECTED_VIDEOS}" "${NUM_ORIGINS}" "${SEED}" "${SAMPLE_FPS}" \
    "${CONFIG_FILE}" "${B_VALUES[-1]}" <<'PY'
import json
import math
import sys
from pathlib import Path

from phase_stable.config import load_phase_stable_config
from phase_stable.sampling import read_manifests_jsonl
from wfs.core import compute_min_peak_distance

(
    manifest_path, inventory_path, output, expected, origins, seed, fps,
    config_path, max_b,
) = sys.argv[1:]
manifests = read_manifests_jsonl(manifest_path)
inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
experiment = load_phase_stable_config(config_path).experiment
if tuple(experiment.methods) != ("dwt", "swt"):
    raise SystemExit(f"policy protocol requires methods=[dwt,swt], got {experiment.methods}")
if experiment.frame_budget != 16:
    raise SystemExit(f"policy protocol requires frame_budget=16, got {experiment.frame_budget}")
expected_ids = [row["video_id"] for row in inventory["videos"]]
actual_ids = [manifest.video_id for manifest in manifests]
if len(manifests) != int(expected) or actual_ids != expected_ids:
    raise SystemExit("sampling manifest IDs/order do not match the val-100 inventory")
for manifest in manifests:
    if manifest.num_origins != int(origins):
        raise SystemExit(f"origin-count mismatch: {manifest.video_id}")
    if manifest.master_seed != int(seed):
        raise SystemExit(f"sampling seed mismatch: {manifest.video_id}")
    if not math.isclose(manifest.sample_fps, float(fps), rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit(f"sample FPS mismatch: {manifest.video_id}")
    if manifest.candidate_count < 16:
        raise SystemExit(f"fewer than 16 candidate frames: {manifest.video_id}")
    distance = compute_min_peak_distance(
        manifest.candidate_count,
        ratio=experiment.min_distance_ratio,
        absolute_min=experiment.min_distance_absolute,
    )
    maximum_feasible = (manifest.candidate_count - 3) // distance + 1
    if maximum_feasible < int(max_b):
        raise SystemExit(
            f"B={max_b} is infeasible before preprocessing: {manifest.video_id}, "
            f"N={manifest.candidate_count}, distance={distance}, max={maximum_feasible}"
        )
payload = {
    "schema_version": 1,
    "num_videos": len(manifests),
    "num_origins_per_video": int(origins),
    "total_video_origins": len(manifests) * int(origins),
    "sample_fps": float(fps),
    "min_candidate_count": min(item.candidate_count for item in manifests),
    "max_candidate_count": max(item.candidate_count for item in manifests),
    "max_requested_b": int(max_b),
    "all_nested_b_feasible": True,
}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"Manifest audit: {len(manifests)} videos x {origins} origins")
PY
}

validate_preprocess() {
  "${PYTHON_BIN}" - \
    "${SIGNALS_PATH}" "${RAW_INVENTORY}" "${SIGNAL_PREFLIGHT}" \
    "${EXPECTED_VIDEOS}" "${NUM_ORIGINS}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

from phase_stable.artifacts import read_signal_records

signals, inventory_path, output, expected, origins = sys.argv[1:]
records = read_signal_records(signals)
inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
expected_items = {(row["video_id"], row["question_id"]) for row in inventory["videos"]}
expected_origins = set(range(int(origins)))
grouped = defaultdict(set)
for record in records:
    if record.dataset != "qvhighlights":
        raise SystemExit(f"unexpected signal dataset: {record.dataset}")
    key = (record.video_id, record.question_id)
    if key not in expected_items:
        raise SystemExit(f"unexpected signal item: {key}")
    if record.origin_id in grouped[key]:
        raise SystemExit(f"duplicate signal origin: {key}, {record.origin_id}")
    grouped[key].add(record.origin_id)
    if not record.visual_features_path or not Path(record.visual_features_path).is_file():
        raise SystemExit(f"missing visual features: {key}, origin={record.origin_id}")
if set(grouped) != expected_items:
    raise SystemExit("signal item coverage does not match the val-100 inventory")
bad = {key: sorted(value) for key, value in grouped.items() if value != expected_origins}
if bad:
    raise SystemExit(f"signal origin coverage mismatch: {list(bad.items())[:3]}")
expected_records = int(expected) * int(origins)
if len(records) != expected_records:
    raise SystemExit(f"expected {expected_records} signals, found {len(records)}")
payload = {
    "schema_version": 1,
    "num_signal_records": len(records),
    "num_items": len(grouped),
    "origins": sorted(expected_origins),
    "num_feature_files": len({str(Path(row.visual_features_path).resolve()) for row in records}),
}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"Signal audit: {len(records)} records with complete 100 x 5 coverage")
PY
}

validate_policy() {
  "${PYTHON_BIN}" - \
    "${SIGNALS_PATH}" "${POLICY_DIR}" "${POLICY_PREFLIGHT}" \
    "${CONFIG_FILE}" "${EXPECTED_VIDEOS}" "${NUM_ORIGINS}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

from phase_stable.artifacts import read_signal_records
from phase_stable.config import load_phase_stable_config

signals_path, policy_raw, output, config_path, expected, origins = sys.argv[1:]
policy_dir = Path(policy_raw)

def rows(path, *, allow_empty=False):
    if not path.is_file():
        raise SystemExit(f"missing policy artifact: {path}")
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"non-object JSONL row: {path}:{line_number}")
            result.append(value)
    if not result and not allow_empty:
        raise SystemExit(f"policy artifact has no rows: {path}")
    return result

signals = read_signal_records(signals_path)
expected_keys = {
    (record.video_id, record.question_id, record.origin_id) for record in signals
}
if len(expected_keys) != int(expected) * int(origins):
    raise SystemExit("input signals do not have exact val-100 x 5 coverage")
trace_rows = rows(policy_dir / "traces.jsonl")
item_rows = rows(policy_dir / "item_metrics.jsonl")
peak_rows = rows(policy_dir / "peak_rows.jsonl")
fidelity_rows = rows(policy_dir / "fidelity_rows.jsonl")

methods = ("dwt", "swt")
policies = ("adaptive", "topc", "nested_b08", "nested_b15", "nested_b20")
expected_arms = {(key, method, policy) for key in expected_keys for method in methods for policy in policies}
seen = {}
for row in trace_rows:
    key = (str(row.get("video_id")), str(row.get("question_id")), int(row.get("origin_id", -1)))
    method = str(row.get("base_method"))
    policy = str(row.get("policy_id"))
    arm = (key, method, policy)
    if arm in seen:
        raise SystemExit(f"duplicate policy trace arm: {arm}")
    if arm not in expected_arms:
        raise SystemExit(f"unexpected policy trace arm: {arm}")
    peaks = row.get("peaks")
    selected = row.get("selected_indices")
    if not isinstance(peaks, list) or not isinstance(selected, list):
        raise SystemExit(f"trace is missing peak/selection arrays: {arm}")
    if str(row.get("method")) != f"{method}_{policy}":
        raise SystemExit(f"trace method label mismatch: {arm}")
    seen[arm] = row
if set(seen) != expected_arms:
    raise SystemExit(f"policy arm coverage mismatch: expected {len(expected_arms)}, found {len(seen)}")

if len(trace_rows) != len(expected_arms):
    raise SystemExit(f"expected {len(expected_arms)} trace rows, found {len(trace_rows)}")
if len(peak_rows) != len(expected_arms):
    raise SystemExit(f"expected {len(expected_arms)} peak rows, found {len(peak_rows)}")
if len(fidelity_rows) != len(expected_arms):
    raise SystemExit(
        f"expected {len(expected_arms)} QV fidelity rows, found {len(fidelity_rows)}"
    )

expected_metric_arms = {
    (video_id, question_id, f"{method}_{policy}")
    for video_id, question_id, _ in expected_keys
    for method in methods
    for policy in policies
}
actual_metric_arms = {
    (str(row.get("video_id")), str(row.get("question_id")), str(row.get("method")))
    for row in item_rows
}
if len(item_rows) != len(expected_metric_arms) or actual_metric_arms != expected_metric_arms:
    raise SystemExit("item-metric policy arm coverage mismatch")

def index_policy_rows(rows_to_index, label):
    indexed = {}
    for row in rows_to_index:
        key = (
            (str(row.get("video_id")), str(row.get("question_id")), int(row.get("origin_id", -1))),
            str(row.get("base_method")),
            str(row.get("policy_id")),
        )
        if key in indexed:
            raise SystemExit(f"duplicate {label} arm: {key}")
        indexed[key] = row
    if set(indexed) != expected_arms:
        raise SystemExit(f"{label} policy arm coverage mismatch")
    return indexed

peak_by_arm = index_policy_rows(peak_rows, "peak")
index_policy_rows(fidelity_rows, "fidelity")

frame_budget = load_phase_stable_config(config_path).experiment.frame_budget
for row in seen.values():
    if len(row["selected_indices"]) != frame_budget:
        raise SystemExit("policy trace violates configured frame budget")
for key in expected_keys:
    for method in methods:
        adaptive = seen[(key, method, "adaptive")]
        topc = seen[(key, method, "topc")]
        if len(adaptive["peaks"]) != len(topc["peaks"]):
            raise SystemExit(f"Top-C does not preserve native count: {key}, {method}")
        adaptive_policy = adaptive.get("boundary_policy", {})
        topc_policy = topc.get("boundary_policy", {})
        if adaptive_policy.get("candidate_set") != "strict_local_maxima":
            raise SystemExit(f"adaptive candidate universe mismatch: {key}, {method}")
        if topc_policy.get("candidate_set") != "strict_local_maxima":
            raise SystemExit(f"Top-C candidate universe mismatch: {key}, {method}")
        peak_row = peak_by_arm[(key, method, "topc")]
        local_peak_set = set(int(value) for value in peak_row.get("local_peak_indices", []))
        if not set(int(value) for value in topc["peaks"]).issubset(local_peak_set):
            raise SystemExit(f"Top-C selected a non-local-maximum sample: {key}, {method}")
        nested = []
        for count in (8, 15, 20):
            row = seen[(key, method, f"nested_b{count:02d}")]
            if len(row["peaks"]) != count:
                raise SystemExit(
                    f"fixed-B cardinality mismatch: {key}, {method}, requested={count}, actual={len(row['peaks'])}"
                )
            if row.get("boundary_policy", {}).get("candidate_set") != "all_valid_interior_samples":
                raise SystemExit(f"nested-B candidate universe mismatch: {key}, {method}")
            nested.append(set(int(value) for value in row["peaks"]))
        if not nested[0] < nested[1] < nested[2]:
            raise SystemExit(f"nested fixed-B sets are not strict supersets: {key}, {method}")

summary_path = policy_dir / "summary.json"
try:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError) as exc:
    raise SystemExit(f"invalid policy summary: {exc}") from exc
if summary.get("command") != "policy-separation" or summary.get("b_values") != [8, 15, 20]:
    raise SystemExit("policy summary command/B-values mismatch")
if summary.get("num_signal_records") != len(signals) or summary.get("num_traces") != len(trace_rows):
    raise SystemExit("policy summary row counts mismatch")
if summary.get("num_peak_rows") != len(peak_rows) or summary.get("num_fidelity_rows") != len(fidelity_rows):
    raise SystemExit("policy summary diagnostic/fidelity row counts mismatch")

payload = {
    "schema_version": 1,
    "num_signal_records": len(signals),
    "num_trace_rows": len(trace_rows),
    "num_item_metric_rows": len(item_rows),
    "num_peak_rows": len(peak_rows),
    "num_fidelity_rows": len(fidelity_rows),
    "methods": list(methods),
    "policies": list(policies),
    "fixed_b_cardinality_exact": True,
    "nested_prefix_sets_verified": True,
}
destination = Path(output)
temporary = destination.with_suffix(destination.suffix + ".tmp")
temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
print(f"Policy audit: {len(trace_rows)} complete arms; fixed-B cardinality and nesting verified")
PY
}

validate_step() {
  case "$1" in
    manifests) validate_manifests ;;
    preprocess) validate_preprocess ;;
    policy) validate_policy ;;
    *) die "internal error: no validator for step $1" ;;
  esac
}

declare -a STEP_OUTPUTS=()
STEP_COMMAND_DISPLAY=""
run_step() {
  local step="$1" fingerprint="$2"
  shift 2
  local marker="${STATE_DIR}/${step}.done.json"
  local log="${LOG_DIR}/${step}.log"
  local command_text="${STEP_COMMAND_DISPLAY:-$(render_command "$@")}"
  if ! is_forced "${step}" && [[ -f "${marker}" ]]; then
    if marker_valid "${marker}" "${fingerprint}" "${STEP_OUTPUTS[@]}"; then
      if ! validate_step "${step}"; then
        rm -f -- "${marker}"
        die "semantic validation failed for resumed step: ${step}"
      fi
      printf '[resume] %-10s fingerprint, checksums, and semantic audit match\n' "${step}"
      return 0
    fi
  fi
  rm -f -- "${marker}"
  printf '[run]    %-10s %s\n' "${step}" "${command_text}"
  "$@" 2>&1 | tee "${log}"
  if ! validate_step "${step}" 2>&1 | tee -a "${log}"; then
    die "semantic validation failed after step: ${step}"
  fi
  write_marker \
    "${marker}" "${step}" "${fingerprint}" "${command_text}" \
    "${STEP_OUTPUTS[@]}"
  printf '[done]   %-10s marker=%s\n' "${step}" "${marker}"
}

step_fingerprint() {
  local step="$1"
  shift
  fingerprint_tokens "${SCRIPT_VERSION}" "script=${SCRIPT_HASH}" "step=${step}" "$@"
}

declare -a MANIFEST_CMD=(
  "${PYTHON_BIN}" -m phase_stable make-benchmark-manifests
  --benchmark qvhighlights
  --questions-file "${SOURCE_MANIFEST}"
  --dataset-root "${DATASET_ROOT}"
  --output "${SAMPLING_MANIFESTS}"
  --catalog-output "${VIDEO_CATALOG}"
  --seed "${SEED}"
  --num-origins "${NUM_ORIGINS}"
  --sample-fps "${SAMPLE_FPS}"
  --no-probe-missing-duration
)
MANIFEST_FP="$(step_fingerprint manifests \
  "code=${CODE_HASH}" "source=${SOURCE_MANIFEST_HASH}" \
  "raw=${RAW_INVENTORY_HASH}" "$(render_command "${MANIFEST_CMD[@]}")")"
STEP_OUTPUTS=(
  "file::${SAMPLING_MANIFESTS}"
  "file::${VIDEO_CATALOG}"
  "file::${MANIFEST_PREFLIGHT}"
)
STEP_COMMAND_DISPLAY=""
run_step manifests "${MANIFEST_FP}" "${MANIFEST_CMD[@]}"

SAMPLING_MANIFESTS_HASH="$(sha256_file "${SAMPLING_MANIFESTS}")"
declare -a PREPROCESS_CMD=(
  "${PYTHON_BIN}" -m phase_stable preprocess-benchmark
  --benchmark qvhighlights
  --questions-file "${SOURCE_MANIFEST}"
  --dataset-root "${DATASET_ROOT}"
  --manifests "${SAMPLING_MANIFESTS}"
  --output-dir "${PREPROCESS_DIR}"
  --signal-jsonl "${SIGNALS_PATH}"
  --feature-model "${FEATURE_MODEL}"
  --device "cuda:${CUDA_DEVICE}"
  --batch-size "${FEATURE_BATCH_SIZE}"
  --frame-buffer-size "${FRAME_BUFFER_SIZE}"
)
if [[ -n "${FEATURE_MODEL_PATH}" ]]; then
  PREPROCESS_CMD+=(--model-path "${FEATURE_MODEL_PATH}")
fi
PREPROCESS_FP="$(step_fingerprint preprocess \
  "code=${CODE_HASH}" "source=${SOURCE_MANIFEST_HASH}" \
  "raw=${RAW_INVENTORY_HASH}" "sampling=${SAMPLING_MANIFESTS_HASH}" \
  "runtime=${RUNTIME_HASH}" "gpu=${GPU_HASH}" "model=${MODEL_FINGERPRINT}" \
  "$(render_command "${PREPROCESS_CMD[@]}")")"
STEP_OUTPUTS=(
  "signal-bundle::${SIGNALS_PATH}"
  "file::${PREPROCESS_DIR}/manifest/run_manifest.json"
  "file::${PREPROCESS_DIR}/manifest/environment.json"
  "file::${SIGNAL_PREFLIGHT}"
)
STEP_COMMAND_DISPLAY=""
run_step preprocess "${PREPROCESS_FP}" "${PREPROCESS_CMD[@]}"

PREPROCESS_MARKER="${STATE_DIR}/preprocess.done.json"
SIGNAL_BUNDLE_HASH="$("${PYTHON_BIN}" - \
  "${PREPROCESS_MARKER}" "signal-bundle::${SIGNALS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["outputs"][sys.argv[2]]["sha256"])
PY
)"
declare -a POLICY_CMD=(
  "${PYTHON_BIN}" -m phase_stable policy-separation
  "${SIGNALS_PATH}" "${POLICY_DIR}"
  --config "${CONFIG_FILE}"
  --b-values "${B_VALUES[@]}"
  --n-bootstrap "${N_BOOTSTRAP}"
  --confidence "${CONFIDENCE}"
  --seed "${SEED}"
)
POLICY_FP="$(step_fingerprint policy \
  "code=${CODE_HASH}" "signals=${SIGNAL_BUNDLE_HASH}" \
  "config=${ANALYSIS_CONFIG_HASH}" "$(render_command "${POLICY_CMD[@]}")")"
STEP_OUTPUTS=(
  "trace-bundle::${POLICY_DIR}/traces.jsonl"
  "file::${POLICY_DIR}/item_metrics.jsonl"
  "file::${POLICY_DIR}/peak_rows.jsonl"
  "file::${POLICY_DIR}/fidelity_rows.jsonl"
  "file::${POLICY_DIR}/summary.json"
  "file::${POLICY_DIR}/manifest/run_manifest.json"
  "file::${POLICY_DIR}/manifest/environment.json"
  "file::${POLICY_PREFLIGHT}"
)
STEP_COMMAND_DISPLAY=""
run_step policy "${POLICY_FP}" "${POLICY_CMD[@]}"

printf 'QV policy experiment complete.\n'
printf '  summary: %s\n' "${POLICY_DIR}/summary.json"
printf '  audit:   %s\n' "${POLICY_PREFLIGHT}"
printf '  config:  %s\n' "${RUN_CONFIG}"
printf '  logs:    %s\n' "${LOG_DIR}"
