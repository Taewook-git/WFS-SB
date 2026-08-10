#!/usr/bin/env bash
# Bootstrap an A100 Linux host for the phase-stable WFS-SB experiments.

set -Eeuo pipefail

readonly DEFAULT_REPO_URL="https://github.com/MAC-AutoML/WFS-SB.git"
readonly DEFAULT_REPO_BRANCH="main"
readonly DEFAULT_LMMS_URL="https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
readonly LMMS_COMMIT="bb1ebe76e7a942386c25c4664f902e0e59e8a401"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
EMBEDDED_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DETECTED_REPO_URL="${DEFAULT_REPO_URL}"
DETECTED_REPO_BRANCH="${DEFAULT_REPO_BRANCH}"
if [[ -d "${EMBEDDED_REPO_ROOT}/.git" ]] && command -v git >/dev/null 2>&1; then
  DETECTED_REPO_URL="$(
    git -C "${EMBEDDED_REPO_ROOT}" remote get-url origin 2>/dev/null || \
      printf '%s\n' "${DEFAULT_REPO_URL}"
  )"
  detected_branch="$(
    git -C "${EMBEDDED_REPO_ROOT}" branch --show-current 2>/dev/null || true
  )"
  if [[ -n "${detected_branch}" ]]; then
    DETECTED_REPO_BRANCH="${detected_branch}"
  fi
  REPO_DIR="${WFS_REPO_DIR:-${EMBEDDED_REPO_ROOT}}"
else
  REPO_DIR="${WFS_REPO_DIR:-${HOME}/WFS-SB}"
fi
REPO_URL="${WFS_REPO_URL:-${DETECTED_REPO_URL}}"
REPO_BRANCH="${WFS_REPO_BRANCH:-${WFS_BRANCH:-${DETECTED_REPO_BRANCH}}}"
PYTHON_BIN="${WFS_PYTHON:-python3.10}"
VENV_DIR="${WFS_VENV_DIR:-}"
LMMS_DIR="${WFS_LMMS_DIR:-}"
LMMS_URL="${WFS_LMMS_URL:-${DEFAULT_LMMS_URL}}"
DATASET_ROOT="${WFS_DATASET_ROOT:-}"
VIDEOMME_ROOT="${WFS_VIDEOMME_ROOT:-}"
MLVU_ROOT="${WFS_MLVU_ROOT:-}"
LVB_ROOT="${WFS_LVB_ROOT:-}"
DATASETS="${WFS_DATASETS:-videomme,mlvu,lvb}"
DATASET_CHECK="${WFS_DATASET_CHECK:-full}"
NON_INTERACTIVE="${WFS_NON_INTERACTIVE:-0}"

TEMP_PATHS=()

log() {
  printf '[bootstrap-a100] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

cleanup() {
  local path
  for path in "${TEMP_PATHS[@]:-}"; do
    if [[ -n "${path}" ]]; then
      rm -f -- "${path}" || true
    fi
  done
  return 0
}

on_error() {
  local exit_code=$?
  log "FAILED at line ${BASH_LINENO[0]} (exit ${exit_code})."
  exit "${exit_code}"
}

trap cleanup EXIT
trap on_error ERR

usage() {
  cat <<'EOF'
Usage: bootstrap_a100.sh [options]

Safely sync WFS-SB, create a Python 3.10 venv, install the exact patched
lmms-eval checkout, authenticate with Hugging Face, validate datasets, and
run CUDA/A100 smoke checks.

Options:
  --repo-url URL             WFS-SB Git URL (env: WFS_REPO_URL)
  --branch BRANCH            WFS-SB branch (env: WFS_REPO_BRANCH or WFS_BRANCH)
  --repo-dir PATH            WFS-SB checkout path (env: WFS_REPO_DIR)
  --python PATH              Python 3.10 executable (env: WFS_PYTHON)
  --venv-dir PATH            Virtualenv path (env: WFS_VENV_DIR)
  --lmms-url URL             lmms-eval Git URL (env: WFS_LMMS_URL)
  --lmms-dir PATH            lmms-eval checkout path (env: WFS_LMMS_DIR)
  --dataset-root PATH        Parent of benchmark roots (env: WFS_DATASET_ROOT)
  --videomme-root PATH       VideoMME root override (env: WFS_VIDEOMME_ROOT)
  --mlvu-root PATH           MLVU root override (env: WFS_MLVU_ROOT)
  --lvb-root PATH            LongVideoBench root override (env: WFS_LVB_ROOT)
  --datasets LIST            Comma list: videomme,mlvu,lvb (env: WFS_DATASETS)
  --dataset-check MODE       paths or full (env: WFS_DATASET_CHECK; default: full)
  --non-interactive          Never prompt for Hugging Face login
  -h, --help                 Show this help and exit

Authentication:
  Set HF_TOKEN in the environment, use a prior official `hf auth login`, or
  allow this script to invoke interactive `hf auth login`. The token is never
  passed as a command-line argument or written directly by this script.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "${2}" ]] || die "${1} requires a value"
}

while (($#)); do
  case "$1" in
    --repo-url)
      need_value "$@"; REPO_URL="$2"; shift 2 ;;
    --branch|--repo-branch)
      need_value "$@"; REPO_BRANCH="$2"; shift 2 ;;
    --repo-dir)
      need_value "$@"; REPO_DIR="$2"; shift 2 ;;
    --python)
      need_value "$@"; PYTHON_BIN="$2"; shift 2 ;;
    --venv-dir)
      need_value "$@"; VENV_DIR="$2"; shift 2 ;;
    --lmms-url)
      need_value "$@"; LMMS_URL="$2"; shift 2 ;;
    --lmms-dir)
      need_value "$@"; LMMS_DIR="$2"; shift 2 ;;
    --dataset-root)
      need_value "$@"; DATASET_ROOT="$2"; shift 2 ;;
    --videomme-root)
      need_value "$@"; VIDEOMME_ROOT="$2"; shift 2 ;;
    --mlvu-root)
      need_value "$@"; MLVU_ROOT="$2"; shift 2 ;;
    --lvb-root)
      need_value "$@"; LVB_ROOT="$2"; shift 2 ;;
    --datasets)
      need_value "$@"; DATASETS="$2"; shift 2 ;;
    --dataset-check)
      need_value "$@"; DATASET_CHECK="$2"; shift 2 ;;
    --non-interactive)
      NON_INTERACTIVE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "unknown option: $1" ;;
  esac
done

[[ -n "${REPO_URL}" ]] || die "repository URL must not be empty"
[[ -n "${REPO_BRANCH}" ]] || die "repository branch must not be empty"
[[ "${DATASET_CHECK}" == "paths" || "${DATASET_CHECK}" == "full" ]] || \
  die "--dataset-check must be 'paths' or 'full'"
[[ "${NON_INTERACTIVE}" == "0" || "${NON_INTERACTIVE}" == "1" ]] || \
  die "WFS_NON_INTERACTIVE must be 0 or 1"

command -v git >/dev/null 2>&1 || die "git is required"
git check-ref-format --branch "${REPO_BRANCH}" >/dev/null 2>&1 || \
  die "invalid Git branch name: ${REPO_BRANCH}"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || \
  [[ -x "${PYTHON_BIN}" ]] || die "Python executable not found: ${PYTHON_BIN}"

validate_python_310() {
  local executable="$1"
  local version
  version="$("${executable}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "${version}" == "3.10" ]] || \
    die "Python 3.10 is required; ${executable} reports ${version}"
}

validate_python_310 "${PYTHON_BIN}"

canonical_path() {
  "${PYTHON_BIN}" -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "$1"
}

REPO_DIR="$(canonical_path "${REPO_DIR}")"

sync_main_repository() {
  local parent remote current_branch local_only remote_only
  if [[ ! -e "${REPO_DIR}" ]]; then
    parent="$(dirname -- "${REPO_DIR}")"
    mkdir -p -- "${parent}"
    log "Cloning WFS-SB branch ${REPO_BRANCH} into ${REPO_DIR}"
    git clone --branch "${REPO_BRANCH}" --single-branch -- "${REPO_URL}" "${REPO_DIR}"
  elif [[ ! -d "${REPO_DIR}/.git" ]]; then
    if [[ -d "${REPO_DIR}" && -z "$(find "${REPO_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      log "Cloning WFS-SB branch ${REPO_BRANCH} into empty ${REPO_DIR}"
      git clone --branch "${REPO_BRANCH}" --single-branch -- "${REPO_URL}" "${REPO_DIR}"
    else
      die "repository path exists but is not an empty directory or Git checkout: ${REPO_DIR}"
    fi
  else
    if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=normal)" ]]; then
      die "WFS-SB checkout is dirty; commit/stash/remove local changes before bootstrap: ${REPO_DIR}"
    fi
    remote="$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null)" || \
      die "WFS-SB checkout has no origin remote: ${REPO_DIR}"
    [[ "${remote}" == "${REPO_URL}" ]] || \
      die "origin URL differs from --repo-url; pass the checkout's URL explicitly"
    log "Fetching WFS-SB branch ${REPO_BRANCH}"
    git -C "${REPO_DIR}" fetch origin \
      "+refs/heads/${REPO_BRANCH}:refs/remotes/origin/${REPO_BRANCH}"
    if git -C "${REPO_DIR}" show-ref --verify --quiet "refs/heads/${REPO_BRANCH}"; then
      git -C "${REPO_DIR}" checkout --quiet "${REPO_BRANCH}"
    else
      git -C "${REPO_DIR}" checkout --quiet -b "${REPO_BRANCH}" --track "origin/${REPO_BRANCH}"
    fi
    current_branch="$(git -C "${REPO_DIR}" branch --show-current)"
    [[ "${current_branch}" == "${REPO_BRANCH}" ]] || \
      die "failed to select requested branch ${REPO_BRANCH}"
    read -r local_only remote_only < <(
      git -C "${REPO_DIR}" rev-list --left-right --count \
        "HEAD...origin/${REPO_BRANCH}"
    )
    [[ "${local_only}" == "0" ]] || \
      die "local branch contains commits not present on origin/${REPO_BRANCH}; refusing to overwrite"
    if [[ "${remote_only}" != "0" ]]; then
      log "Fast-forwarding WFS-SB by ${remote_only} commit(s)"
      git -C "${REPO_DIR}" merge --ff-only "origin/${REPO_BRANCH}"
    else
      log "WFS-SB checkout is already current"
    fi
  fi
}

sync_main_repository
REPO_DIR="$(cd -- "${REPO_DIR}" && pwd -P)"

VENV_DIR="${VENV_DIR:-${HOME}/.venvs/wfs-sb-a100}"
LMMS_DIR="${LMMS_DIR:-${REPO_DIR}/lmms-eval}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_DIR}/datasets}"
VIDEOMME_ROOT="${VIDEOMME_ROOT:-${DATASET_ROOT}/videomme}"
MLVU_ROOT="${MLVU_ROOT:-${DATASET_ROOT}/mlvu}"
LVB_ROOT="${LVB_ROOT:-${DATASET_ROOT}/longvideobench}"

VENV_DIR="$(canonical_path "${VENV_DIR}")"
LMMS_DIR="$(canonical_path "${LMMS_DIR}")"
DATASET_ROOT="$(canonical_path "${DATASET_ROOT}")"
VIDEOMME_ROOT="$(canonical_path "${VIDEOMME_ROOT}")"
MLVU_ROOT="$(canonical_path "${MLVU_ROOT}")"
LVB_ROOT="$(canonical_path "${LVB_ROOT}")"

case "${VENV_DIR}" in
  "${REPO_DIR}"|"${REPO_DIR}"/*)
    if ! git -C "${REPO_DIR}" check-ignore --quiet --no-index -- \
      "${VENV_DIR}/.wfs-bootstrap-ignore-check"; then
      die "venv inside the repository is not ignored and would make safe reruns fail; choose --venv-dir outside the repo"
    fi
    ;;
esac

normalize_datasets() {
  local raw normalized
  local -A seen=()
  REQUESTED_DATASETS=()
  IFS=',' read -r -a raw <<<"${DATASETS}"
  for normalized in "${raw[@]}"; do
    normalized="${normalized,,}"
    normalized="${normalized//[[:space:]]/}"
    [[ "${normalized}" == "longvideobench" ]] && normalized="lvb"
    case "${normalized}" in
      videomme|mlvu|lvb) ;;
      "") continue ;;
      *) die "unsupported dataset in --datasets: ${normalized}" ;;
    esac
    if [[ -z "${seen[${normalized}]:-}" ]]; then
      REQUESTED_DATASETS+=("${normalized}")
      seen["${normalized}"]=1
    fi
  done
  ((${#REQUESTED_DATASETS[@]} > 0)) || die "--datasets must select at least one benchmark"
}

normalize_datasets

validate_dataset_paths() {
  local dataset root annotation raw_dir first_video
  for dataset in "${REQUESTED_DATASETS[@]}"; do
    case "${dataset}" in
      videomme)
        root="${VIDEOMME_ROOT}"
        annotation="${root}/videomme_json_file.json"
        raw_dir="${root}/data"
        ;;
      mlvu)
        root="${MLVU_ROOT}"
        annotation="${root}/mlvu_dev.json"
        raw_dir="${root}/video"
        ;;
      lvb)
        root="${LVB_ROOT}"
        annotation="${root}/lvb_val.json"
        raw_dir="${root}/videos"
        ;;
    esac
    [[ -d "${root}" && -r "${root}" && -x "${root}" ]] || \
      die "missing or unreadable ${dataset} dataset root: ${root}"
    [[ -r "${annotation}" ]] || die "missing/read-protected ${dataset} annotation: ${annotation}"
    if [[ ! -e "${raw_dir}" ]]; then
      [[ "${DATASET_CHECK}" == "paths" ]] || \
        die "missing ${dataset} raw-video directory: ${raw_dir}"
      mkdir -p -- "${raw_dir}"
      log "Created empty ${dataset} raw-video directory for a later download: ${raw_dir}"
    fi
    [[ -d "${raw_dir}" && -r "${raw_dir}" && -x "${raw_dir}" ]] || \
      die "missing or unreadable ${dataset} raw-video directory: ${raw_dir}"
    if [[ "${DATASET_CHECK}" == "full" ]]; then
      first_video="$(find "${raw_dir}" -type f \
        \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.webm' \) \
        -print -quit)"
      [[ -n "${first_video}" ]] || die "no video files found in ${raw_dir}"
    fi
  done
}

# Validate annotations early; paths mode intentionally permits an empty raw directory
# so the authenticated environment can subsequently run a dataset fetcher.
validate_dataset_paths

if [[ -e "${VENV_DIR}" && ! -x "${VENV_DIR}/bin/python" ]]; then
  die "venv path exists but has no executable bin/python: ${VENV_DIR}"
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Creating Python 3.10 virtual environment at ${VENV_DIR}"
  mkdir -p -- "$(dirname -- "${VENV_DIR}")"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
validate_python_310 "${VENV_PYTHON}"

validate_virtualenv() {
  local executable="$1" expected="$2"
  "${executable}" - "${expected}" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
actual = Path(sys.prefix).resolve()
if sys.prefix == sys.base_prefix or actual != expected:
    raise SystemExit(f"not the selected virtualenv: expected={expected}, actual={actual}")
PY
}

validate_virtualenv "${VENV_PYTHON}" "${VENV_DIR}" || \
  die "selected bin/python is not the requested virtualenv: ${VENV_DIR}"

log "Upgrading pip/setuptools/wheel"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

[[ -r "${REPO_DIR}/requirements.txt" ]] || die "missing requirements.txt"
[[ -r "${REPO_DIR}/requirements-phase-stable.txt" ]] || \
  die "missing requirements-phase-stable.txt"

log "Installing WFS-SB dependencies"
"${VENV_PYTHON}" -m pip install -r "${REPO_DIR}/requirements.txt"
"${VENV_PYTHON}" -m pip install -r "${REPO_DIR}/requirements-phase-stable.txt"

patch_paths() {
  git -C "${LMMS_DIR}" apply --numstat "${PATCH_FILE}" | awk -F '\t' '{print $3}'
}

validate_exact_lmms_patch() {
  local status_entry path expected_id actual_id temp_index temp_diff
  local -a expected_paths=()
  local -A expected_lookup=()

  mapfile -t expected_paths < <(patch_paths)
  ((${#expected_paths[@]} > 0)) || die "lmms-eval patch contains no paths"
  for path in "${expected_paths[@]}"; do
    expected_lookup["${path}"]=1
  done

  while IFS= read -r -d '' status_entry; do
    path="${status_entry:3}"
    [[ -n "${expected_lookup[${path}]:-}" ]] || \
      die "lmms-eval has a change outside the supplied patch: ${path}"
  done < <(git -C "${LMMS_DIR}" status --porcelain=v1 -z --untracked-files=normal)

  git -C "${LMMS_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1 || \
    die "lmms-eval is partially patched or conflicts with the supplied patch"

  temp_index="$(mktemp)"
  temp_diff="$(mktemp)"
  TEMP_PATHS+=("${temp_index}" "${temp_diff}")
  rm -f -- "${temp_index}"
  GIT_INDEX_FILE="${temp_index}" git -C "${LMMS_DIR}" read-tree "${LMMS_COMMIT}"
  GIT_INDEX_FILE="${temp_index}" git -C "${LMMS_DIR}" add -A -- "${expected_paths[@]}"
  GIT_INDEX_FILE="${temp_index}" git -C "${LMMS_DIR}" diff --cached --binary \
    "${LMMS_COMMIT}" -- >"${temp_diff}"

  expected_id="$(git patch-id --stable <"${PATCH_FILE}" | awk 'NR == 1 {print $1}')"
  actual_id="$(git patch-id --stable <"${temp_diff}" | awk 'NR == 1 {print $1}')"
  [[ -n "${expected_id}" && "${actual_id}" == "${expected_id}" ]] || \
    die "lmms-eval worktree is not exactly the supplied WFS patch"
}

setup_lmms_eval() {
  local remote head
  PATCH_FILE="${REPO_DIR}/lmms-eval-diff/lmms_eval_wfs.patch"
  [[ -r "${PATCH_FILE}" ]] || die "missing lmms-eval patch: ${PATCH_FILE}"

  if [[ ! -e "${LMMS_DIR}" ]]; then
    mkdir -p -- "$(dirname -- "${LMMS_DIR}")"
    log "Cloning lmms-eval"
    git clone -- "${LMMS_URL}" "${LMMS_DIR}"
  elif [[ ! -d "${LMMS_DIR}/.git" ]]; then
    die "lmms-eval path exists but is not a Git checkout: ${LMMS_DIR}"
  fi

  remote="$(git -C "${LMMS_DIR}" remote get-url origin 2>/dev/null)" || \
    die "lmms-eval checkout has no origin remote: ${LMMS_DIR}"
  [[ "${remote}" == "${LMMS_URL}" ]] || \
    die "lmms-eval origin differs from --lmms-url"

  git -C "${LMMS_DIR}" fetch --quiet origin "${LMMS_COMMIT}"
  head="$(git -C "${LMMS_DIR}" rev-parse HEAD)"
  if [[ "${head}" != "${LMMS_COMMIT}" ]]; then
    [[ -z "$(git -C "${LMMS_DIR}" status --porcelain --untracked-files=normal)" ]] || \
      die "lmms-eval is dirty at a non-base commit; refusing checkout"
    git -C "${LMMS_DIR}" checkout --quiet --detach "${LMMS_COMMIT}"
  fi
  [[ "$(git -C "${LMMS_DIR}" rev-parse HEAD)" == "${LMMS_COMMIT}" ]] || \
    die "lmms-eval failed to select exact base commit ${LMMS_COMMIT}"

  if git -C "${LMMS_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
    [[ -z "$(git -C "${LMMS_DIR}" status --porcelain --untracked-files=normal)" ]] || \
      die "lmms-eval has unexpected changes before patching"
    log "Applying WFS lmms-eval patch"
    git -C "${LMMS_DIR}" apply "${PATCH_FILE}"
  fi
  validate_exact_lmms_patch
  log "lmms-eval is at the exact base commit with the expected patch"
}

setup_lmms_eval

log "Installing patched lmms-eval in editable mode"
"${VENV_PYTHON}" -m pip install -e "${LMMS_DIR}"
log "Ensuring the official Hugging Face 'hf' CLI is installed"
"${VENV_PYTHON}" -m pip install --upgrade "huggingface_hub[cli]>=0.35,<1"
"${VENV_PYTHON}" -m pip check

authenticate_hugging_face() {
  local hf_cli="${VENV_DIR}/bin/hf"
  [[ -x "${hf_cli}" ]] || die "official Hugging Face 'hf' CLI is unavailable in the venv"

  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "Validating Hugging Face authentication from HF_TOKEN"
    if ! "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1
import os
from huggingface_hub import HfApi, login

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit(1)
login(token=token, add_to_git_credential=False)
os.environ.pop("HF_TOKEN", None)
HfApi().whoami()
PY
    then
      die "HF_TOKEN authentication failed"
    fi
    return
  fi

  if "${hf_cli}" auth whoami >/dev/null 2>&1; then
    log "Using existing official Hugging Face login"
    return
  fi
  [[ "${NON_INTERACTIVE}" == "0" && -t 0 && -t 1 ]] || \
    die "no Hugging Face login is available; set HF_TOKEN or rerun interactively"
  log "Starting interactive Hugging Face login"
  "${hf_cli}" auth login
  "${hf_cli}" auth whoami >/dev/null 2>&1 || die "interactive Hugging Face login failed"
}

authenticate_hugging_face

if [[ "${DATASET_CHECK}" == "full" ]]; then
  log "Checking every annotation-referenced raw video"
  "${VENV_PYTHON}" - \
    "${REQUESTED_DATASETS[*]}" \
    "${VIDEOMME_ROOT}" "${MLVU_ROOT}" "${LVB_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

selected = set(sys.argv[1].split())
roots = {
    "videomme": Path(sys.argv[2]),
    "mlvu": Path(sys.argv[3]),
    "lvb": Path(sys.argv[4]),
}
layouts = {
    "videomme": ("videomme_json_file.json", "data", lambda row: f"{row['videoID']}.mp4"),
    "mlvu": ("mlvu_dev.json", "video", lambda row: row["video_name"]),
    "lvb": ("lvb_val.json", "videos", lambda row: row["video_path"]),
}
for dataset in sorted(selected):
    annotation_name, raw_name, filename = layouts[dataset]
    root = roots[dataset]
    rows = json.loads((root / annotation_name).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{dataset}: annotation root is not a list")
    expected = {root / raw_name / str(filename(row)) for row in rows}
    missing = sorted(path for path in expected if not path.is_file())
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise SystemExit(
            f"{dataset}: {len(missing)}/{len(expected)} referenced videos are missing; {preview}"
        )
    print(f"dataset {dataset}: {len(expected)} referenced videos present")
PY
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on the A100 host"
log "NVIDIA driver-visible GPUs"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

log "Running dependency, CUDA, A100, and CLI smoke checks"
(
  cd -- "${REPO_DIR}"
  "${VENV_PYTHON}" - <<'PY'
import importlib
import platform

modules = (
    "av",
    "decord",
    "numpy",
    "PIL",
    "pywt",
    "qwen_vl_utils",
    "scipy",
    "sklearn",
    "torchaudio",
    "torchvision",
    "transformers",
    "yaml",
    "phase_stable",
    "lmms_eval",
)
for name in modules:
    importlib.import_module(name)

import torch
from phase_stable.cli import build_parser
from phase_stable.config import load_phase_stable_config

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA")
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
a100_indices = [index for index, name in enumerate(names) if "A100" in name.upper()]
if not a100_indices:
    raise SystemExit(f"no visible A100 GPU; visible devices={names}")
index = a100_indices[0]
torch.cuda.set_device(index)
left = torch.randn((512, 512), device=f"cuda:{index}", dtype=torch.float16)
right = torch.randn((512, 512), device=f"cuda:{index}", dtype=torch.float16)
result = left @ right
if not bool(torch.isfinite(result).all()):
    raise SystemExit("A100 tensor smoke produced non-finite values")
torch.cuda.synchronize(index)
properties = torch.cuda.get_device_properties(index)

parser = build_parser()
commands = set(parser._subparsers._group_actions[0].choices)
required = {
    "analyze-signals",
    "controlled-shifts",
    "evaluate-predictions",
    "export-keyframes",
    "make-benchmark-manifests",
    "matched-boundaries",
    "preprocess-benchmark",
    "selection-baselines",
}
if not required.issubset(commands):
    raise SystemExit(f"phase_stable CLI is missing commands: {sorted(required - commands)}")
load_phase_stable_config("configs/phase_stable_icassp.yaml")
print(f"python={platform.python_version()} torch={torch.__version__} cuda={torch.version.cuda}")
print(
    f"a100={properties.name} capability={properties.major}.{properties.minor} "
    f"memory_gib={properties.total_memory / 2**30:.1f}"
)
print("phase_stable dependency/CUDA smoke: OK")
PY
)

log "Bootstrap complete"
log "Repository: ${REPO_DIR}"
log "Virtualenv: ${VENV_DIR}"
log "Activate with: source '${VENV_DIR}/bin/activate'"
