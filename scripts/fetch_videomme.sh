#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Download and extract only the VideoMME archive chunks needed by this run.

Usage:
  bash scripts/fetch_videomme.sh [options]

Options:
  --dataset-root PATH    VideoMME root (default: datasets/videomme)
  --questions-file PATH  Annotation JSON (default: <root>/videomme_json_file.json)
  --video-count N        First N unique videos; 0 means all (default: 20)
  --cache-dir PATH       Download cache (default: <root>/.archives)
  --repo-id ID           Hugging Face dataset repo (default: lmms-eval/Video-MME)
  --keep-archives        Do not remove each zip after successful extraction
  --dry-run              Print downloads without changing files
  -h, --help             Show this help

The script is resumable. It checks the exact videoID values in the local
annotation and stops as soon as every requested MP4 exists.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="$repo_root/datasets/videomme"
questions_file=""
video_count=20
cache_dir=""
repo_id="lmms-eval/Video-MME"
keep_archives=0
dry_run=0

while (($#)); do
  case "$1" in
    --dataset-root) dataset_root="$2"; shift 2 ;;
    --questions-file) questions_file="$2"; shift 2 ;;
    --video-count) video_count="$2"; shift 2 ;;
    --cache-dir) cache_dir="$2"; shift 2 ;;
    --repo-id) repo_id="$2"; shift 2 ;;
    --keep-archives) keep_archives=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$video_count" =~ ^[0-9]+$ ]] || {
  printf '%s\n' '--video-count must be a non-negative integer.' >&2
  exit 2
}

dataset_root="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$dataset_root")"
questions_file="${questions_file:-$dataset_root/videomme_json_file.json}"
cache_dir="${cache_dir:-$dataset_root/.archives}"
data_dir="$dataset_root/data"

for command in python hf; do
  command -v "$command" >/dev/null 2>&1 || {
    printf 'Missing command: %s\n' "$command" >&2
    exit 1
  }
done
[[ -f "$questions_file" ]] || {
  printf 'Annotation file not found: %s\n' "$questions_file" >&2
  exit 1
}

mapfile -t required_ids < <(
  python - "$questions_file" "$video_count" <<'PY'
import json
import sys

path, raw_limit = sys.argv[1], sys.argv[2]
limit = int(raw_limit)
with open(path, encoding="utf-8") as handle:
    rows = json.load(handle)
seen = set()
for row in rows:
    video_id = str(row["videoID"]).strip()
    if video_id and video_id not in seen:
        seen.add(video_id)
        print(video_id)
        if limit and len(seen) >= limit:
            break
if not seen:
    raise SystemExit("annotation contains no videoID values")
if limit and len(seen) != limit:
    raise SystemExit(f"requested {limit} videos, annotation contains {len(seen)}")
PY
)

missing_ids=()
refresh_missing() {
  missing_ids=()
  local video_id
  for video_id in "${required_ids[@]}"; do
    [[ -s "$data_dir/$video_id.mp4" ]] || missing_ids+=("$video_id")
  done
}

refresh_missing
if ((${#missing_ids[@]} == 0)); then
  printf 'VideoMME data already complete: %d/%d videos in %s\n' \
    "${#required_ids[@]}" "${#required_ids[@]}" "$data_dir"
  exit 0
fi

printf 'Need %d of %d requested VideoMME videos.\n' \
  "${#missing_ids[@]}" "${#required_ids[@]}"
if ((dry_run)); then
  printf 'Would scan/download archives videos_chunked_01.zip ... videos_chunked_20.zip from %s\n' "$repo_id"
  exit 0
fi

mkdir -p "$data_dir" "$cache_dir"
for chunk_number in $(seq -w 1 20); do
  archive_name="videos_chunked_${chunk_number}.zip"
  archive_path="$cache_dir/$archive_name"
  before=${#missing_ids[@]}
  printf '[VideoMME] chunk %s: %d videos still missing\n' "$chunk_number" "$before"
  if [[ ! -s "$archive_path" ]]; then
    hf download "$repo_id" "$archive_name" \
      --repo-type dataset \
      --local-dir "$cache_dir"
  fi
  python - "$archive_path" "$data_dir" <<'PY'
import shutil
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with zipfile.ZipFile(archive) as handle:
    members = [
        member
        for member in handle.infolist()
        if not member.is_dir() and Path(member.filename).suffix.lower() == ".mp4"
    ]
    if not members:
        raise SystemExit(f"archive contains no MP4 files: {archive}")
    for member in members:
        output = destination / Path(member.filename).name
        if output.is_file() and output.stat().st_size > 0:
            continue
        partial = output.with_suffix(output.suffix + ".part")
        try:
            with handle.open(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
PY
  refresh_missing
  after=${#missing_ids[@]}
  printf '[VideoMME] chunk %s supplied %d requested videos\n' \
    "$chunk_number" "$((before - after))"
  if ((keep_archives == 0)); then
    rm -f -- "$archive_path"
  fi
  if ((after == 0)); then
    break
  fi
done

refresh_missing
if ((${#missing_ids[@]})); then
  printf 'Missing %d requested videos after all archives. First missing IDs:\n' \
    "${#missing_ids[@]}" >&2
  printf '  %s\n' "${missing_ids[@]:0:20}" >&2
  exit 1
fi

printf 'VideoMME ready: %d videos in %s\n' "${#required_ids[@]}" "$data_dir"
