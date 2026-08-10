from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from phase_stable.cli import build_parser


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_stage0.sh"


def _bash() -> str:
    if os.name == "nt":
        # Prefer Git Bash over the WindowsApps WSL launcher returned by which().
        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    else:
        candidates = [shutil.which("bash")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            probe = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if "GNU bash" in probe.stdout:
                return candidate
    pytest.skip("GNU bash is unavailable")


def _posix_arg(path: Path) -> str:
    # ``C:/...`` is understood by both Git Bash and native Windows Python.
    return path.resolve().as_posix()


def test_stage0_script_has_valid_bash_syntax_and_documents_interface() -> None:
    bash = _bash()
    subprocess.run([bash, "-n", _posix_arg(SCRIPT)], check=True)
    result = subprocess.run(
        [bash, _posix_arg(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--full" in result.stdout
    assert "--force-step" in result.stdout
    assert "BATCH_SIZE=32" in result.stdout
    assert "FRAME_BUFFER_SIZE=256" in result.stdout
    assert "fingerprint match" in result.stdout


def test_stage0_script_only_invokes_registered_phase_stable_commands() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    invoked = {
        "make-benchmark-manifests",
        "preprocess-benchmark",
        "analyze-signals",
        "matched-boundaries",
        "selection-baselines",
        "export-keyframes",
    }
    parser = build_parser()
    subparser_action = next(
        action for action in parser._actions if action.dest == "command"
    )
    registered = set(subparser_action.choices)

    assert invoked <= registered
    assert all(f"-m phase_stable {command}" in source for command in invoked)
    assert "convert-predictions" not in source
    assert 'MANIFEST_CMD+=(--video-indices "${VIDEO_INDEX_ARRAY[@]}")' in source
    assert 'if [[ "${FULL_RUN}" == 0 ]]' in source


def _write_fake_python(tmp_path: Path) -> tuple[Path, Path]:
    call_log = tmp_path / "phase_calls.jsonl"
    wrapper_python = tmp_path / "fake_python_driver.py"
    wrapper_python.write_text(
        """
import json
import os
import subprocess
import sys
from pathlib import Path

REAL = os.environ["REAL_PYTHON"]
args = sys.argv[1:]
if args[:2] == ["-m", "phase_stable"] and len(args) >= 3:
    command = args[2]
    if command != "--help":
        with Path(os.environ["PHASE_CALL_LOG"]).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(args, ensure_ascii=False) + "\\n")
    if command == "preprocess-benchmark":
        import numpy as np
        from phase_stable.artifacts import OriginSignalRecord, write_signal_records
        from phase_stable.sampling import read_manifests_jsonl

        def option(name):
            return args[args.index(name) + 1]

        manifests = read_manifests_jsonl(option("--manifests"))
        signal_path = Path(option("--signal-jsonl"))
        output_dir = Path(option("--output-dir"))
        records = []
        for manifest in manifests:
            for origin in manifest.origins:
                timestamps = np.asarray(origin.target_timestamps_sec, dtype=float)
                scores = 0.5 + 0.25 * np.sin(timestamps / 3.0 + origin.origin_id * 0.1)
                features = np.stack((np.sin(timestamps), np.cos(timestamps)), axis=1)
                feature_path = output_dir / "features" / f"{manifest.video_id}_o{origin.origin_id}.npy"
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(feature_path, features, allow_pickle=False)
                records.append(
                    OriginSignalRecord(
                        dataset="mlvu",
                        video_id=manifest.video_id,
                        question_id="Q1",
                        origin_id=origin.origin_id,
                        origin_sec=origin.origin_sec,
                        timestamps_sec=origin.target_timestamps_sec,
                        actual_pts_sec=origin.target_timestamps_sec,
                        source_frame_indices=tuple(range(manifest.candidate_count)),
                        relevance_scores=tuple(float(value) for value in scores),
                        pixel_hashes=tuple("a" * 64 for _ in timestamps),
                        visual_features_path=str(feature_path.resolve()),
                    )
                )
        write_signal_records(signal_path, records)
        manifest_dir = output_dir / "manifest"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "run_manifest.json").write_text("{}\\n", encoding="utf-8")
        (manifest_dir / "environment.json").write_text("{}\\n", encoding="utf-8")
        print(f"fake preprocess wrote {len(records)} records")
        raise SystemExit(0)
raise SystemExit(subprocess.call([REAL, *args]))
""".lstrip(),
        encoding="utf-8",
    )

    launcher = tmp_path / "fake-python"
    real_python = Path(sys.executable).resolve().as_posix()
    driver = wrapper_python.resolve().as_posix()
    launcher.write_text(
        f'#!/usr/bin/env bash\nexec "{real_python}" "{driver}" "$@"\n',
        encoding="utf-8",
        newline="\n",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return launcher, call_log


def _phase_commands(call_log: Path) -> list[str]:
    return [
        json.loads(line)[2]
        for line in call_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_stage0_resume_requires_matching_marker_and_output_checksum(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Linux runner integration is prohibitively slow through Git Bash")
    bash = _bash()
    dataset_root = tmp_path / "mlvu"
    (dataset_root / "video").mkdir(parents=True)
    (dataset_root / "video" / "clip.mp4").write_bytes(b"raw-video-placeholder")
    questions = dataset_root / "questions.json"
    questions.write_text(
        json.dumps(
            [
                {
                    "video_name": "clip.mp4",
                    "question_id": "Q1",
                    "question": "What happened?",
                    "candidates": ["one", "two"],
                    "answer": "A",
                    "duration": 40.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        (REPO_ROOT / "configs" / "phase_stable_icassp.yaml")
        .read_text(encoding="utf-8")
        .replace("frame_budget: 16", "frame_budget: 2"),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    fake_python, call_log = _write_fake_python(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "REAL_PYTHON": sys.executable,
            "PHASE_CALL_LOG": str(call_log),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    command = [
        bash,
        _posix_arg(SCRIPT),
        "--benchmark",
        "mlvu",
        "--questions-file",
        _posix_arg(questions),
        "--dataset-root",
        _posix_arg(dataset_root),
        "--run-dir",
        _posix_arg(run_dir),
        "--config",
        _posix_arg(config),
        "--video-indices",
        "0",
        "--seed",
        "7",
        "--num-origins",
        "2",
        "--sample-fps",
        "1",
        "--frame-budget",
        "2",
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--frame-buffer-size",
        "4",
        "--bootstrap",
        "10",
        "--matched-count",
        "2",
        "--python",
        _posix_arg(fake_python),
    ]

    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=120)
    first_commands = _phase_commands(call_log)
    assert first_commands == [
        "make-benchmark-manifests",
        "preprocess-benchmark",
        "analyze-signals",
        "matched-boundaries",
        "selection-baselines",
        "export-keyframes",
        "export-keyframes",
    ]

    # Identical invocation must trust matching markers rather than mere files.
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=120)
    assert _phase_commands(call_log) == first_commands

    # A present but modified intermediate invalidates its producing marker.
    signals = run_dir / "origin_signals.jsonl"
    signals.write_text(signals.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, timeout=120)
    assert _phase_commands(call_log) == [*first_commands, "preprocess-benchmark"]

    # Forcing one step does not needlessly invalidate an unchanged downstream export.
    subprocess.run(
        [*command, "--force-step", "baselines"],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        timeout=120,
    )
    assert _phase_commands(call_log)[-1] == "selection-baselines"

    markers = run_dir / ".stage0_state"
    marker = json.loads((markers / "preprocess.done.json").read_text(encoding="utf-8"))
    assert len(marker["fingerprint"]) == 64
    assert len(marker["config_hash"]) == 64
    assert any(key.startswith("signal-bundle::") for key in marker["outputs"])
    assert (run_dir / "stage0_config.json").is_file()
    assert (run_dir / "manifest_preflight.json").is_file()
