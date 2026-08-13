from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from phase_stable.artifacts import OriginSignalRecord
from phase_stable.policy import qvhighlights_selection_fidelity
from phase_stable.qv_blind import load_blind_qv_query_videos
from phase_stable.rc12_experiment import build_decision_specs

REPO = Path(__file__).parents[1]


def load_script(name: str):
    path = REPO / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_rgb(index: int) -> str:
    rgb = np.full((1, 1, 3), index, dtype=np.uint8)
    return hashlib.sha256(memoryview(rgb).cast("B")).hexdigest()


@pytest.fixture(scope="module")
def decision_grid():
    records = []
    contracts = {}
    expected_items = set()
    origins = (0.1, 0.3, 0.5, 0.7, 0.9)
    for item_index in range(100):
        source = f"s{item_index:010d}"
        video_id = f"{source}_0.0_150.0"
        question_id = str(item_index)
        expected_items.add((video_id, question_id))
        contracts[video_id] = {
            "video_path": str((REPO / "fake" / f"{video_id}.mp4").resolve()),
            "duration_sec": 150.0,
            "support_start_sec": max(origins),
            "support_stop_sec": min(origin + 149 for origin in origins),
        }
        for origin_id, origin_sec in enumerate(origins):
            timestamps = tuple(origin_sec + index for index in range(150))
            relevance = np.zeros(150, dtype=float)
            relevance[21 + origin_id] = 10.0
            relevance[102 - origin_id] = 9.0
            records.append(
                OriginSignalRecord(
                    dataset="qvhighlights",
                    video_id=video_id,
                    question_id=question_id,
                    origin_id=origin_id,
                    origin_sec=origin_sec,
                    timestamps_sec=timestamps,
                    actual_pts_sec=timestamps,
                    source_frame_indices=tuple(range(150)),
                    relevance_scores=tuple(relevance),
                    metadata={
                        "feature_model": "blip2",
                        "feature_model_revision": "frozen",
                        "sample_fps": 1.0,
                        "query": f"query {item_index}",
                        "query_metadata": {
                            "annotation_row_index": item_index,
                            "source_id": source,
                            "label_firewall": "query_only_before_blind_seal",
                        },
                    },
                )
            )
    decisions = build_decision_specs(
        records,
        contracts,
        treatment_method="phasefuse_nested_r2",
    )
    return decisions, expected_items, contracts


def test_blind_loader_never_calls_standard_label_parser(tmp_path, monkeypatch):
    rows = []
    for index in range(100):
        source = f"s{index:010d}"
        rows.append(
            {
                "video_id": f"{source}_0.0_150.0",
                "query_id": str(index),
                "query": f"query {index}",
                "duration": 150.0,
                "metadata": {
                    "annotation_row_index": index,
                    "source_id": source,
                    "label_firewall": "query_only_before_blind_seal",
                },
            }
        )
    manifest = tmp_path / "query.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    from phase_stable import benchmarks

    monkeypatch.setattr(
        benchmarks,
        "_validate_qvhighlights_labels",
        lambda *args, **kwargs: pytest.fail("standard QV label parser was called"),
    )
    videos = load_blind_qv_query_videos(manifest, tmp_path)
    assert len(videos) == 100
    assert all(
        set(video.queries[0].metadata)
        == {"annotation_row_index", "source_id", "label_firewall"}
        for video in videos
    )


def test_blind_loader_rejects_schema_or_metadata_drift(tmp_path):
    source = "s0000000000"
    base = {
        "video_id": f"{source}_0.0_150.0",
        "query_id": "0",
        "query": "query",
        "duration": 150.0,
        "metadata": {
            "annotation_row_index": 0,
            "source_id": source,
            "label_firewall": "query_only_before_blind_seal",
        },
    }
    for mutation in (
        {**base, "relevant_windows": [[0, 2]]},
        {**base, "metadata": {**base["metadata"], "extra": 1}},
        {**base, "metadata": {**base["metadata"], "source_id": "wrong"}},
    ):
        rows = [mutation] + [
            {
                **base,
                "video_id": f"s{index:010d}_0.0_150.0",
                "query_id": str(index),
                "metadata": {
                    **base["metadata"],
                    "annotation_row_index": index,
                    "source_id": f"s{index:010d}",
                },
            }
            for index in range(1, 100)
        ]
        path = tmp_path / "bad.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError):
            load_blind_qv_query_videos(path, tmp_path)


def test_frozen_fidelity_is_exact_existing_metric_definition():
    analyzer = load_script("analyze_nested_r2_qv_stage0.py")
    label = {
        "relevant_windows": [[2.0, 4.0], [6.0, 8.0]],
        "relevant_clip_ids": [1, 3],
        "saliency_scores": [[4, 4, 4], [2, 2, 2]],
    }
    # Includes a half-open right boundary and duplicate selected clip IDs.
    actual = [2.0, 3.9, 4.0, 6.0, 6.1, 9.0]
    record = OriginSignalRecord(
        dataset="qvhighlights",
        video_id="source_0.0_10.0",
        question_id="q",
        origin_id=0,
        origin_sec=0.0,
        timestamps_sec=tuple(float(index) for index in range(len(actual))),
        actual_pts_sec=tuple(actual),
        source_frame_indices=tuple(range(len(actual))),
        relevance_scores=tuple(np.zeros(len(actual))),
        metadata={
            "query_metadata": {
                "relevant_windows_sec": label["relevant_windows"],
                "relevant_clip_ids": label["relevant_clip_ids"],
                "saliency_votes": label["saliency_scores"],
            }
        },
    )
    expected = qvhighlights_selection_fidelity(record, list(range(len(actual))))
    observed = analyzer.fidelity(actual, label)
    assert expected is not None
    for metric in observed:
        assert observed[metric] == pytest.approx(expected[metric])


def test_materializer_join_requires_every_prelabel_hash_binding(tmp_path):
    module = load_script("materialize_nested_r2_qv_blind.py")
    cohort = tmp_path / "cohort.tsv"
    labels = tmp_path / "labels.preseal.jsonl"
    query = tmp_path / "query.jsonl"
    materialization = tmp_path / "materialization.json"
    decisions = tmp_path / "decisions.jsonl"
    traces = tmp_path / "traces.jsonl"
    sources = tmp_path / "sources.json"
    blind_seal = tmp_path / "seal.json"
    output = tmp_path / "joined.jsonl"
    joined = tmp_path / "join.json"
    for path, content in ((decisions, "d\n"), (traces, "t\n"), (sources, "{}\n")):
        path.write_text(content)
    cohort_rows = []
    label_rows = []
    for index in range(100):
        source = f"s{index:010d}"
        vid = f"{source}_0.0_150.0"
        cohort_rows.append(f"{index}\t{source}\t{index}\t{vid}\t{index}\n")
        label_rows.append(
            {
                "source_id": source,
                "annotation_row_index": index,
                "vid": vid,
                "qid": str(index),
                "relevant_windows": [[0, 2]],
                "relevant_clip_ids": [0],
                "saliency_scores": [[1, 1, 1]],
            }
        )
    cohort.write_text("".join(cohort_rows))
    label_bytes = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in label_rows
    ).encode()
    labels.write_bytes(label_bytes)
    query.write_text("query\n")
    hashes = {
        "blind_materialization_sha256": module.sha256_file(materialization)
        if materialization.exists()
        else "pending",
        "query_manifest_sha256": module.sha256_file(query),
        "sealed_labels_sha256": module.sha256_file(labels),
    }
    materialization.write_text(
        json.dumps(
            {
                **hashes,
                "query_manifest_sha256": module.sha256_file(query),
                "sealed_labels_sha256": module.sha256_file(labels),
                "sealed_label_plaintext_sha256": module.sha256_file(labels),
            }
        )
    )
    seal = {
        "status": "sealed_before_label_join",
        "decisions_sha256": module.sha256_file(decisions),
        "traces_sha256": module.sha256_file(traces),
        "source_video_bundle_sha256": module.sha256_file(sources),
        "blind_materialization_sha256": module.sha256_file(materialization),
        "query_manifest_sha256": module.sha256_file(query),
        "sealed_labels_sha256": module.sha256_file(labels),
        "preregistration_sha256": module.EXPECTED_PREREG_SHA256,
        "cohort_sha256": module.EXPECTED_COHORT_SHA256,
    }
    blind_seal.write_text(json.dumps(seal))
    args = type(
        "Args",
        (),
        {
            "sealed_labels": labels,
            "blind_seal": blind_seal,
            "decisions": decisions,
            "traces": traces,
            "source_video_bundle": sources,
            "materialization_manifest": materialization,
            "query_manifest": query,
            "cohort": cohort,
            "output_labels": output,
            "join_manifest": joined,
        },
    )()
    monkeypatches = [
        ("EXPECTED_COHORT_SHA256", module.sha256_file(cohort)),
    ]
    originals = {name: getattr(module, name) for name, _ in monkeypatches}
    try:
        for name, value in monkeypatches:
            setattr(module, name, value)
        seal["cohort_sha256"] = module.EXPECTED_COHORT_SHA256
        blind_seal.write_text(json.dumps(seal))
        module.join(args)
        assert output.read_bytes() == label_bytes
        query.write_text("substituted\n")
        with pytest.raises(ValueError, match="query_manifest_sha256"):
            module.join(args)
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


def test_runner_is_qwen_free_and_hard_binds_frozen_contract():
    source = (REPO / "scripts" / "run_nested_r2_qv_stage0.sh").read_text()
    lower = source.lower()
    assert "lmms_eval" not in lower
    assert "run_mllm" not in lower
    assert "qwen" in lower  # explicit prohibition/help and runtime flag
    for token in (
        "NESTED_R2_QV_STAGE0",
        "20260810",
        "SAMPLE_FPS=1.0",
        "520bf73fd0ef6ccce791cd3f06908040aef2d8cc",
        "93f537621bfb693fac5a52760a5b83bd93eb3dd1885292e680b62bc81d962b1f",
        "/home/elicer/WFS-SB/.phasefuse_gpu.lock",
        "--expected-git-head",
    ):
        assert token in source
    bash = shutil.which("bash")
    if os.name == "nt":
        bash = r"C:\Program Files\Git\bin\bash.exe"
    if bash and Path(bash).is_file():
        subprocess.run(
            [bash, "-n", str(REPO / "scripts" / "run_nested_r2_qv_stage0.sh")],
            check=True,
        )


def test_query_validator_is_structural_not_substring_based(tmp_path):
    validator = load_script("validate_nested_r2_qv_stage0.py")
    rows = []
    for index in range(100):
        source = f"s{index:010d}"
        rows.append(
            {
                "video_id": f"{source}_0.0_150.0",
                "query_id": str(index),
                "query": "Explain the literal word relevant_windows safely",
                "duration": 150.0,
                "metadata": {
                    "annotation_row_index": index,
                    "source_id": source,
                    "label_firewall": "query_only_before_blind_seal",
                },
            }
        )
    path = tmp_path / "queries.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(validator._query_identity(path)) == 100
    rows[0]["metadata"]["relevant_windows"] = [[0, 2]]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        validator._query_identity(path)


def test_decision_audit_accepts_frozen_grid_and_rejects_corruption(decision_grid):
    validator = load_script("validate_nested_r2_qv_stage0.py")
    decisions, expected_items, _ = decision_grid
    audit = validator.audit_decisions(decisions, expected_items)
    assert audit["num_rows"] == 1000
    assert sum(audit["relocation_count_cells"].values()) == 500
    assert any(
        row["decision_metadata"].get("relocation_count", 0) > 0
        for row in decisions
        if row["method"] == "phasefuse_nested_r2"
    )

    bad = copy.deepcopy(decisions)
    treatment = next(row for row in bad if row["method"] == "phasefuse_nested_r2")
    treatment["quantized_scores"][0] = 0.15
    with pytest.raises(ValueError, match="quantized score"):
        validator.audit_decisions(bad, expected_items)

    bad = copy.deepcopy(decisions)
    treatment = next(row for row in bad if row["method"] == "phasefuse_nested_r2")
    treatment["video_path"] = str((REPO / "fake" / "substituted.mp4").resolve())
    with pytest.raises(ValueError, match="paired decision"):
        validator.audit_decisions(bad, expected_items)

    bad = copy.deepcopy(decisions)
    treatment = next(
        row
        for row in bad
        if row["method"] == "phasefuse_nested_r2"
        and row["decision_metadata"]["relocation_count"] > 0
    )
    treatment["decision_metadata"]["relocations"][0]["quantized_gain"] += 0.1
    with pytest.raises(ValueError, match="relocation provenance"):
        validator.audit_decisions(bad, expected_items)


def _trace_grid(decisions):
    by_cell = {}
    for row in decisions:
        key = (row["video_id"], row["question_id"], row["origin_id"])
        by_cell.setdefault(key, {})[row["method"]] = row
    traces = []
    for (_, _, _), arms in sorted(by_cell.items()):
        union_targets = sorted(
            set(arms["canonical_uniform"]["target_timestamps_sec"])
            | set(arms["phasefuse_nested_r2"]["target_timestamps_sec"])
        )
        union_indices = list(range(len(union_targets)))
        union_hashes = [_sha_rgb(index) for index in union_indices]
        provenance = [
            {
                "target_sec": target,
                "actual_pts_sec": target,
                "abs_error_sec": 0.0,
                "decode_error_ms": 0.0,
                "decoded_frame_index": index,
                "pixel_hash": union_hashes[index],
                "decode_pass_index": 0,
                "attempted_by": list(arms),
                "midpoint_tie_policy": "earlier_pts",
                "arm_attempts": [],
            }
            for index, target in enumerate(union_targets)
        ]
        for method, decision in arms.items():
            lattice = decision["lattice_timestamps_sec"]
            chosen = [
                union_targets.index(value)
                for value in decision["target_timestamps_sec"]
            ]
            attempts = []
            for attempt_index, target in enumerate(decision["target_timestamps_sec"]):
                union_index = union_targets.index(target)
                attempts.append(
                    {
                        "method": method,
                        "role": "canonical_uniform"
                        if method == "canonical_uniform"
                        else "nested_r2",
                        "attempt_index": attempt_index,
                        "stage": "primary",
                        "repair_round": 0,
                        "repair_source": None,
                        "target_source": "test",
                        "canonical_index": lattice.index(target),
                        "candidate_rank": attempt_index,
                        "candidate_score": 0.0,
                        "target_sec": target,
                        "actual_pts_sec": target,
                        "abs_error_sec": 0.0,
                        "decode_error_ms": 0.0,
                        "decoded_frame_index": union_index,
                        "pixel_hash": union_hashes[union_index],
                        "decode_pass_index": 0,
                        "midpoint_tie_policy": "earlier_pts",
                        "distance_relaxed": False,
                        "priority_nearest_distance_sec": None,
                        "priority_capped_distance_sec": None,
                        "status": "selected",
                        "duplicate_winner_target_sec": None,
                        "duplicate_resolution_stage": None,
                    }
                )
            trace = {
                "schema_version": 1,
                "dataset": "qvhighlights",
                "video_id": decision["video_id"],
                "question_id": decision["question_id"],
                "origin_id": decision["origin_id"],
                "origin_sec": decision["origin_sec"],
                "method": method,
                "timestamps_sec": union_targets,
                "actual_pts_sec": union_targets,
                "source_frame_indices": union_indices,
                "pixel_hashes": union_hashes,
                "selected_indices": chosen,
                "selected_timestamps_sec": [union_targets[index] for index in chosen],
                "selected_actual_pts_sec": [union_targets[index] for index in chosen],
                "selected_source_frame_indices": [
                    union_indices[index] for index in chosen
                ],
                "selected_pixel_hashes": [union_hashes[index] for index in chosen],
                "record_metadata": {},
                "candidate_provenance": copy.deepcopy(provenance),
                "canonical_decode": {
                    "schema_version": 1,
                    "fresh_source_decode": True,
                    "prohibit_scout_frame_remap": True,
                    "decoder_backend": "pyav_sequential_nearest_pts",
                    "midpoint_tie_policy": "earlier_pts",
                    "frame_budget": 16,
                    "candidate_union_size": len(union_targets),
                    "decode_pass_count": 1,
                    "decode_passes": [{"pass_index": 0}],
                    "repair_attempt_count": 0,
                    "duplicate_rejection_count": 0,
                    "initial_duplicate_rejection_count": 0,
                    "repair_duplicate_rejection_count": 0,
                    "duplicate_replacement_count": 0,
                    "distance_relaxation_count": 0,
                    "attempts": attempts,
                },
            }
            traces.append(trace)
    return traces


def test_trace_audit_binds_union_attempts_and_counts(decision_grid, monkeypatch):
    validator = load_script("validate_nested_r2_qv_stage0.py")
    decisions, _, contracts = decision_grid
    traces = _trace_grid(decisions)
    source_inventory = {
        video_id: {"video_path": contract["video_path"]}
        for video_id, contract in contracts.items()
    }

    class FakeFrame:
        def __init__(self, index):
            self.index = index

        def to_ndarray(self, *, format):
            assert format == "rgb24"
            return np.full((1, 1, 3), self.index, dtype=np.uint8)

    class FakeContainer:
        def __init__(self):
            self.streams = type("Streams", (), {"video": [object()]})()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def decode(self, stream):
            del stream
            return (FakeFrame(index) for index in range(32))

    monkeypatch.setattr(validator.av, "open", lambda path: FakeContainer())
    assert (
        validator.audit_traces(traces, decisions, source_inventory)["num_rows"] == 1000
    )

    bad = copy.deepcopy(traces)
    bad[0]["selected_pixel_hashes"][0] = "0" * 64
    with pytest.raises(ValueError, match="selected/candidate-union"):
        validator.audit_traces(bad, decisions, source_inventory)

    bad = copy.deepcopy(traces)
    bad[0]["canonical_decode"]["attempts"][0]["canonical_index"] += 1
    with pytest.raises(ValueError, match="attempt lattice"):
        validator.audit_traces(bad, decisions, source_inventory)

    bad = copy.deepcopy(traces)
    bad[0]["canonical_decode"]["repair_attempt_count"] = 1
    with pytest.raises(ValueError, match="count provenance"):
        validator.audit_traces(bad, decisions, source_inventory)

    bad = copy.deepcopy(traces)
    key = (
        bad[0]["video_id"],
        bad[0]["question_id"],
        bad[0]["origin_id"],
    )
    treatment = next(
        row
        for row in bad
        if (row["video_id"], row["question_id"], row["origin_id"]) == key
        and row["method"] == "phasefuse_nested_r2"
    )
    treatment["candidate_provenance"][0]["arm_attempts"].append({"tampered": True})
    with pytest.raises(ValueError, match="candidate union"):
        validator.audit_traces(bad, decisions, source_inventory)


def test_source_checkpoint_and_runtime_recompute_current_bytes(tmp_path, monkeypatch):
    validator = load_script("validate_nested_r2_qv_stage0.py")

    videos_root = (tmp_path / "videos").resolve()
    videos_root.mkdir()
    members = []
    for index in range(100):
        source = f"s{index:010d}"
        video_id = f"{source}_0.0_150.0"
        path = videos_root / f"{video_id}.mp4"
        path.write_bytes(f"video-{index}".encode())
        members.append(
            {
                "video_id": video_id,
                "question_id": str(index),
                "source_id": source,
                "video_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": validator.sha256_file(path),
            }
        )
    inventory_path = tmp_path / "inventory.json"
    inventory = {
        "schema_version": 1,
        "status": "frozen_qv_source_inventory",
        "videos_root": str(videos_root),
        "num_videos": 100,
        "videos": members,
    }
    inventory_path.write_text(json.dumps(inventory))
    monkeypatch.setattr(validator, "EXPECTED_VIDEOS_ROOT", videos_root)
    assert len(validator._source_inventory(inventory_path)) == 100
    members[0]["video_path"] = str(videos_root / "wrong.mp4")
    inventory_path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="escaped frozen videos root"):
        validator._source_inventory(inventory_path)

    snapshot = (tmp_path / "snapshot").resolve()
    snapshot.mkdir()
    (snapshot / "a").write_bytes(b"alpha")
    (snapshot / "b").write_bytes(b"beta")
    snapshot_rows = validator._snapshot_rows(snapshot)
    tree = validator._canonical_tree_sha256(snapshot_rows)
    checkpoint = {
        "schema_version": 1,
        "algorithm": "sha256(canonical-json(sorted[{path,size_bytes,sha256}]))",
        "snapshot_path": str(snapshot),
        "revision": snapshot.name,
        "content_tree_sha256": tree,
        "num_files": len(snapshot_rows),
        "total_size_bytes": sum(row["size_bytes"] for row in snapshot_rows),
        "files": snapshot_rows,
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(json.dumps(checkpoint))
    monkeypatch.setattr(validator, "EXPECTED_CHECKPOINT_ROOT", snapshot)
    monkeypatch.setattr(validator, "EXPECTED_CHECKPOINT_REVISION", snapshot.name)
    monkeypatch.setattr(validator, "EXPECTED_CHECKPOINT_TREE_SHA256", tree)
    monkeypatch.setattr(validator, "EXPECTED_CHECKPOINT_FILE_COUNT", 2)
    monkeypatch.setattr(
        validator,
        "EXPECTED_CHECKPOINT_TOTAL_BYTES",
        sum(row["size_bytes"] for row in snapshot_rows),
    )
    validator._audit_checkpoint_preflight(checkpoint_path)
    (snapshot / "a").write_bytes(b"ALPHA")
    with pytest.raises(ValueError, match="member path/size/content"):
        validator._audit_checkpoint_preflight(checkpoint_path)

    repo = (tmp_path / "repo").resolve()
    (repo / "phase_stable").mkdir(parents=True)
    (repo / "scripts").mkdir()
    for relative, content in (
        ("phase_stable/a.py", "A=1\n"),
        ("scripts/check_nested_r2_qv.py", "B=2\n"),
        ("scripts/decode_rc12_exact.py", "C=3\n"),
        ("scripts/run_nested_r2_qv_stage0.sh", "#!/bin/sh\n"),
    ):
        (repo / relative).write_text(content)
    rows = validator._current_runtime_code_rows(repo)
    head = "a" * 40
    packages = {}
    for name in validator.RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    runtime = {
        "schema_version": 1,
        "git_head": head,
        "git_status_clean": True,
        "code_tree_sha256": hashlib.sha256(
            json.dumps(rows, separators=(",", ":")).encode()
        ).hexdigest(),
        "code_files": rows,
        "python": validator.sys.version,
        "python_executable_raw": validator.sys.executable,
        "python_executable_resolved": str(Path(validator.sys.executable).resolve()),
        "packages": packages,
        "checkpoint_preflight_sha256": validator.sha256_file(checkpoint_path),
        "qwen_commands_present": False,
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime))
    monkeypatch.setattr(
        validator.subprocess,
        "check_output",
        lambda command, **kwargs: f"{head}\n" if "rev-parse" in command else "",
    )
    validator._audit_runtime_manifest(runtime_path, checkpoint_path, head, repo=repo)
    (repo / "scripts/decode_rc12_exact.py").write_text("C=4\n")
    with pytest.raises(ValueError, match="runtime manifest"):
        validator._audit_runtime_manifest(
            runtime_path, checkpoint_path, head, repo=repo
        )


def test_gpu_pid_parser_and_strict_hierarchical_gate():
    preflight = load_script("preflight_nested_r2_qv_stage0.py")
    analyzer = load_script("analyze_nested_r2_qv_stage0.py")
    assert (
        preflight.numeric_gpu_pids(
            ["[Insufficient Permissions], [N/A], [N/A]", "not-a-pid, python, 1 MiB"]
        )
        == []
    )
    assert preflight.numeric_gpu_pids([" 1234, python, 1024 MiB", "5678,x,2"]) == [
        1234,
        5678,
    ]

    fail, endpoints = analyzer.ordered_endpoint_gates(
        {
            "selected_relevant_fraction": {"ci_low": 0.0},
            "relevant_clip_recall": {"ci_low": 1.0},
        }
    )
    assert fail is False
    assert endpoints["relevant_clip_recall"]["hierarchically_tested"] is False
    assert endpoints["relevant_clip_recall"]["passed_if_tested"] is None
    passed, endpoints = analyzer.ordered_endpoint_gates(
        {
            "selected_relevant_fraction": {"ci_low": 1e-12},
            "relevant_clip_recall": {"ci_low": -1e-12},
        }
    )
    assert passed is True
    assert endpoints["relevant_clip_recall"]["hierarchically_tested"] is True
    assert endpoints["relevant_clip_recall"]["passed_if_tested"] is False
    assert analyzer.BOOTSTRAP_RESAMPLES == 10_000
    assert analyzer.BOOTSTRAP_SEED == 20260813
