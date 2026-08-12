from __future__ import annotations

from pathlib import Path

import numpy as np

from phase_stable import phasefuse_preprocess
from phase_stable.multiphase import build_multiphase_manifest, split_dense_record
from phase_stable.preprocess import QuerySpec
from phase_stable.sampling import DecodedFrameMatch


class _Extractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], int]] = []

    def compute(self, frames, query, batch_size):
        pixels = np.asarray([int(frame[0, 0, 0]) for frame in frames], dtype=float)
        self.calls.append((query, tuple(int(value) for value in pixels), batch_size))
        bias = 0.1 if query == "first" else 0.2
        scores = pixels / 255.0 + bias
        features = np.stack((pixels, pixels + 1.0), axis=1).astype(np.float32)
        return scores, features


def _fake_matches(targets):
    order = sorted(enumerate(targets), key=lambda item: (item[1], item[0]))
    for flat_index, target in order:
        source_index = int(np.floor(float(target) * 2.0 + 1e-12))
        actual = source_index / 2.0
        rgb = np.full((2, 2, 3), source_index, dtype=np.uint8)
        yield flat_index, DecodedFrameMatch(
            target_timestamp_sec=float(target),
            actual_pts_sec=actual,
            decode_error_ms=1000.0 * abs(float(target) - actual),
            pixel_hash=f"{source_index:064x}",
            rgb=rgb,
            decoded_frame_index=source_index,
        )


def test_multiphase_preprocess_decodes_once_deduplicates_and_writes_dense_features(
    monkeypatch, tmp_path: Path
):
    manifest = build_multiphase_manifest(
        "video-1",
        5.0,
        base_sample_fps=1.0,
        num_phases=2,
        num_outer_origins=2,
        outer_origins_sec=(0.0, 0.2),
    )
    decode_calls = []

    def fake_iter(video_path, targets, *, stream_index=0):
        decode_calls.append((Path(video_path), tuple(targets), stream_index))
        yield from _fake_matches(targets)

    monkeypatch.setattr(
        phasefuse_preprocess,
        "iter_nearest_frames_pyav_indexed",
        fake_iter,
    )
    extractor = _Extractor()
    records = phasefuse_preprocess.preprocess_multiphase_video_streaming(
        tmp_path / "source.mp4",
        manifest,
        dataset="demo",
        queries=(
            QuerySpec("q1", "first", {"gold": "A"}),
            QuerySpec("q2", "second"),
        ),
        extractor=extractor,
        output_dir=tmp_path / "artifacts",
        batch_size=7,
        frame_buffer_size=1,
        stream_index=1,
        record_metadata={"feature_model": "fake"},
    )

    assert len(decode_calls) == 1
    assert decode_calls[0][2] == 1
    assert len(decode_calls[0][1]) == (
        manifest.num_outer_origins * manifest.dense_candidate_count
    )
    # Every physical source frame is scored once per query, despite overlapping
    # outer-origin targets, repeated dense targets, and a one-frame nominal
    # buffer that would split a repeated-source group without the guard.
    first_pixels = tuple(
        pixel
        for query, pixels, _ in extractor.calls
        if query == "first"
        for pixel in pixels
    )
    second_pixels = tuple(
        pixel
        for query, pixels, _ in extractor.calls
        if query == "second"
        for pixel in pixels
    )
    assert len(first_pixels) == len(set(first_pixels))
    assert first_pixels == second_pixels

    assert [(row.question_id, row.origin_id) for row in records] == [
        ("q1", 0),
        ("q1", 1),
        ("q2", 0),
        ("q2", 1),
    ]
    for record in records:
        assert len(record.timestamps_sec) == manifest.dense_candidate_count
        assert record.metadata["sample_fps"] == manifest.dense_sample_fps
        assert record.metadata["multiphase_decode_passes"] == 1
        assert record.metadata["feature_layout"] == "dense_target_aligned"
        feature_path = Path(record.visual_features_path)
        features = np.load(feature_path, allow_pickle=False)
        assert features.shape == (manifest.dense_candidate_count, 2)
        assert record.metadata["feature_shape"] == list(features.shape)
        for left, right in zip(record.source_frame_indices, record.source_frame_indices[1:]):
            if left == right:
                indices = np.flatnonzero(np.asarray(record.source_frame_indices) == left)
                assert np.all(features[indices] == features[indices[0]])
        streams = split_dense_record(record, manifest)
        assert len(streams) == manifest.num_phases
        assert all(
            len(stream.dense_indices) == manifest.candidate_count_per_phase
            for stream in streams
        )


def test_multiphase_preprocess_rejects_incomplete_decode(monkeypatch, tmp_path: Path):
    manifest = build_multiphase_manifest(
        "v",
        4.0,
        num_phases=2,
        num_outer_origins=1,
        outer_origins_sec=(0.0,),
    )

    def incomplete(*args, **kwargs):
        matches = list(_fake_matches(args[1]))
        yield from matches[:-1]

    monkeypatch.setattr(
        phasefuse_preprocess,
        "iter_nearest_frames_pyav_indexed",
        incomplete,
    )
    try:
        phasefuse_preprocess.preprocess_multiphase_video_streaming(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=(QuerySpec("q", "query"),),
            extractor=_Extractor(),
            output_dir=tmp_path,
        )
    except RuntimeError as exc:
        assert "expected" in str(exc)
    else:  # pragma: no cover - explicit failure message is clearer than assert False
        raise AssertionError("incomplete decode must fail")
