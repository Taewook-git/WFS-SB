from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phase_stable import preprocess
from phase_stable.artifacts import read_signal_records, write_signal_records
from phase_stable.sampling import DecodedFrameMatch, build_sampling_manifest


def _decoded_for(manifest):
    decoded = {}
    source_index = 10
    for origin in manifest.origins:
        matches = []
        for target in origin.target_timestamps_sec:
            actual = target + 0.01
            rgb = np.full(
                (2, 3, 3), origin.origin_id * 50 + source_index, dtype=np.uint8
            )
            matches.append(
                DecodedFrameMatch(
                    target_timestamp_sec=target,
                    actual_pts_sec=actual,
                    decode_error_ms=10.0,
                    pixel_hash=f"{source_index:064x}",
                    rgb=rgb,
                    decoded_frame_index=source_index,
                )
            )
            source_index += 2
        decoded[origin.origin_id] = tuple(matches)
    return decoded


class _Extractor:
    def __init__(self):
        self.calls = []

    def compute(self, frames, query, batch_size):
        pixel_values = np.asarray([frame[0, 0, 0] for frame in frames], dtype=float)
        self.calls.append((tuple(id(frame) for frame in frames), query, batch_size))
        score_bias = 0.1 if query == "first query" else 0.2
        scores = pixel_values / 255.0 + score_bias
        features = np.stack((pixel_values, pixel_values + 1.0), axis=1)
        return scores, features


def test_preprocess_decodes_once_reuses_frames_and_returns_jsonl_ready_records(
    monkeypatch, tmp_path: Path
):
    manifest = build_sampling_manifest(
        "video-1", 3.4, master_seed=17, num_origins=2
    )
    decoded = _decoded_for(manifest)
    decode_calls = []

    def fake_decode(video_path, supplied_manifest, *, stream_index=0):
        decode_calls.append((video_path, supplied_manifest, stream_index))
        return decoded

    monkeypatch.setattr(preprocess, "decode_manifest_pyav", fake_decode)
    extractor = _Extractor()
    records = preprocess.preprocess_video_manifest(
        tmp_path / "video.mp4",
        manifest,
        dataset="demo",
        queries=[
            preprocess.QuerySpec("q1", "first query", {"task": "test"}),
            {"question_id": "q2", "query": "second query"},
        ],
        extractor=extractor,
        output_dir=tmp_path / "artifacts",
        batch_size=4,
        stream_index=1,
    )

    assert len(decode_calls) == 1
    assert decode_calls[0][1] is manifest
    assert decode_calls[0][2] == 1
    assert len(extractor.calls) == 4
    assert len(records) == 4
    assert [(record.question_id, record.origin_id) for record in records] == [
        ("q1", 0),
        ("q1", 1),
        ("q2", 0),
        ("q2", 1),
    ]

    # The exact same decoded arrays are reused for the second query.
    assert extractor.calls[0][0] == extractor.calls[2][0]
    assert extractor.calls[1][0] == extractor.calls[3][0]
    first_record = records[0]
    first_matches = decoded[0]
    assert first_record.timestamps_sec == manifest.origins[0].target_timestamps_sec
    assert first_record.actual_pts_sec == tuple(match.actual_pts_sec for match in first_matches)
    assert first_record.source_frame_indices == tuple(
        match.decoded_frame_index for match in first_matches
    )
    assert first_record.pixel_hashes == tuple(match.pixel_hash for match in first_matches)
    assert first_record.metadata["decoded_frame_indices"] == list(
        first_record.source_frame_indices
    )
    assert first_record.metadata["source_pts_sec"] == list(first_record.actual_pts_sec)
    assert first_record.metadata["decode_error_ms"] == [10.0] * manifest.candidate_count

    feature_paths = [Path(record.visual_features_path) for record in records]
    assert len(set(feature_paths)) == 4
    assert all(path.is_file() and path.suffix == ".npy" for path in feature_paths)
    features = np.load(feature_paths[0], allow_pickle=False)
    assert features.shape == (manifest.candidate_count, 2)
    assert first_record.metadata["feature_shape"] == list(features.shape)

    jsonl_path = tmp_path / "signals.jsonl"
    write_signal_records(jsonl_path, records)
    assert read_signal_records(jsonl_path) == records


@pytest.mark.parametrize("bad_output", ["scores", (np.ones(3),)])
def test_preprocess_requires_exact_score_feature_pair(
    monkeypatch, tmp_path: Path, bad_output
):
    manifest = build_sampling_manifest("v", 3.2, num_origins=1)
    monkeypatch.setattr(
        preprocess, "decode_manifest_pyav", lambda *args, **kwargs: _decoded_for(manifest)
    )

    class BadExtractor:
        def compute(self, frames, query, batch_size):
            return bad_output

    with pytest.raises(TypeError, match=r"exactly \(scores, features\)"):
        preprocess.preprocess_video_manifest(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=[preprocess.QuerySpec("q", "query")],
            extractor=BadExtractor(),
            output_dir=tmp_path,
        )


@pytest.mark.parametrize("mismatch", ["scores", "features"])
def test_preprocess_strictly_validates_extractor_frame_alignment(
    monkeypatch, tmp_path: Path, mismatch: str
):
    manifest = build_sampling_manifest("v", 3.2, num_origins=1)
    monkeypatch.setattr(
        preprocess, "decode_manifest_pyav", lambda *args, **kwargs: _decoded_for(manifest)
    )

    class BadExtractor:
        def compute(self, frames, query, batch_size):
            length = len(frames)
            scores = np.ones(length - (mismatch == "scores"), dtype=float)
            features = np.ones((length - (mismatch == "features"), 4), dtype=float)
            return scores, features

    with pytest.raises(ValueError, match=f"{mismatch[:-1] if mismatch.endswith('s') else mismatch} length mismatch"):
        preprocess.preprocess_video_manifest(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=[preprocess.QuerySpec("q", "query")],
            extractor=BadExtractor(),
            output_dir=tmp_path,
        )


def test_preprocess_accepts_tensor_like_outputs_without_importing_model_framework(
    monkeypatch, tmp_path: Path
):
    manifest = build_sampling_manifest("v", 2.2, num_origins=1)
    monkeypatch.setattr(
        preprocess, "decode_manifest_pyav", lambda *args, **kwargs: _decoded_for(manifest)
    )

    class TensorLike:
        def __init__(self, array):
            self.array = np.asarray(array)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

    class Extractor:
        def compute(self, frames, query, batch_size):
            count = len(frames)
            return TensorLike(np.arange(count)), TensorLike(np.ones((count, 2)))

    records = preprocess.preprocess_video_manifest(
        tmp_path / "video.mp4",
        manifest,
        dataset="demo",
        queries=[preprocess.QuerySpec("q", "query")],
        extractor=Extractor(),
        output_dir=tmp_path,
    )
    assert len(records) == 1
    assert records[0].relevance_scores == tuple(
        float(index) for index in range(manifest.candidate_count)
    )


def test_preprocess_rejects_missing_decoded_provenance(monkeypatch, tmp_path: Path):
    manifest = build_sampling_manifest("v", 2.2, num_origins=1)
    decoded = _decoded_for(manifest)
    first = decoded[0][0]
    decoded[0] = (
        DecodedFrameMatch(
            first.target_timestamp_sec,
            first.actual_pts_sec,
            first.decode_error_ms,
            first.pixel_hash,
            first.rgb,
        ),
        *decoded[0][1:],
    )
    monkeypatch.setattr(preprocess, "decode_manifest_pyav", lambda *args, **kwargs: decoded)

    with pytest.raises(ValueError, match="decoded_frame_index is missing"):
        preprocess.preprocess_video_manifest(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=[preprocess.QuerySpec("q", "query")],
            extractor=_Extractor(),
            output_dir=tmp_path,
        )


def test_query_and_extractor_validation(tmp_path: Path):
    manifest = build_sampling_manifest("v", 2.2, num_origins=1)
    with pytest.raises(ValueError, match="unique"):
        preprocess.preprocess_video_manifest(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=[
                preprocess.QuerySpec("q", "one"),
                preprocess.QuerySpec("q", "two"),
            ],
            extractor=_Extractor(),
            output_dir=tmp_path,
        )
    with pytest.raises(TypeError, match="extractor must provide"):
        preprocess.preprocess_video_manifest(
            tmp_path / "video.mp4",
            manifest,
            dataset="demo",
            queries=[preprocess.QuerySpec("q", "query")],
            extractor=object(),
            output_dir=tmp_path,
        )


def test_streaming_preprocess_bounds_rgb_buffer_and_writes_memmaps(
    monkeypatch, tmp_path: Path
):
    manifest = build_sampling_manifest("stream-video", 5.2, num_origins=2)
    decoded = _decoded_for(manifest)
    indexed = []
    flat_index = 0
    for origin in manifest.origins:
        for match in decoded[origin.origin_id]:
            indexed.append((match.target_timestamp_sec, flat_index, match))
            flat_index += 1
    indexed.sort(key=lambda value: value[0])
    decode_calls = []

    def fake_iterator(video_path, targets, *, stream_index=0):
        decode_calls.append((video_path, tuple(targets), stream_index))
        for _, target_index, match in indexed:
            yield target_index, match

    monkeypatch.setattr(
        preprocess, "iter_nearest_frames_pyav_indexed", fake_iterator
    )
    extractor = _Extractor()
    records = preprocess.preprocess_video_manifest_streaming(
        tmp_path / "video.mp4",
        manifest,
        dataset="demo",
        queries=[
            preprocess.QuerySpec("q1", "first query"),
            preprocess.QuerySpec("q2", "second query"),
        ],
        extractor=extractor,
        output_dir=tmp_path / "streaming",
        batch_size=16,
        frame_buffer_size=3,
    )

    assert len(decode_calls) == 1
    assert len(records) == 4
    assert max(len(call[0]) for call in extractor.calls) <= 3
    assert all(record.metadata["streaming_decode"] for record in records)
    assert all(record.metadata["frame_buffer_size"] == 3 for record in records)
    for record in records:
        features = np.load(record.visual_features_path, allow_pickle=False)
        assert features.shape == (manifest.candidate_count, 2)
        assert np.all(np.isfinite(features))
