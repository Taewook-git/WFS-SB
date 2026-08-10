from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from phase_stable import sampling


def test_stratified_origins_are_deterministic_video_specific_and_in_strata():
    origins = sampling.generate_stratified_origins(
        2026, "video-한글", 5, period_sec=1.0
    )

    assert origins == sampling.generate_stratified_origins(
        2026, "video-한글", 5, period_sec=1.0
    )
    assert origins != sampling.generate_stratified_origins(
        2026, "different-video", 5, period_sec=1.0
    )
    assert all(index / 5 <= value < (index + 1) / 5 for index, value in enumerate(origins))
    assert all(
        0.0 <= sampling.deterministic_hash_uniform(2026, "video-한글", index) < 1.0
        for index in range(5)
    )


def test_common_candidate_count_and_timestamp_grids_follow_protocol_formula():
    origins = (0.05, 0.31, 0.92)

    count = sampling.common_candidate_count(
        3.0, origins, sample_fps=1.0, epsilon_sec=1e-9
    )
    grids = sampling.common_target_timestamps(
        3.0, origins, sample_fps=1.0, epsilon_sec=1e-9
    )

    assert count == 3
    assert grids == (
        (0.05, 1.05, 2.05),
        (0.31, 1.31, 2.31),
        (0.92, 1.92, 2.92),
    )
    assert {len(grid) for grid in grids} == {count}
    assert sampling.common_candidate_count(0.1, (0.2, 0.8)) == 0
    assert sampling.common_target_timestamps(0.1, (0.2, 0.8)) == ((), ())


def test_sampling_manifest_json_and_jsonl_round_trip(tmp_path):
    first = sampling.build_sampling_manifest(
        "비디오-1", 5.2, master_seed=19, num_origins=5, sample_fps=1.0
    )
    second = sampling.build_sampling_manifest(
        "video-2", 2.6, master_seed=19, num_origins=3, sample_fps=2.0
    )

    assert sampling.manifest_from_json(sampling.manifest_to_json(first)) == first

    json_path = tmp_path / "nested" / "offset.json"
    sampling.write_manifest_json(json_path, first)
    assert sampling.read_manifest_json(json_path) == first

    jsonl_path = tmp_path / "offsets.jsonl"
    sampling.write_manifests_jsonl(jsonl_path, (first, second))
    assert sampling.read_manifests_jsonl(jsonl_path) == [first, second]
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: sampling.generate_stratified_origins(0, "", 5), ValueError),
        (lambda: sampling.generate_stratified_origins(0, "v", 0), ValueError),
        (lambda: sampling.generate_stratified_origins(0, "v", 5, period_sec=0), ValueError),
        (lambda: sampling.common_candidate_count(-1, (0.0,)), ValueError),
        (lambda: sampling.common_candidate_count(1, ()), ValueError),
        (lambda: sampling.common_target_timestamps(1, (float("nan"),)), ValueError),
        (lambda: sampling.build_sampling_manifest("v", 1, sample_fps=0), ValueError),
    ],
)
def test_sampling_input_validation(call, error):
    with pytest.raises(error):
        call()


def test_manifest_rejects_tampered_candidate_count():
    manifest = sampling.build_sampling_manifest("v", 4.0, num_origins=3)
    payload = manifest.to_dict()
    payload["candidate_count"] += 1

    with pytest.raises(ValueError, match="candidate_count"):
        sampling.SamplingManifest.from_dict(payload)


def test_nearest_pts_indices_preserves_order_and_breaks_ties_earlier():
    assert sampling.nearest_pts_indices(
        [0.0, 2.0, 4.0], [3.0, 1.0, 4.9, 0.1]
    ) == [1, 0, 2, 0]
    assert sampling.nearest_pts_indices([], []) == []

    with pytest.raises(ValueError, match="nondecreasing"):
        sampling.nearest_pts_indices([0.0, 2.0, 1.0], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        sampling.nearest_pts_indices([], [1.0])


def test_import_does_not_require_pyav_and_decode_error_is_clear(monkeypatch, tmp_path):
    fake_video = tmp_path / "video.mp4"
    fake_video.touch()
    original_import_module = sampling.importlib.import_module

    def missing_pyav(name):
        if name == "av":
            raise ModuleNotFoundError("No module named 'av'")
        return original_import_module(name)

    monkeypatch.setattr(sampling.importlib, "import_module", missing_pyav)
    with pytest.raises(RuntimeError, match="PyAV is required.*pip install av"):
        sampling.decode_nearest_frames_pyav(fake_video, [0.0])


class _FakeFrame:
    def __init__(self, pts: int, value: int):
        self.pts = pts
        self.time_base = Fraction(1, 1)
        self._value = value

    def to_ndarray(self, *, format: str):
        assert format == "rgb24"
        return np.full((2, 3, 3), self._value, dtype=np.uint8)


class _FakeContainer:
    def __init__(self):
        stream = SimpleNamespace(time_base=Fraction(1, 1))
        self.streams = SimpleNamespace(video=[stream])
        self.frames = [_FakeFrame(0, 10), _FakeFrame(2, 20), _FakeFrame(4, 40)]
        self.decode_calls = 0
        self.closed = False

    def decode(self, stream):
        assert stream is self.streams.video[0]
        self.decode_calls += 1
        yield from self.frames

    def close(self):
        self.closed = True


def test_pyav_decoder_uses_one_sequential_pass_and_earlier_tie_break(monkeypatch, tmp_path):
    fake_video = tmp_path / "video.mp4"
    fake_video.touch()
    container = _FakeContainer()
    fake_av = SimpleNamespace(open=lambda path: container)
    monkeypatch.setattr(sampling, "_import_pyav", lambda: fake_av)

    matches = sampling.decode_nearest_frames_pyav(
        fake_video, [3.0, 1.0, 5.0, 0.0]
    )

    assert container.decode_calls == 1
    assert container.closed
    assert [match.target_timestamp_sec for match in matches] == [3.0, 1.0, 5.0, 0.0]
    assert [match.actual_pts_sec for match in matches] == [2.0, 0.0, 4.0, 0.0]
    assert [match.decode_error_ms for match in matches] == [1000.0] * 3 + [0.0]
    assert [int(match.rgb[0, 0, 0]) for match in matches] == [20, 10, 40, 10]
    assert all(len(match.pixel_hash) == 64 for match in matches)


def test_decode_manifest_flattens_all_origins_into_one_decode_call(monkeypatch):
    manifest = sampling.build_sampling_manifest(
        "v", 3.2, master_seed=7, num_origins=3
    )
    seen = []

    def fake_decode(video_path, targets, *, stream_index=0):
        seen.append((video_path, list(targets), stream_index))
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        return [
            sampling.DecodedFrameMatch(target, target, 0.0, "0" * 64, rgb)
            for target in targets
        ]

    monkeypatch.setattr(sampling, "decode_nearest_frames_pyav", fake_decode)
    decoded = sampling.decode_manifest_pyav("unused.mp4", manifest, stream_index=2)

    assert len(seen) == 1
    assert seen[0][2] == 2
    assert list(decoded) == [0, 1, 2]
    assert {len(matches) for matches in decoded.values()} == {manifest.candidate_count}
