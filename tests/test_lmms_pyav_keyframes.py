from __future__ import annotations

import importlib.util
import sys
import types
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODIFIED_ROOT = REPO_ROOT / "lmms-eval-diff" / "modified_files" / "lmms_eval"
HELPER = (
    MODIFIED_ROOT
    / "models"
    / "model_utils"
    / "qwen2_5_vl_keyframe_vision_process.py"
)
PATCH = REPO_ROOT / "lmms-eval-diff" / "lmms_eval_wfs.patch"


def _load_helper(monkeypatch: pytest.MonkeyPatch):
    package = types.ModuleType("qwen_vl_utils")
    vision_process = types.ModuleType("qwen_vl_utils.vision_process")
    for name in ("ceil_by_factor", "extract_vision_info", "fetch_image", "smart_resize"):
        setattr(vision_process, name, lambda *args, **kwargs: None)
    package.vision_process = vision_process
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", package)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils.vision_process", vision_process)

    spec = importlib.util.spec_from_file_location("wfs_qwen_keyframe_helper", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, frame_count: int = 5) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(
                np.full((64, 64, 3), index * 32, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_pyav_helper_returns_exact_requested_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper(monkeypatch)
    video_path = tmp_path / "source.mp4"
    _write_video(video_path)

    with av.open(str(video_path)) as container:
        decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    requested = [3, 0, 3]
    expected = torch.from_numpy(np.stack([decoded[index] for index in requested])).permute(
        0, 3, 1, 2
    )

    video, metadata, sample_fps = module._read_video_pyav_keyframe(
        {"video": str(video_path)}, requested
    )

    assert video.dtype == torch.uint8
    assert video.is_contiguous()
    assert torch.equal(video, expected)
    assert metadata["video_backend"] == "pyav"
    assert metadata["frames_indices"] == requested
    assert metadata["total_num_frames"] == 5
    assert sample_fps > 0


def test_pyav_helper_rejects_out_of_range_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_helper(monkeypatch)
    video_path = tmp_path / "source.mp4"
    _write_video(video_path)

    with pytest.raises(IndexError, match="missing=\\[5\\], total_num_frames=5"):
        module._read_video_pyav_keyframe({"video": str(video_path)}, [5])


def test_pyav_helper_supports_frames_without_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_helper(monkeypatch)

    class FrameWithoutDuration:
        def __init__(self, index: int) -> None:
            self.pts = index
            self.time_base = Fraction(1, 5)
            self.index = index

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "rgb24"
            return np.full((8, 8, 3), self.index, dtype=np.uint8)

    stream = types.SimpleNamespace(
        duration=2,
        time_base=Fraction(1, 5),
        average_rate=Fraction(5, 1),
    )

    class FakeContainer:
        streams = types.SimpleNamespace(video=[stream])
        duration = 400_000

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def decode(self, selected_stream):
            assert selected_stream is stream
            return iter([FrameWithoutDuration(0), FrameWithoutDuration(1)])

    fake_av = types.SimpleNamespace(open=lambda _: FakeContainer(), time_base=1_000_000)
    monkeypatch.setitem(sys.modules, "av", fake_av)

    video, metadata, sample_fps = module._read_video_pyav_keyframe(
        {"video": "synthetic.mp4"}, [0, 1]
    )

    assert tuple(video.shape) == (2, 3, 8, 8)
    assert metadata["total_num_frames"] == 2
    assert sample_fps == pytest.approx(5.0)


def test_qwen_paths_preserve_keyframes_without_decord_resampling() -> None:
    chat = (MODIFIED_ROOT / "models" / "chat" / "qwen2_5_vl.py").read_text(
        encoding="utf-8"
    )
    simple = (MODIFIED_ROOT / "models" / "simple" / "qwen2_5_vl.py").read_text(
        encoding="utf-8"
    )
    protocol = (MODIFIED_ROOT / "protocol.py").read_text(encoding="utf-8")

    for source in (chat, simple):
        assert "process_vision_info_keyframe" in source
        assert "return_video_kwargs=True" in source
        assert "**keyframe_video_kwargs" in source
        assert "if video_inputs is not None and not self.use_keyframe" in source
    assert "import decord" not in simple
    assert "from decord" not in protocol
    assert 'video_kwargs = {}' in HELPER.read_text(encoding="utf-8")
    assert "do_sample_frames" not in HELPER.read_text(encoding="utf-8")


def test_distributed_patch_contains_the_verified_pyav_path() -> None:
    patch = PATCH.read_text(encoding="utf-8")

    assert "+def _read_video_pyav_keyframe(" in patch
    assert '+        "video_backend": "pyav",' in patch
    assert patch.count("+                    return_video_kwargs=True,") == 2
    assert "diff --git a/lmms_eval/protocol.py b/lmms_eval/protocol.py" in patch
    assert "-from decord import VideoReader, cpu" in patch
