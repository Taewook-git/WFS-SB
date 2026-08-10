import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from qwen_vl_utils.vision_process import (
    ceil_by_factor,
    extract_vision_info,
    fetch_image,
    smart_resize,
)


SPATIAL_MERGE_SIZE = 2
VIDEO_MIN_TOKEN_NUM = 128
VIDEO_MAX_TOKEN_NUM = 768
FRAME_FACTOR = 2
MAX_NUM_WORKERS_FETCH_VIDEO = 8

MODEL_SEQ_LEN = int(float(os.environ.get("MODEL_SEQ_LEN", 128000)))
logger = logging.getLogger(__name__)


def _validate_frame_indices(frame_idx: Sequence[int]) -> List[int]:
    if frame_idx is None:
        raise ValueError("frame_idx is required for keyframe video decoding")

    if isinstance(frame_idx, torch.Tensor):
        if frame_idx.ndim != 1:
            raise ValueError("frame_idx must be a one-dimensional sequence")
        frame_idx = frame_idx.tolist()

    if isinstance(frame_idx, (str, bytes)):
        raise TypeError("frame_idx must be a sequence of non-negative integers")

    try:
        values = list(frame_idx)
    except TypeError as exc:
        raise TypeError("frame_idx must be a sequence of non-negative integers") from exc

    if not values:
        raise ValueError("frame_idx must contain at least one frame index")

    indices = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("frame_idx must contain only non-negative integers")
        index = int(value)
        if index < 0:
            raise ValueError("frame_idx must contain only non-negative integers")
        indices.append(index)
    return indices


def _positive_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _read_video_pyav_keyframe(
    ele: Dict[str, Any], frame_idx: Sequence[int]
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """Decode exact presentation-order frame indices with PyAV.

    The complete video is decoded so that both index validation and
    ``total_num_frames`` use the same presentation-order frame sequence as the
    requested indices.  Only requested frames are converted to RGB arrays.
    """

    import av

    indices = _validate_frame_indices(frame_idx)
    wanted = set(indices)
    video_path = ele["video"]
    if video_path.startswith("file://"):
        video_path = video_path[7:]

    started_at = time.time()
    selected_frames = {}
    selected_timestamps = {}
    decoded_timestamps = []
    last_frame_end = None

    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"video contains no video stream: {video_path}")
        stream = container.streams.video[0]

        stream_duration = None
        if stream.duration is not None and stream.time_base is not None:
            stream_duration = _positive_float(stream.duration * stream.time_base)
        container_duration = None
        if container.duration is not None:
            container_duration = _positive_float(container.duration / av.time_base)

        video_fps = _positive_float(stream.average_rate)
        total_frames = 0
        first_timestamp = None

        for source_index, frame in enumerate(container.decode(stream)):
            total_frames = source_index + 1
            timestamp = None
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
                if math.isfinite(timestamp):
                    decoded_timestamps.append(timestamp)
                    if first_timestamp is None:
                        first_timestamp = timestamp
                else:
                    timestamp = None

            if timestamp is not None:
                frame_duration = None
                frame_duration_value = getattr(frame, "duration", None)
                if frame_duration_value is not None and frame.time_base is not None:
                    frame_duration = _positive_float(
                        frame_duration_value * frame.time_base
                    )
                last_frame_end = timestamp + (frame_duration or 0.0)

            if source_index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise RuntimeError(
                        "PyAV returned an invalid rgb24 frame: "
                        f"index={source_index}, dtype={rgb.dtype}, shape={rgb.shape}"
                    )
                selected_frames[source_index] = (
                    torch.from_numpy(np.ascontiguousarray(rgb))
                    .permute(2, 0, 1)
                    .contiguous()
                )
                selected_timestamps[source_index] = timestamp

    if total_frames == 0:
        raise ValueError(f"video contains no decodable frames: {video_path}")

    missing = sorted(wanted.difference(selected_frames))
    if missing:
        raise IndexError(
            f"keyframe indices out of range for {video_path}: "
            f"missing={missing}, total_num_frames={total_frames}"
        )

    duration = stream_duration or container_duration
    if duration is None and first_timestamp is not None and last_frame_end is not None:
        duration = _positive_float(last_frame_end - first_timestamp)
    if duration is None and len(decoded_timestamps) > 1:
        timestamp_span = decoded_timestamps[-1] - decoded_timestamps[0]
        if timestamp_span > 0:
            duration = _positive_float(
                timestamp_span * total_frames / (total_frames - 1)
            )
    if duration is None and video_fps is not None:
        duration = _positive_float(total_frames / video_fps)

    if video_fps is None and len(decoded_timestamps) > 1:
        timestamp_span = decoded_timestamps[-1] - decoded_timestamps[0]
        if timestamp_span > 0:
            video_fps = _positive_float((total_frames - 1) / timestamp_span)
    if video_fps is None and duration is not None:
        video_fps = _positive_float(total_frames / duration)
    if video_fps is None or duration is None:
        raise RuntimeError(f"could not determine video timing metadata: {video_path}")

    video = torch.stack([selected_frames[index] for index in indices])
    if video.dtype != torch.uint8 or video.ndim != 4 or video.shape[1] != 3:
        raise RuntimeError(
            "PyAV keyframe decode did not produce a TCHW RGB uint8 tensor: "
            f"dtype={video.dtype}, shape={tuple(video.shape)}"
        )

    sample_fps = len(indices) / duration
    height, width = int(video.shape[2]), int(video.shape[3])
    video_metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "duration": duration,
        "video_backend": "pyav",
        "height": height,
        "width": width,
    }
    logger.info(
        "pyav: video_path=%r, total_frames=%d, video_fps=%.6f, "
        "duration=%.6f, frame_indices=%s, frame_timestamps=%s, time=%.3fs",
        video_path,
        total_frames,
        video_fps,
        duration,
        indices,
        [selected_timestamps[index] for index in indices],
        time.time() - started_at,
    )
    return video, video_metadata, sample_fps


def fetch_video_keyframe(
    ele: Dict[str, Any],
    image_patch_size: int = 14,
    return_video_sample_fps: bool = False,
    return_video_metadata: bool = False,
    frame_idx: Optional[Sequence[int]] = None,
) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
    video_frame_min_pixels = VIDEO_MIN_TOKEN_NUM * image_factor * image_factor
    video_frame_max_pixels = VIDEO_MAX_TOKEN_NUM * image_factor * image_factor
    if isinstance(ele["video"], str):
        video, video_metadata, sample_fps = _read_video_pyav_keyframe(
            ele, frame_idx=frame_idx
        )
    else:
        # The input is a list of frames.
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        max_workers = min(MAX_NUM_WORKERS_FETCH_VIDEO, len(ele["video"]))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    fetch_image,
                    {"image": video_element, **process_info},
                    image_factor,
                )
                for video_element in ele["video"]
            ]
            image_list = [future.result() for future in futures]

        nframes = ceil_by_factor(len(image_list), FRAME_FACTOR)
        if len(image_list) < nframes:
            image_list.extend([image_list[-1]] * (nframes - len(image_list)))

        sample_fps = ele.get("sample_fps", 2.0)
        video = torch.stack(
            [
                torch.from_numpy(np.array(image).transpose(2, 0, 1))
                for image in image_list
            ]
        )

        raw_fps = process_info.pop("raw_fps", sample_fps)
        video_metadata = {
            "fps": raw_fps,
            "frames_indices": list(range(len(video))),
            "total_num_frames": (nframes / sample_fps) * raw_fps,
        }

    nframes, _, height, width = video.shape
    min_pixels = ele.get("min_pixels", video_frame_min_pixels)
    total_pixels = ele.get(
        "total_pixels", MODEL_SEQ_LEN * image_factor * image_factor * 0.9
    )
    max_pixels = max(
        min(video_frame_max_pixels, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels_supposed = ele.get("max_pixels", max_pixels)
    if max_pixels_supposed > max_pixels:
        logger.warning(
            "The given max_pixels[%s] exceeds limit[%s].",
            max_pixels_supposed,
            max_pixels,
        )
    max_pixels = min(max_pixels_supposed, max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()

    final_video = (video, video_metadata) if return_video_metadata else video
    if return_video_sample_fps:
        return final_video, sample_fps
    return final_video


def _indices_for_videos(
    frame_idx: Sequence[int], number_of_videos: int
) -> List[Sequence[int]]:
    if number_of_videos == 0:
        return []
    if frame_idx is None:
        raise ValueError("frame_idx is required for keyframe video decoding")

    values = list(frame_idx)
    if number_of_videos == 1 and (
        not values or isinstance(values[0], Integral)
    ):
        return [values]
    if len(values) != number_of_videos:
        raise ValueError(
            "frame_idx must provide one index sequence per video: "
            f"got {len(values)} sequences for {number_of_videos} videos"
        )
    return values


def process_vision_info_keyframe(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
    frame_idx: Optional[Sequence[int]] = None,
) -> Tuple[
    Optional[List[Image.Image]],
    Optional[List[Union[torch.Tensor, List[Image.Image]]]],
    Optional[Dict[str, Any]],
]:
    vision_infos = extract_vision_info(conversations)
    number_of_videos = sum("video" in info for info in vision_infos)
    frame_indices = _indices_for_videos(frame_idx, number_of_videos)

    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    video_index = 0
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(
                fetch_image(vision_info, image_patch_size=image_patch_size)
            )
        elif "video" in vision_info:
            video_input, video_sample_fps = fetch_video_keyframe(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
                frame_idx=frame_indices[video_index],
            )
            video_index += 1
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url or video should in content.")
    if not image_inputs:
        image_inputs = None
    if not video_inputs:
        video_inputs = None

    video_kwargs = {"do_sample_frames": False}
    if not return_video_metadata:  # Backward compatibility for Qwen2.5-VL.
        video_kwargs["fps"] = video_sample_fps_list

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs
