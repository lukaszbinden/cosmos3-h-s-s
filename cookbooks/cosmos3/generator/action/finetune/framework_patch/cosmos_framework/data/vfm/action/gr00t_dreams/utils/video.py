# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import av
import cv2
import numpy as np

import torch  # noqa: F401 # isort: skip
import torchvision  # noqa: F401 # isort: skip


# ``decord`` is a fast video decoder but ships no aarch64 wheels (and no
# cp313 wheels even on x86_64), so the cosmos3 train extras can't safely
# depend on it.  We keep the codepath for opt-in users who install decord
# manually, but the import is deferred so this module loads cleanly when
# decord isn't present.  Default video_backend across this module is
# ``torchvision_av`` (pyav under the hood; ``av`` IS in cosmos3 train
# extras), matching the runtime fallback already used in dataset.py.
def _import_decord():
    try:
        import decord  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - opt-in path
        raise ImportError(
            "video_backend='decord' was requested but the ``decord`` package "
            "is not installed.  Install it manually (``uv pip install decord``) "
            "or switch to video_backend='torchvision_av' / 'pyav' / 'opencv'."
        ) from e
    return decord


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "torchvision_av",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    if video_backend == "decord":
        decord = _import_decord()
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        frames = []
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    else:
        raise NotImplementedError


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "torchvision_av",
    video_backend_kwargs: dict = {},
) -> np.ndarray:
    """Get frames from a video at specified timestamps.
    Args:
        video_path (str): Path to the video file.
        timestamps (list[int] | np.ndarray): Timestamps to retrieve frames for, in seconds.
        video_backend (str, optional): Video backend to use. Defaults to "torchvision_av"
            (pyav-based, bundled with cosmos3 train extras).  "decord" is also
            supported but requires manual installation (no aarch64 / cp313 wheels).
    Returns:
        np.ndarray: Frames at the specified timestamps.
    """
    if video_backend == "decord":
        decord = _import_decord()
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        num_frames = len(vr)
        # Retrieve the timestamps for each frame in the video
        frame_ts: np.ndarray = vr.get_frame_timestamp(range(num_frames))
        # Map each requested timestamp to the closest frame index
        # Only take the first element of the frame_ts array which corresponds to start_seconds
        indices = np.abs(frame_ts[:, :1] - timestamps).argmin(axis=0)
        frames = vr.get_batch(indices)
        return frames.asnumpy()
    elif video_backend == "opencv":
        # Open the video file
        cap = cv2.VideoCapture(video_path, **video_backend_kwargs)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        # Retrieve the total number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate timestamps for each frame
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_ts = np.arange(num_frames) / fps
        frame_ts = frame_ts[:, np.newaxis]  # Reshape to (num_frames, 1) for broadcasting
        # Map each requested timestamp to the closest frame index
        indices = np.abs(frame_ts - timestamps).argmin(axis=0)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Unable to read frame at index {idx}")
            frames.append(frame)
        cap.release()
        frames = np.array(frames)
        return frames
    elif video_backend in ("torchvision_av", "pyav_single"):
        # Single-threaded pyav decode (replaces torchvision.io.VideoReader).
        #
        # WHY NOT torchvision.io.VideoReader: it opens the libav decoder with
        # ffmpeg's default *multithreaded* settings. On a contended login node
        # with a low per-user thread/pids ceiling (RLIMIT_NPROC / cgroup
        # pids.max), ffmpeg's ``avcodec_open2`` can't spawn its worker threads
        # and fails with ``BlockingIOError(11, 'Resource temporarily
        # unavailable', 'avcodec_open2(h264)')`` (EAGAIN) — on the very first
        # frame, for h264 1080p CMR video. Forcing single-threaded decode
        # (``thread_type="NONE"``, ``thread_count=1``) makes ``avcodec_open2``
        # allocate no worker threads, sidestepping the limit entirely. It's a
        # touch slower per clip but robust; for stats we only sample a few
        # thousand short clips so the cost is negligible.
        #
        # We also ALWAYS close the container in a finally block: the reader
        # holds a file descriptor + decoder context, and the previous
        # torchvision code only closed it on the success path, so any decode
        # exception leaked an fd (compounding under the retry loop into
        # ``[Errno 24] Too many open files``).
        container = av.open(video_path)
        loaded_frames = []
        loaded_ts = []
        try:
            stream = container.streams.video[0]
            # Force single-threaded decode (the actual EAGAIN fix).
            stream.thread_type = "NONE"
            try:
                stream.codec_context.thread_count = 1
                stream.codec_context.thread_type = "NONE"
            except Exception:  # noqa: BLE001
                pass  # older/newer pyav: thread_type on the stream is enough

            time_base = stream.time_base
            first_ts = float(timestamps[0])
            last_ts = float(timestamps[-1])

            # Seek to the closest keyframe at/before first_ts. ``seek`` takes a
            # time in stream time_base units; ``any_frame=False`` -> keyframe.
            if time_base is not None:
                seek_target = int(first_ts / float(time_base))
                container.seek(seek_target, stream=stream, any_frame=False, backward=True)

            read_past_last = False
            for frame in container.decode(stream):
                # frame.pts is in time_base units; convert to seconds.
                current_ts = float(frame.pts * time_base) if (frame.pts is not None and time_base is not None) else 0.0
                # HWC uint8 RGB to match the torchvision path's final layout.
                arr = frame.to_ndarray(format="rgb24")
                loaded_frames.append(arr)
                loaded_ts.append(current_ts)
                if read_past_last:
                    break
                if current_ts >= last_ts:
                    read_past_last = True
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

        if len(loaded_frames) == 0:
            raise ValueError(
                f"No frames loaded from {video_path} for timestamps {timestamps[0]:.3f} to {timestamps[-1]:.3f}"
            )

        # Match requested timestamps to closest loaded frames (like decord/opencv backends do)
        loaded_ts = np.array(loaded_ts).reshape(-1, 1)  # (num_loaded, 1)
        requested_ts = np.array(timestamps)  # (num_requested,)
        indices = np.abs(loaded_ts - requested_ts).argmin(axis=0)
        # loaded_frames are already HWC rgb24; stack to (T, H, W, C).
        frames = np.array([loaded_frames[i] for i in indices])
        return frames
    else:
        raise NotImplementedError


def get_all_frames(
    video_path: str,
    video_backend: str = "torchvision_av",
    video_backend_kwargs: dict = {},
    resize_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Get all frames from a video.
    Args:
        video_path (str): Path to the video file.
        video_backend (str, optional): Video backend to use. Defaults to "torchvision_av"
            (pyav-based, bundled with cosmos3 train extras).  "decord" is also
            supported but requires manual installation.
        video_backend_kwargs (dict, optional): Keyword arguments for the video backend.
        resize_size (tuple[int, int], optional): Resize size for the frames. Defaults to None.
    """
    if video_backend == "decord":
        decord = _import_decord()
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
    elif video_backend == "pyav":
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.to_ndarray(format="rgb24")
            frames.append(frame)
        frames = np.array(frames)
    elif video_backend == "torchvision_av":
        # set backend and reader
        torchvision.set_video_backend("pyav")
        reader = torchvision.io.VideoReader(video_path, "video")
        frames = []
        for frame in reader:
            frames.append(frame["data"])
        frames = np.array(frames)
        frames = frames.transpose(0, 2, 3, 1)
    else:
        raise NotImplementedError(f"Video backend {video_backend} not implemented")
    # resize frames if specified
    if resize_size is not None:
        frames = [cv2.resize(frame, resize_size) for frame in frames]
        frames = np.array(frames)
    return frames


def get_all_frames_and_timestamps(
    video_path: str,
    video_backend: str = "torchvision_av",
    video_backend_kwargs: dict = {},
) -> tuple[np.ndarray, np.ndarray]:
    """Get all frames from a video.

    Defaults to ``torchvision_av`` (pyav-based, bundled with cosmos3 train
    extras).  ``decord`` is supported but requires manual installation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frames and timestamps.
    """
    if video_backend == "decord":
        decord = _import_decord()
        vr = decord.VideoReader(video_path, **video_backend_kwargs)
        frames = vr.get_batch(range(len(vr))).asnumpy()
        return frames, vr.get_frame_timestamp(range(len(vr)))[:, 0]

    elif video_backend == "pyav":
        container = av.open(video_path)
        stream = container.streams.video[0]
        assert stream.time_base is not None
        frames = []
        timestamps = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(frame.pts * stream.time_base)
        container.close()
        return np.stack(frames), np.array(timestamps)

    else:
        raise NotImplementedError
