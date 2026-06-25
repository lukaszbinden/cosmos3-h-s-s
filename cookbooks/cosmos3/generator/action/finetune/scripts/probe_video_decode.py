#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Probe a single Open-H video decode — fast check before a full stats/training run.

Reproduces the dataset's video read for ONE clip so you can confirm whether the
decoder works on THIS node, without waiting for the whole pipeline. Decodes the
same file three ways and reports which succeed / how long they take:

  1. multithreaded pyav  (ffmpeg default; the mode torchvision.io.VideoReader
     used — this is the one that throws ``BlockingIOError(11, ...,
     'avcodec_open2(h264)')`` / EAGAIN on a contended login node)
  2. single-threaded pyav (``thread_type="NONE"``; the fix in utils/video.py)
  3. the overlay's get_frames_by_timestamps(..., video_backend="torchvision_av")
     (which now routes through the single-threaded path)

If (1) fails but (2)/(3) succeed -> the EAGAIN was a thread-spawn limit and the
single-threaded fix resolves it. If ALL fail -> the node is too contended (run
on a compute node) or the file/codec is genuinely broken.

It auto-discovers a video for a given dataset leaf (first chunk, first camera).

Usage::

    # probe a CMR leaf (1080p h264 — the one that failed):
    python scripts/probe_video_decode.py \
        --dataset-path "$OPENH_SURGICAL_ROOT/cmr_surgical/cholecystectomy"

    # probe srth (AV1):
    python scripts/probe_video_decode.py \
        --dataset-path "$OPENH_SURGICAL_ROOT/jhu/imerse/srth_porcine_chole"

    # probe an explicit file:
    python scripts/probe_video_decode.py --video /path/to/episode_000000.mp4
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

# Match the stats script's thread caps so the probe reflects the real run.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")


def _find_a_video(dataset_path: Path) -> Path | None:
    vids = dataset_path / "videos"
    if not vids.exists():
        return None
    # videos/chunk-*/observation.images.<cam>/episode_*.mp4 (or .mkv)
    for ext in ("*.mp4", "*.mkv", "*.avi", "*.webm"):
        hits = sorted(vids.glob(f"**/{ext}"))
        if hits:
            return hits[0]
    return None


def _probe_multithreaded(path: str) -> tuple[bool, str]:
    import av

    t = time.time()
    try:
        container = av.open(path)
        try:
            stream = container.streams.video[0]
            # ffmpeg default: multithreaded (this is what torchvision did)
            stream.thread_type = "AUTO"
            n = 0
            for _frame in container.decode(stream):
                n += 1
                if n >= 13:
                    break
        finally:
            container.close()
        return True, f"decoded {n} frames in {time.time() - t:.2f}s (codec={stream.codec_context.name})"
    except Exception as e:  # noqa: BLE001
        return False, f"{e!r}  (after {time.time() - t:.2f}s)"


def _probe_single_threaded(path: str) -> tuple[bool, str]:
    import av

    t = time.time()
    try:
        container = av.open(path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "NONE"
            try:
                stream.codec_context.thread_count = 1
                stream.codec_context.thread_type = "NONE"
            except Exception:  # noqa: BLE001
                pass
            n = 0
            for _frame in container.decode(stream):
                n += 1
                if n >= 13:
                    break
        finally:
            container.close()
        return True, f"decoded {n} frames in {time.time() - t:.2f}s (codec={stream.codec_context.name})"
    except Exception as e:  # noqa: BLE001
        return False, f"{e!r}  (after {time.time() - t:.2f}s)"


def _probe_overlay(path: str, start_s: float = 0.0) -> tuple[bool, str]:
    """Use the overlay's get_frames_by_timestamps (the real code path)."""
    t = time.time()
    try:
        from cosmos_framework.data.vfm.action.gr00t_dreams.utils.video import (
            get_frames_by_timestamps,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"import failed (overlay applied + venv active?): {e!r}"
    try:
        import numpy as np

        # 13 frames at 0.1s spacing starting at ``start_s`` (test deep seeks too).
        ts = start_s + np.arange(13, dtype=np.float32) * 0.1
        frames = get_frames_by_timestamps(path, ts, video_backend="torchvision_av")
        return True, f"got frames shape={getattr(frames, 'shape', None)} in {time.time() - t:.2f}s"
    except Exception as e:  # noqa: BLE001
        return False, f"{e!r}  (after {time.time() - t:.2f}s)"


def _probe_seek_breakdown(path: str, start_s: float) -> str:
    """Time seek vs decode vs rgb-convert for a DEEP start, to find the bottleneck."""
    import av

    try:
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        tb = stream.time_base
        dur = float(stream.duration * tb) if stream.duration else None
        nframes = stream.frames or 0
        t0 = time.time()
        container.seek(int(start_s / float(tb)), stream=stream, any_frame=False, backward=True)
        t_seek = time.time() - t0
        # Decode 13 frames from the seek point; time decode and rgb-convert.
        t_dec = 0.0
        t_rgb = 0.0
        n = 0
        first_pts_s = None
        a = time.time()
        for frame in container.decode(stream):
            t_dec += time.time() - a
            if first_pts_s is None and frame.pts is not None:
                first_pts_s = float(frame.pts * tb)
            b = time.time()
            _ = frame.to_ndarray(format="rgb24")
            t_rgb += time.time() - b
            n += 1
            if n >= 13:
                break
            a = time.time()
        container.close()
        return (
            f"video: {nframes} frames, dur={dur:.1f}s | seek->{start_s:.1f}s took {t_seek:.3f}s, "
            f"landed at pts={first_pts_s:.2f}s | decode 13f={t_dec:.2f}s, rgb24-convert={t_rgb:.2f}s"
        )
    except Exception as e:  # noqa: BLE001
        return f"breakdown failed: {e!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", default=None, help="dataset leaf (auto-finds a video under videos/)")
    ap.add_argument("--video", default=None, help="explicit video file to probe")
    args = ap.parse_args()

    if args.video:
        video = Path(args.video)
    elif args.dataset_path:
        video = _find_a_video(Path(args.dataset_path))
        if video is None:
            print(f"[error] no video found under {args.dataset_path}/videos/")
            sys.exit(2)
    else:
        raise SystemExit("provide --dataset-path or --video")

    if not video.exists():
        print(f"[error] not found: {video}")
        sys.exit(2)

    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"[node] RLIMIT_NPROC soft={soft} hard={hard}  nproc={os.cpu_count()}")
    print(f"[probe] {video}\n")

    results = {}
    print("1) multithreaded pyav (ffmpeg default — the failing mode):")
    ok1, msg1 = _probe_multithreaded(str(video))
    results["multithreaded"] = ok1
    print(f"   {'[ok]  ' if ok1 else '[FAIL]'} {msg1}\n")

    print("2) single-threaded pyav (the utils/video.py fix):")
    ok2, msg2 = _probe_single_threaded(str(video))
    results["single_threaded"] = ok2
    print(f"   {'[ok]  ' if ok2 else '[FAIL]'} {msg2}\n")

    print("3) overlay get_frames_by_timestamps(video_backend='torchvision_av') @ t=0:")
    ok3, msg3 = _probe_overlay(str(video), start_s=0.0)
    results["overlay"] = ok3
    print(f"   {'[ok]  ' if ok3 else '[FAIL]'} {msg3}\n")

    print("4) overlay @ DEEP start (t=30s) — tests whether seek works or it decodes from frame 0:")
    ok4, msg4 = _probe_overlay(str(video), start_s=30.0)
    results["overlay_deep"] = ok4
    print(f"   {'[ok]  ' if ok4 else '[FAIL]'} {msg4}\n")

    print("5) seek/decode/rgb breakdown @ t=30s:")
    print(f"   {_probe_seek_breakdown(str(video), 30.0)}\n")

    print("=" * 70)
    if results["single_threaded"] and not results["multithreaded"]:
        print("DIAGNOSIS: multithreaded decode fails but single-threaded works ->")
        print("  the EAGAIN was a thread-spawn limit; the utils/video.py fix resolves it.")
        print("  Make sure the overlay is applied (re-run apply_overlay.sh) so result (3) is [ok].")
    elif results["single_threaded"] and results["overlay"]:
        print("DIAGNOSIS: decode works (incl. the real overlay path). You're good to re-run stats.")
    elif not any(results.values()):
        print("DIAGNOSIS: ALL decode modes fail on this node.")
        print("  Likely a contended login node (cgroup pids limit) or a broken file/codec.")
        print("  Try running on a COMPUTE node (srun/sbatch), or check the file with ffprobe.")
    else:
        print("DIAGNOSIS: mixed result — see per-mode messages above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
