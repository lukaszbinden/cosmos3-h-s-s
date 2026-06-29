# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Frame Decay Score (FDS) — faithful port of od-hamlyn-cmr's compute_frame_decay.

FDS measures how a generated video clip diverges from ground truth over the
prediction horizon. Per generated frame (the conditioning frame is skipped) it
computes L1 and SSIM against the GT frame, then summarizes:

  * mean_l1  -> the headline FDS (lower is better)
  * l1_slope -> the "decay": linear-fit slope of per-frame L1 over time
                (>0 means error grows across the horizon)
  * mean_ssim (higher is better) + per-chunk early/mid/late breakdowns.

Reference: od-hamlyn-cmr/scripts/cosmos_h_surgical_simulator_quant_eval.py
::compute_frame_decay (kept numerically identical: normalize to [-1,1] via
/127.5-1, data_range=2.0, skip frame 0).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- SSIM (optional): prefer scikit-image, fall back to pytorch_msssim, else None.
try:
    from skimage.metrics import structural_similarity as _sk_ssim

    def _compute_ssim(im1: np.ndarray, im2: np.ndarray, data_range: float = 2.0) -> float:
        # im1/im2: H,W,C float; channel_axis=2
        return float(_sk_ssim(im1, im2, data_range=data_range, channel_axis=2))
except Exception:  # noqa: BLE001
    try:
        import torch
        from pytorch_msssim import ssim as _pt_ssim

        def _compute_ssim(im1: np.ndarray, im2: np.ndarray, data_range: float = 2.0) -> float:
            t1 = torch.from_numpy(im1).permute(2, 0, 1).unsqueeze(0).float()
            t2 = torch.from_numpy(im2).permute(2, 0, 1).unsqueeze(0).float()
            return float(_pt_ssim(t1, t2, data_range=data_range).item())
    except Exception:  # noqa: BLE001
        _compute_ssim = None  # type: ignore[assignment]


# Per-chunk breakdown over GENERATED frames (after skipping the conditioning frame).
# CHUNK_SIZE=12 mirrors od-hamlyn-cmr (1 action-chunk = 12 predicted frames).
CHUNK_SIZE = 12
MAX_CHUNKS = 6
CHUNK_RANGES = {
    "early_c1": (0, CHUNK_SIZE),  # frames 0-11
    "mid_c2c3": (CHUNK_SIZE, CHUNK_SIZE * 3),  # frames 12-35
    "late_c4c6": (CHUNK_SIZE * 3, CHUNK_SIZE * MAX_CHUNKS),  # frames 36-71
}


def _chunk_mean(values: list[float], start: int, end: int) -> float:
    sl = values[start:end]
    return float(np.mean(sl)) if sl else float("nan")


def compute_frame_decay(gt_video: np.ndarray, gen_video: np.ndarray) -> dict[str, Any]:
    """Per-frame L1 + SSIM between GT and generated uint8 [T,H,W,C] videos.

    Skips frame 0 (conditioning frame). Images normalized to [-1, 1] (data_range=2.0).
    Returns scalar summaries + per-frame arrays + per-chunk breakdowns. Numerically
    identical to od-hamlyn-cmr's compute_frame_decay.
    """
    T = min(int(gt_video.shape[0]), int(gen_video.shape[0]))
    gt_norm = gt_video[:T].astype(np.float32) / 127.5 - 1.0
    gen_norm = gen_video[:T].astype(np.float32) / 127.5 - 1.0

    l1_list: list[float] = []
    ssim_list: list[float] = []
    for t in range(1, T):
        l1_list.append(float(np.mean(np.abs(gen_norm[t] - gt_norm[t]))))
        if _compute_ssim is not None:
            ssim_list.append(_compute_ssim(gen_norm[t], gt_norm[t], data_range=2.0))

    mean_l1 = float(np.mean(l1_list)) if l1_list else float("nan")
    std_l1 = float(np.std(l1_list)) if l1_list else float("nan")
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else float("nan")
    std_ssim = float(np.std(ssim_list)) if ssim_list else float("nan")

    l1_slope = float("nan")
    if len(l1_list) > 1:
        l1_slope = float(np.polyfit(np.arange(len(l1_list)), l1_list, 1)[0])

    per_chunk: dict[str, dict[str, float]] = {}
    for cname, (s, e) in CHUNK_RANGES.items():
        per_chunk[cname] = {
            "l1": _chunk_mean(l1_list, s, e),
            "ssim": _chunk_mean(ssim_list, s, e) if ssim_list else float("nan"),
        }

    return {
        "mean_l1": mean_l1,
        "std_l1": std_l1,
        "mean_ssim": mean_ssim,
        "std_ssim": std_ssim,
        "l1_slope": l1_slope,
        "l1_per_frame": l1_list,
        "ssim_per_frame": ssim_list,
        "num_frames": len(l1_list),
        "per_chunk": per_chunk,
    }


def video_tensor_to_uint8(video) -> np.ndarray:
    """Convert a decoded video tensor in [-1,1], shape [C,T,H,W] (or [1,C,T,H,W]),
    to a uint8 numpy array [T,H,W,C] suitable for compute_frame_decay."""
    import torch

    v = video
    if isinstance(v, torch.Tensor):
        if v.dim() == 5:  # [B,C,T,H,W] -> assume B=1
            v = v[0]
        v = v.detach().float().cpu()
        v = ((v.clamp(-1, 1) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)  # [C,T,H,W]
        v = v.permute(1, 2, 3, 0).contiguous().numpy()  # [T,H,W,C]
        return v
    arr = np.asarray(v)
    return arr


def ssim_available() -> bool:
    return _compute_ssim is not None
