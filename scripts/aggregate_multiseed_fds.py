#!/usr/bin/env python3
"""Aggregate FDS by episode within seed, then compute uncertainty across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--chss-root", required=True)
    p.add_argument("--c3hss-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--expected-episodes", type=int, default=10)
    return p.parse_args()


def load_model(root: Path, filename: str, seeds: list[int], expected_episodes: int):
    seed_curves = []
    seed_scalars = []
    seed_ssim = []
    episodes_ref = None
    checkpoint = None
    records = []
    for seed in seeds:
        payload = json.loads((root / f"seed{seed}" / filename).read_text())
        episodes = [int(v) for v in payload["episodes"]]
        if len(episodes) != expected_episodes or len(payload["results"]) != expected_episodes:
            raise RuntimeError(f"seed {seed}: expected {expected_episodes} episodes, got {len(payload['results'])}")
        if episodes_ref is None:
            episodes_ref = episodes
        elif episodes != episodes_ref:
            raise RuntimeError(f"seed {seed}: episode mismatch {episodes} vs {episodes_ref}")
        if int(payload["seed"]) != seed:
            raise RuntimeError(f"seed directory seed{seed} contains payload seed={payload['seed']}")
        checkpoint = payload["checkpoint"]
        curves = np.asarray([r["fds"]["l1_per_frame"] for r in payload["results"]], dtype=np.float64)
        if curves.ndim != 2 or curves.shape[0] != expected_episodes:
            raise RuntimeError(f"seed {seed}: invalid FDS curve shape {curves.shape}")
        seed_curve = curves.mean(axis=0)
        scalar_l1 = float(np.mean([r["fds"]["mean_l1"] for r in payload["results"]]))
        scalar_ssim = float(np.mean([r["fds"]["mean_ssim"] for r in payload["results"]]))
        seed_curves.append(seed_curve)
        seed_scalars.append(scalar_l1)
        seed_ssim.append(scalar_ssim)
        records.append({"seed": seed, "episode_mean_l1": scalar_l1, "episode_mean_ssim": scalar_ssim})
    return {
        "checkpoint": checkpoint,
        "episodes": episodes_ref,
        "curves": np.asarray(seed_curves),
        "scalar_l1": np.asarray(seed_scalars),
        "scalar_ssim": np.asarray(seed_ssim),
        "records": records,
    }


def summarize(model: dict) -> dict:
    l1 = model["scalar_l1"]
    ssim = model["scalar_ssim"]
    return {
        "checkpoint": model["checkpoint"],
        "seed_mean_l1_values": l1.tolist(),
        "mean_l1_across_seeds": float(l1.mean()),
        "std_l1_across_seeds": float(l1.std(ddof=1)),
        "sem_l1_across_seeds": float(l1.std(ddof=1) / np.sqrt(len(l1))),
        "seed_mean_ssim_values": ssim.tolist(),
        "mean_ssim_across_seeds": float(ssim.mean()),
        "std_ssim_across_seeds": float(ssim.std(ddof=1)),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chss = load_model(Path(args.chss_root), "chss_results.json", args.seeds, args.expected_episodes)
    c3hss = load_model(Path(args.c3hss_root), "c3hss_results.json", args.seeds, args.expected_episodes)
    if chss["episodes"] != c3hss["episodes"]:
        raise RuntimeError(f"model episode mismatch: {chss['episodes']} vs {c3hss['episodes']}")
    if chss["curves"].shape != c3hss["curves"].shape:
        raise RuntimeError(f"model curve shape mismatch: {chss['curves'].shape} vs {c3hss['curves'].shape}")

    np.savez_compressed(
        out / "fds_seed_level_curves.npz",
        seeds=np.asarray(args.seeds),
        episodes=np.asarray(chss["episodes"]),
        chss=chss["curves"],
        c3hss=c3hss["curves"],
    )
    rows = []
    for model_name, model in (("C-H-S-S", chss), ("C3-H-S-S", c3hss)):
        for record in model["records"]:
            rows.append({"model": model_name, **record})
    with (out / "fds_per_seed.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "episode_mean_l1", "episode_mean_ssim"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "aggregation": "For each seed, average 10 episode curves; then mean/std across 3 seed-level curves.",
        "error_band": "plus/minus 1 sample standard deviation across the 3 seed-level curves",
        "seeds": args.seeds,
        "episodes": chss["episodes"],
        "num_episodes_per_seed": args.expected_episodes,
        "predicted_frames": int(chss["curves"].shape[1]),
        "chss": summarize(chss),
        "c3hss": summarize(c3hss),
    }
    (out / "fds_3seed_summary.json").write_text(json.dumps(summary, indent=2))

    x = np.arange(1, chss["curves"].shape[1] + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for model, label, color in ((chss, "C-H-S-S", "#1f77b4"), (c3hss, "C3-H-S-S", "#d62728")):
        mean = model["curves"].mean(axis=0)
        sd = model["curves"].std(axis=0, ddof=1)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.20, linewidth=0)
    for boundary in range(12, len(x), 12):
        ax.axvline(boundary, color="0.75", linewidth=0.7, linestyle="--")
    ax.set(
        title="Long-horizon FDS: mean across 3 seeds ± 1 SD",
        xlabel="Predicted frame",
        ylabel="L1 error in [-1, 1] space",
    )
    ax.text(
        0.01,
        0.01,
        "Each seed curve is first averaged over 10 matched episodes",
        transform=ax.transAxes,
        fontsize=9,
        color="0.35",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fds_3seed_mean_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
