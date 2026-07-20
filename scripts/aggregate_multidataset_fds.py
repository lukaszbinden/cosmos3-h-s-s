#!/usr/bin/env python3
"""Aggregate balanced multi-dataset FDS within seed, then across seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUBSETS = ["cholecystectomy", "hysterectomy", "inguinal_hernia", "prostatectomy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chss-root", type=Path, required=True)
    parser.add_argument("--c3hss-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes-per-subset", type=int, default=5)
    parser.add_argument("--c3-label", default="C3-H-S-S")
    return parser.parse_args()


def load_model(
    root: Path,
    filename: str,
    subsets: list[str],
    seeds: list[int],
    episodes_per_subset: int,
) -> dict:
    seed_curves = []
    seed_l1 = []
    seed_ssim = []
    subset_seed_l1 = {subset: [] for subset in subsets}
    subset_seed_ssim = {subset: [] for subset in subsets}
    episode_ids: dict[str, list[int]] = {}
    checkpoint = None
    rows = []

    for seed in seeds:
        all_curves = []
        all_l1 = []
        all_ssim = []
        for subset in subsets:
            payload = json.loads((root / subset / f"seed{seed}" / filename).read_text())
            if int(payload["seed"]) != seed:
                raise RuntimeError(f"{subset}/seed{seed} contains payload seed={payload['seed']}")
            if payload.get("dataset_name", Path(payload["dataset"]).name) != subset:
                raise RuntimeError(f"Dataset mismatch in {subset}/seed{seed}")
            results = payload["results"]
            if len(results) != episodes_per_subset:
                raise RuntimeError(
                    f"{subset}/seed{seed}: expected {episodes_per_subset} episodes, got {len(results)}"
                )
            ids = [int(result["episode_id"]) for result in results]
            if subset not in episode_ids:
                episode_ids[subset] = ids
            elif episode_ids[subset] != ids:
                raise RuntimeError(f"Episode mismatch for {subset}: {ids} vs {episode_ids[subset]}")
            curves = np.asarray([result["fds"]["l1_per_frame"] for result in results], dtype=np.float64)
            if curves.ndim != 2 or curves.shape[0] != episodes_per_subset:
                raise RuntimeError(f"Invalid curve shape for {subset}/seed{seed}: {curves.shape}")
            l1 = [float(result["fds"]["mean_l1"]) for result in results]
            ssim = [float(result["fds"]["mean_ssim"]) for result in results]
            subset_l1 = float(np.mean(l1))
            subset_ssim = float(np.mean(ssim))
            subset_seed_l1[subset].append(subset_l1)
            subset_seed_ssim[subset].append(subset_ssim)
            rows.append(
                {
                    "seed": seed,
                    "subset": subset,
                    "episode_mean_l1": subset_l1,
                    "episode_mean_ssim": subset_ssim,
                }
            )
            all_curves.append(curves)
            all_l1.extend(l1)
            all_ssim.extend(ssim)
            checkpoint = payload["checkpoint"]

        combined = np.concatenate(all_curves, axis=0)
        seed_curves.append(combined.mean(axis=0))
        seed_l1.append(float(np.mean(all_l1)))
        seed_ssim.append(float(np.mean(all_ssim)))
        rows.append(
            {
                "seed": seed,
                "subset": "ALL_FOUR_CMR_SUBSETS",
                "episode_mean_l1": seed_l1[-1],
                "episode_mean_ssim": seed_ssim[-1],
            }
        )

    return {
        "checkpoint": checkpoint,
        "episode_ids": episode_ids,
        "curves": np.asarray(seed_curves),
        "scalar_l1": np.asarray(seed_l1),
        "scalar_ssim": np.asarray(seed_ssim),
        "subset_l1": {key: np.asarray(value) for key, value in subset_seed_l1.items()},
        "subset_ssim": {key: np.asarray(value) for key, value in subset_seed_ssim.items()},
        "rows": rows,
    }


def stats(values: np.ndarray) -> dict[str, object]:
    return {
        "seed_values": values.tolist(),
        "mean_across_seeds": float(values.mean()),
        "std_across_seeds": float(values.std(ddof=1)),
        "sem_across_seeds": float(values.std(ddof=1) / np.sqrt(len(values))),
    }


def summarize(model: dict, subsets: list[str]) -> dict:
    return {
        "checkpoint": model["checkpoint"],
        "l1": stats(model["scalar_l1"]),
        "ssim": stats(model["scalar_ssim"]),
        "per_subset": {
            subset: {
                "l1": stats(model["subset_l1"][subset]),
                "ssim": stats(model["subset_ssim"][subset]),
            }
            for subset in subsets
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chss = load_model(
        args.chss_root, "chss_results.json", args.subsets, args.seeds, args.episodes_per_subset
    )
    c3hss = load_model(
        args.c3hss_root, "c3hss_results.json", args.subsets, args.seeds, args.episodes_per_subset
    )
    if chss["episode_ids"] != c3hss["episode_ids"]:
        raise RuntimeError("Models do not contain identical subset/episode selections")
    if chss["curves"].shape != c3hss["curves"].shape:
        raise RuntimeError(f"Curve shape mismatch: {chss['curves'].shape} vs {c3hss['curves'].shape}")

    np.savez_compressed(
        args.output_dir / "fds_seed_level_curves.npz",
        seeds=np.asarray(args.seeds),
        subsets=np.asarray(args.subsets),
        chss=chss["curves"],
        c3hss=c3hss["curves"],
    )
    with (args.output_dir / "fds_per_seed_and_subset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "seed", "subset", "episode_mean_l1", "episode_mean_ssim"],
        )
        writer.writeheader()
        for model_name, model in (("C-H-S-S", chss), (args.c3_label, c3hss)):
            for row in model["rows"]:
                writer.writerow({"model": model_name, **row})

    summary = {
        "aggregation": (
            f"For each seed, average {args.episodes_per_subset} episodes from each of all four "
            f"operative CMR subsets ({args.episodes_per_subset * len(args.subsets)} balanced episodes); "
            "then compute mean/std across the 3 seed-level curves."
        ),
        "error_band": "plus/minus 1 sample standard deviation across the 3 seed-level curves",
        "subsets": args.subsets,
        "episodes_per_subset_per_seed": args.episodes_per_subset,
        "episodes_per_seed": args.episodes_per_subset * len(args.subsets),
        "episode_ids": chss["episode_ids"],
        "seeds": args.seeds,
        "predicted_frames": int(chss["curves"].shape[1]),
        "chss": summarize(chss, args.subsets),
        "c3hss": summarize(c3hss, args.subsets),
    }
    (args.output_dir / "fds_4subset_3seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    x = np.arange(1, chss["curves"].shape[1] + 1)
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    for model, label, color in (
        (chss, "C-H-S-S", "#1f77b4"),
        (c3hss, args.c3_label, "#d62728"),
    ):
        mean = model["curves"].mean(axis=0)
        sd = model["curves"].std(axis=0, ddof=1)
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.20, linewidth=0)
    for boundary in range(12, len(x), 12):
        ax.axvline(boundary, color="0.75", linewidth=0.7, linestyle="--")
    ax.set_title(
        "Very-long-horizon FDS across all four operative CMR subsets\n"
        "5 episodes per subset × 3 seeds; mean ± 1 seed SD"
    )
    ax.set_xlabel("Predicted frame")
    ax.set_ylabel("L1 error in [-1, 1] space")
    ax.text(
        0.01,
        0.01,
        "Cholecystectomy · Hysterectomy · Inguinal hernia · Prostatectomy (20 episodes per seed)",
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.35",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "fds_4subset_3seed_mean_curve.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
