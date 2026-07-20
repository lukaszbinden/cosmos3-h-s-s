#!/usr/bin/env python3
"""Write the concise report for the mixed-mode C3 forward-dynamics evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--chss-checkpoint", required=True)
    parser.add_argument("--c3-checkpoint", required=True)
    parser.add_argument("--c3-label", default=None)
    parser.add_argument("--c3-training-note", default=None)
    return parser.parse_args()


def metric(summary: dict, model: str, name: str) -> tuple[float, float]:
    item = summary[model][name]
    return float(item["mean_across_seeds"]), float(item["std_across_seeds"])


def main() -> None:
    args = parse_args()
    checkpoint_name = Path(args.c3_checkpoint).name
    try:
        c3_step = int(checkpoint_name.removeprefix("iter_"))
        c3_step_label = f"{c3_step:,}"
    except ValueError:
        c3_step_label = checkpoint_name
    c3_label = args.c3_label or f"Mixed-mode C3-H-S-S @ {c3_step_label} steps (FD inference)"
    summary = json.loads((args.run_dir / "fds_4subset_3seed_summary.json").read_text())
    ch_l1, ch_l1_sd = metric(summary, "chss", "l1")
    c3_l1, c3_l1_sd = metric(summary, "c3hss", "l1")
    ch_ssim, ch_ssim_sd = metric(summary, "chss", "ssim")
    c3_ssim, c3_ssim_sd = metric(summary, "c3hss", "ssim")
    subset_names = {
        "cholecystectomy": "Cholecystectomy",
        "hysterectomy": "Hysterectomy",
        "inguinal_hernia": "Inguinal hernia",
        "prostatectomy": "Prostatectomy",
    }
    rows = []
    for subset in summary["subsets"]:
        ch = summary["chss"]["per_subset"][subset]
        c3 = summary["c3hss"]["per_subset"][subset]
        rows.append(
            f"| {subset_names.get(subset, subset)} | "
            f"{ch['l1']['mean_across_seeds']:.6f} ± {ch['l1']['std_across_seeds']:.6f} | "
            f"{c3['l1']['mean_across_seeds']:.6f} ± {c3['l1']['std_across_seeds']:.6f} | "
            f"{ch['ssim']['mean_across_seeds']:.6f} | "
            f"{c3['ssim']['mean_across_seeds']:.6f} |"
        )

    report = f"""# C-H-S-S vs mixed-mode C3-H-S-S in FD mode

## Result

| Model | Balanced four-subset mean L1 | Seed SD | Mean SSIM | SSIM seed SD |
|---|---:|---:|---:|---:|
| C-H-S-S | {ch_l1:.6f} | {ch_l1_sd:.6f} | {ch_ssim:.6f} | {ch_ssim_sd:.6f} |
| {c3_label} | {c3_l1:.6f} | {c3_l1_sd:.6f} | {c3_ssim:.6f} | {c3_ssim_sd:.6f} |

Lower L1 and higher SSIM are better. Error statistics are computed across three seed-level curves, where each seed is first averaged over five episodes from each of all four operative CMR subsets (20 episodes per seed).

## Per-subset results

| Operative CMR subset | C-H-S-S L1 | Mixed C3 FD L1 | C-H-S-S SSIM | Mixed C3 FD SSIM |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Comparison parameters

- Subsets: cholecystectomy, hysterectomy, inguinal hernia, and prostatectomy.
- Episodes: five matched test episodes per subset and seed.
- Seeds: 0, 1, and 2.
- Horizon: 17 autoregressive chunks, 204 predicted frames plus one conditioning frame.
- C3 inference mode: `forward_dynamics`; frame 0 and action tokens are clean conditioning, and future vision tokens are generated.
- C3 rollout and autoregressive feedback resolution: native `832×480`, matching the corrected mixed-mode training loader.
- Common FDS scoring grid: `512×288` for both models. C3 prediction and GT are downsampled only after the native-resolution rollout.
- Comparison-video display grid: `512×288` per panel, with ground truth left, C-H-S-S center, and mixed-mode C3-H-S-S right.
- C-H-S-S checkpoint: `{args.chss_checkpoint}`.
- C3-H-S-S checkpoint: `{args.c3_checkpoint}`.
{f'- C3 training provenance: {args.c3_training_note}' if args.c3_training_note else ''}
- The unchanged C-H-S-S rollouts were reused from `../20260701_120846/chss`; C3 was newly generated from the mixed-mode checkpoint.

## Artifacts

- `fds_4subset_3seed_mean_curve.png`: mean long-horizon L1/FDS curves with ±1 sample SD across three seeds.
- `fds_4subset_3seed_summary.json`: overall and per-subset metrics.
- `fds_per_seed_and_subset.csv`: per-model, seed, and subset aggregates.
- `fds_seed_level_curves.npz`: seed-level curves used for plotting.
- `comparison/<subset>/seed{{0,1,2}}`: 60 GT | C-H-S-S | mixed C3 comparison videos.
- `comparison/manifest.json`: video paths and frame counts.
- `c3hss/<subset>/seed*`: native 832×480 C3 rollout videos and result JSON files.
"""
    (args.run_dir / "c-h-s-s_vs_c3-h-s-s_report.md").write_text(report)


if __name__ == "__main__":
    main()
