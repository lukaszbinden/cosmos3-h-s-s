#!/usr/bin/env python3
"""Compare nearest-training-window interventions across two checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--earlier-summary", required=True)
    parser.add_argument("--later-summary", required=True)
    parser.add_argument("--heldout-summary", required=True)
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).resolve().read_text())


def _episode_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        record["subset"],
        record["target_arm"],
        int(record["episode_id"]),
        int(record["base_index"]),
    )


def _by_seed(record: dict[str, Any]) -> dict[int, float]:
    return {
        int(item["seed"]): float(item["response_fraction"])
        for item in record["by_seed"]
    }


def main() -> None:
    args = parse_args()
    earlier = _load(args.earlier_summary)
    later = _load(args.later_summary)
    heldout = _load(args.heldout_summary)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    earlier_iteration = int(earlier["iteration"])
    later_iteration = int(later["iteration"])
    if earlier_iteration >= later_iteration:
        raise ValueError("Earlier iteration must precede later iteration")

    earlier_episodes = {
        _episode_key(record): record for record in earlier["episode_groups"]
    }
    later_episodes = {
        _episode_key(record): record for record in later["episode_groups"]
    }
    if earlier_episodes.keys() != later_episodes.keys():
        raise ValueError("Checkpoint summaries do not contain the same windows")

    episode_comparisons = []
    for key in sorted(earlier_episodes):
        earlier_record = earlier_episodes[key]
        later_record = later_episodes[key]
        earlier_seeds = _by_seed(earlier_record)
        later_seeds = _by_seed(later_record)
        if earlier_seeds.keys() != later_seeds.keys():
            raise ValueError(f"Seed mismatch for {key}")
        deltas = {
            seed: later_seeds[seed] - earlier_seeds[seed] for seed in earlier_seeds
        }
        episode_comparisons.append(
            {
                "subset": key[0],
                "target_arm": key[1],
                "episode_id": key[2],
                "base_index": key[3],
                "earlier_mean": float(earlier_record["mean"]),
                "later_mean": float(later_record["mean"]),
                "mean_delta": (
                    float(later_record["mean"]) - float(earlier_record["mean"])
                ),
                "earlier_localized_seed_count": int(
                    earlier_record["localized_seed_count"]
                ),
                "later_localized_seed_count": int(later_record["localized_seed_count"]),
                "seed_count": int(earlier_record["seed_count"]),
                "paired_seed_deltas": [
                    {"seed": seed, "delta": delta}
                    for seed, delta in sorted(deltas.items())
                ],
            }
        )

    earlier_cell = earlier["cell_groups"]
    later_cell = later["cell_groups"]
    if len(earlier_cell) != 1 or len(later_cell) != 1:
        raise ValueError("Expected one aggregate subset/arm cell per checkpoint")
    aggregate = {
        "earlier": earlier_cell[0],
        "later": later_cell[0],
        "mean_delta": float(later_cell[0]["mean"]) - float(earlier_cell[0]["mean"]),
    }

    heldout_focus = heldout.get("focus_episode")
    if not heldout_focus:
        raise ValueError("Held-out summary has no focus episode")

    all_training_localized = all(
        record["earlier_localized_seed_count"] == record["seed_count"]
        and record["later_localized_seed_count"] == record["seed_count"]
        for record in episode_comparisons
    )
    heldout_never_localized = int(heldout_focus["localized_seed_count"]) == 0
    if all_training_localized and heldout_never_localized:
        verdict = "training_windows_localize_but_heldout_focus_does_not"
    elif heldout_never_localized:
        verdict = "mixed_training_localization_and_heldout_focus_failure"
    else:
        verdict = "no_clean_training_vs_heldout_separation"

    with Path(args.selection_csv).resolve().open(newline="") as handle:
        selection_rows = list(csv.DictReader(handle))
    selected_keys = {
        (row["subset"], int(row["episode_id"]), int(row["base_index"]))
        for row in selection_rows
    }
    measured_keys = {
        (record["subset"], record["episode_id"], record["base_index"])
        for record in episode_comparisons
    }
    if not measured_keys.issubset(selected_keys):
        raise ValueError("A measured window is absent from the selection CSV")

    payload = {
        "diagnostic": "nearest-training-window checkpoint comparison",
        "verdict": verdict,
        "interpretation": (
            "Training-window success shows that the checkpoint and action wiring can "
            "bind PSM1 interventions to the correct visual tool in familiar windows. "
            "It does not establish held-out generalization."
        ),
        "earlier_iteration": earlier_iteration,
        "later_iteration": later_iteration,
        "aggregate": aggregate,
        "episodes": episode_comparisons,
        "heldout_focus": heldout_focus,
        "sources": {
            "earlier_summary": str(Path(args.earlier_summary).resolve()),
            "later_summary": str(Path(args.later_summary).resolve()),
            "heldout_summary": str(Path(args.heldout_summary).resolve()),
            "selection_csv": str(Path(args.selection_csv).resolve()),
        },
    }
    (output_dir / "checkpoint_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    with (output_dir / "checkpoint_comparison.csv").open("w", newline="") as handle:
        fieldnames = [
            "subset",
            "target_arm",
            "episode_id",
            "base_index",
            "earlier_mean",
            "later_mean",
            "mean_delta",
            "earlier_localized_seed_count",
            "later_localized_seed_count",
            "seed_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in episode_comparisons:
            writer.writerow({key: record[key] for key in fieldnames})

    lines = [
        "# Nearest-training-window intervention audit",
        "",
        f"Verdict: `{verdict}`.",
        "",
        (
            f"The six nearest unique HF SutureBot training windows had an aggregate "
            f"intended-tool response of {aggregate['earlier']['mean']:.3f} at iter "
            f"{earlier_iteration} and {aggregate['later']['mean']:.3f} at iter "
            f"{later_iteration}. A response above 0.5 means the intervention changed "
            "the commanded PSM1 more than the other visible tool."
        ),
        "",
        "| Window | Earlier mean | Later mean | Delta | Earlier localized | Later localized |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in episode_comparisons:
        lines.append(
            f"| {record['episode_id']}:{record['base_index']} | "
            f"{record['earlier_mean']:.3f} | {record['later_mean']:.3f} | "
            f"{record['mean_delta']:+.3f} | "
            f"{record['earlier_localized_seed_count']}/{record['seed_count']} | "
            f"{record['later_localized_seed_count']}/{record['seed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Held-out contrast",
            "",
            (
                f"The prior HF focus window "
                f"{heldout_focus['episode_id']}:{heldout_focus['base_index']} had a "
                f"mean intended-tool response of {heldout_focus['mean']:.3f} and "
                f"localized in {heldout_focus['localized_seed_count']}/"
                f"{heldout_focus['seed_count']} seeds."
            ),
            "",
            "## Interpretation boundary",
            "",
            (
                "These are deliberately selected training windows. Correct localization "
                "here rules out a universal PSM1/PSM2 label swap or a globally broken "
                "intervention path, but it can reflect memorization or familiar visual "
                "modes. It does not by itself show held-out action following."
            ),
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
