#!/usr/bin/env python3
"""Aggregate motion-matched tool-response diagnostics across diffusion seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4]
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "localized_seed_count": sum(value > 0.5 for value in values),
        "seed_count": len(values),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    found_seeds = []
    for seed in args.expected_seeds:
        path = input_dir / f"seed{seed}" / "matched_response_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text())
        if payload.get("seed") != seed:
            raise ValueError(
                f"{path} records seed {payload.get('seed')}, expected {seed}"
            )
        summaries.append(payload)
        found_seeds.append(seed)

    cell_values: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    episode_values: dict[tuple[str, str, int, int], list[tuple[int, float]]] = (
        defaultdict(list)
    )
    for payload in summaries:
        seed = int(payload["seed"])
        for record in payload["aggregate_groups"]:
            cell_values[(record["subset"], record["target_arm"])].append(
                (seed, float(record["intended_response_fraction"]))
            )
        for record in payload["episode_groups"]:
            episode_values[
                (
                    record["subset"],
                    record["target_arm"],
                    int(record["episode_id"]),
                    int(record["base_index"]),
                )
            ].append((seed, float(record["intended_response_fraction"])))

    cell_groups = []
    for (subset, target_arm), pairs in sorted(cell_values.items()):
        pairs.sort()
        values = [value for _, value in pairs]
        cell_groups.append(
            {
                "subset": subset,
                "target_arm": target_arm,
                "by_seed": [
                    {"seed": seed, "response_fraction": value} for seed, value in pairs
                ],
                **_summary(values),
            }
        )

    episode_groups = []
    for (subset, target_arm, episode_id, base_index), pairs in sorted(
        episode_values.items()
    ):
        pairs.sort()
        values = [value for _, value in pairs]
        episode_groups.append(
            {
                "subset": subset,
                "target_arm": target_arm,
                "episode_id": episode_id,
                "base_index": base_index,
                "by_seed": [
                    {"seed": seed, "response_fraction": value} for seed, value in pairs
                ],
                **_summary(values),
            }
        )

    focus = next(
        record
        for record in episode_groups
        if record["subset"] == "hf_suturebot"
        and record["target_arm"] == "psm1"
        and record["episode_id"] == 1382
        and record["base_index"] == 381
    )
    if focus["localized_seed_count"] == 0:
        focus_verdict = "systematic_cross_localization"
    elif focus["localized_seed_count"] == focus["seed_count"]:
        focus_verdict = "localized_in_all_seeds"
    else:
        focus_verdict = "sampling_sensitive"

    payload = {
        "diagnostic": "multi-seed motion-matched tool response",
        "model": summaries[0]["model"],
        "iteration": summaries[0]["iteration"],
        "seeds": found_seeds,
        "localization_threshold": 0.5,
        "focus_episode": {
            "subset": "hf_suturebot",
            "target_arm": "psm1",
            "episode_id": 1382,
            "base_index": 381,
            "verdict": focus_verdict,
            **focus,
        },
        "cell_groups": cell_groups,
        "episode_groups": episode_groups,
        "source_summaries": [
            str(input_dir / f"seed{seed}" / "matched_response_summary.json")
            for seed in found_seeds
        ],
    }
    (output_dir / "multiseed_response_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    lines = [
        "# Multi-seed motion-matched response",
        "",
        f"Model: {payload['model']}. Seeds: {', '.join(map(str, found_seeds))}.",
        "Response fractions above 0.5 localize the counterfactual change to",
        "the tool mapped to the commanded arm.",
        "",
        "| Dataset | Target | Mean | Std | Min–max | Localized seeds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in cell_groups:
        lines.append(
            f"| {record['subset']} | {record['target_arm']} | "
            f"{record['mean']:.3f} | {record['population_std']:.3f} | "
            f"{record['min']:.3f}–{record['max']:.3f} | "
            f"{record['localized_seed_count']}/{record['seed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Episode replication",
            "",
            "| Dataset | Target | Episode:base | Mean | Std | Min–max | Localized seeds |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for record in episode_groups:
        lines.append(
            f"| {record['subset']} | {record['target_arm']} | "
            f"{record['episode_id']}:{record['base_index']} | "
            f"{record['mean']:.3f} | {record['population_std']:.3f} | "
            f"{record['min']:.3f}–{record['max']:.3f} | "
            f"{record['localized_seed_count']}/{record['seed_count']} |"
        )
    lines.extend(
        [
            "",
            "## HF PSM1 episode 1382",
            "",
            (
                f"Verdict: `{focus_verdict}`. It localized in "
                f"{focus['localized_seed_count']}/{focus['seed_count']} seeds."
            ),
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
