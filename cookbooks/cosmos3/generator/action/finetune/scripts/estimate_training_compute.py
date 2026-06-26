#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Estimate cumulative training compute (FLOP) for the Open-H 44D finetune.

Implements the EU AI Act "cumulative training compute" approximation from
``doc/Cummulative Compute Calculation.pdf``::

    Compute  ~=  6 * N * D

where ``N`` = total model parameters and ``D`` = number of training examples
(or tokens) the model is trained on. This is the SAME formula and the SAME
definition of ``D`` (training examples) used to report the Cosmos-Predict2.5
Open-H number (1.48e17 FLOP, with N=2e9 => D~=1.23e7 examples), so the two are
directly comparable.

``D`` is genuinely ambiguous, so this script reports BOTH well-defined notions:

  * ``D_seen``  (compute basis): how many training examples the configured run
    actually processes = ``examples_per_step * num_steps``. With count-based
    batching (``max_samples_per_batch`` set, as in this cookbook's toml),
    ``examples_per_step = max_samples_per_batch * world_size`` exactly. This is
    what ``6ND`` wants and the number to report for compliance.
  * ``D_dataset`` (epoch basis): the number of UNIQUE training windows available
    in ``OPEN_H_DATASET_SPECS`` for one pass. The loader enumerates one window
    per frame (``base_index in range(trajectory_length)``), so for non-CMR
    leaves ``D_dataset ~= sum(total_frames)``; CMR leaves are clutch-filtered to
    a fraction of their frames (use ``--cmr-keep-frac`` to approximate, or
    ``--from-filter-cache`` to read the exact post-filter counts).

The headline figure uses ``D_seen``. ``D_dataset`` is printed alongside so you
can see how many epochs the run corresponds to (``epochs ~= D_seen /
D_dataset``).

With ``--energy`` it also estimates training **energy** (kWh/MWh) and **CO2e**
(kg/t) the energy-first way (the FLOP number is NOT used for this, since
FLOP->energy needs an efficiency assumption equivalent to just knowing
GPU-hours)::

    Energy(kWh) = GPU_hours x (GPU_TDP_kW x util) x PUE
    CO2e(kg)    = Energy(kWh) x grid_carbon_intensity(kgCO2e/kWh)

GPU-hours come from ``--from-sacct <jobid>`` (queried from Slurm, preferred) or
``--gpu-hours <measured>``, else estimated from ``sec_per_iter x steps x
world_size``. PUE and carbon intensity are external assumptions (defaults:
PUE=1.2, 0.35 kgCO2e/kWh world avg, H100 700W @ 70%).

Defaults mirror the committed run config
(``toml/sft_config/action_fdm_open_h_sft_nano.toml`` +
``scripts/slurm_train.sbatch``): N=8e9 (Qwen3-VL-8B backbone),
max_samples_per_batch=64, 6 nodes x 8 GPU = 48 GPUs, max_iter=20000.

Usage::

    # EVERYTHING at once (compute + dataset/epochs + energy + CO2):
    python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT" --all

    # ...with measured GPU-hours from Slurm for a reportable energy/CO2 figure.
    # Easiest: sum the WHOLE resubmit chain by job name (no ids to enumerate):
    python scripts/estimate_training_compute.py --all --no-dataset \\
        --sacct-name --pue 1.1 --carbon-intensity 0.05
    # ...or pass explicit id(s) (comma-separated for resubmits):
    python scripts/estimate_training_compute.py --all --no-dataset \\
        --from-sacct <jobid[,jobid2,...]> --pue 1.1 --carbon-intensity 0.05

    # Headline number from the committed run shape (reads specs for D_dataset):
    python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT"

    # Override run shape (e.g. a longer schedule or different cluster):
    python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT" \\
        --max-iter 40000 --nodes 8 --gpus-per-node 8 --batch-per-rank 64

    # Just the formula, no spec/info.json reads (give D directly):
    python scripts/estimate_training_compute.py --examples 12300000

    # Use exact CMR post-filter window counts (after the filter caches exist):
    python scripts/estimate_training_compute.py --root "$OPENH_SURGICAL_ROOT" \\
        --from-filter-cache --num-frames 13
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 6ND constant from the EU AI Act methodology doc.
FLOP_PER_PARAM_PER_EXAMPLE = 6

# Committed run defaults (action_fdm_open_h_sft_nano.toml + slurm_train.sbatch).
DEFAULT_N_PARAMS = 8_000_000_000  # Qwen3-VL-8B backbone (total params for 6ND).
DEFAULT_BATCH_PER_RANK = 64  # max_samples_per_batch in the experiment.
DEFAULT_NODES = 6
DEFAULT_GPUS_PER_NODE = 8
DEFAULT_MAX_ITER = 20_000
DEFAULT_NUM_FRAMES = 13  # 1 context + 12 prediction.
# Slurm --job-name in slurm_train.sbatch (the resubmit chain shares this name).
DEFAULT_SACCT_NAME = "healthcareeng_holoscan-cosmos3.openh44d"

# Cosmos-Predict2.5 Open-H reference point (for the comparison line).
PREDICT25_FLOP = 1.48e17
PREDICT25_N = 2_000_000_000

INFO_FILENAME = "meta/info.json"
EU_AI_ACT_THRESHOLD = 1e25

# --- Energy / CO2 defaults (the genuinely external assumptions) ---
# H100 SXM board power (TDP). The committed run uses --constraint=h100.
DEFAULT_GPU_TDP_W = 700.0
# Fraction of TDP actually drawn on average during training (board-level).
# Large-model SFT typically sits ~0.6-0.75 of TDP sustained; 0.70 is a
# transparent mid default. Override with measured nvidia-smi power if available.
DEFAULT_GPU_UTIL = 0.70
# Power Usage Effectiveness: datacenter overhead multiplier (cooling, etc.).
DEFAULT_PUE = 1.2
# Grid carbon intensity (kg CO2e per kWh). 0.35 ~ world average electricity.
# Use ~0.05 for hydro/nuclear-heavy DCs; ~0.385 for US average.
DEFAULT_CARBON_KG_PER_KWH = 0.35
# Per-iteration wall-clock seconds (one optimizer step, all ranks in parallel).
# There is NO logged throughput for this run yet, so this is an explicit
# placeholder assumption for the --from-config energy path. A 8B MoT forward
# dynamics step at 480p with bs64/rank is heavy; ~6 s/iter is a deliberately
# rough mid guess. REPLACE with a measured value (--sec-per-iter) once the
# iter_speed callback logs real numbers, or use measured GPU-hours
# (--gpu-hours) directly.
DEFAULT_SEC_PER_ITER = 6.0


def _read_info_total_frames(dataset_path: Path) -> int | None:
    info_path = dataset_path / INFO_FILENAME
    try:
        with info_path.open() as f:
            info = json.load(f)
    except Exception:  # noqa: BLE001
        return None
    tf = info.get("total_frames")
    return int(tf) if tf is not None else None


def _is_cmr(embodiment: str) -> bool:
    return embodiment == "cmr_versius"


def _read_filter_cache_count(dataset_path: Path, num_frames: int) -> int | None:
    """Best-effort: sum post-filter step counts from CMR filter cache(s).

    The cache filename embeds a hash of action_delta_indices; rather than
    recompute that hash here, we scan meta/ for ``cmr_filter_cache_*`` files and
    read whichever records a per-episode filtered step list / count. Returns
    None if no usable cache is found (caller falls back to --cmr-keep-frac).
    """
    meta = dataset_path / "meta"
    if not meta.is_dir():
        return None
    best: int | None = None
    for cache in sorted(meta.glob("cmr_filter_cache_*")):
        try:
            with cache.open() as f:
                data = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        total = data.get("total_filtered_steps")
        if isinstance(total, int):
            best = total if best is None else max(best, total)
            continue
        # Fallback: sum lengths of any per-episode filtered index lists.
        eps = data.get("filtered_episodes") or data.get("episodes")
        if isinstance(eps, dict):
            s = 0
            for v in eps.values():
                if isinstance(v, list):
                    s += len(v)
                elif isinstance(v, int):
                    s += v
            if s:
                best = s if best is None else max(best, s)
    return best


def _gather_specs(root: str | None):
    """Yield (path, embodiment_str) from OPEN_H_DATASET_SPECS.

    Imports cosmos_framework lazily so the pure-formula path (``--examples``)
    works without the patched framework installed.
    """
    from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
    from cosmos_framework.data.vfm.action.gr00t_dreams.groot_configs import (
        get_open_h_multi_train_specs,
    )

    for spec in get_open_h_multi_train_specs(base_path=root):
        emb = spec["embodiment"]
        emb = emb.value if isinstance(emb, EmbodimentTag) else emb
        yield Path(spec["path"]), emb


def compute_d_dataset(
    root: str | None,
    *,
    cmr_keep_frac: float,
    from_filter_cache: bool,
    num_frames: int,
    verbose: bool,
) -> tuple[int, list[str]]:
    """Sum unique training windows across the spec mixture (epoch size)."""
    notes: list[str] = []
    total = 0
    n_specs = 0
    n_missing = 0
    cmr_frames = 0
    cmr_windows = 0
    for path, emb in _gather_specs(root):
        n_specs += 1
        frames = _read_info_total_frames(path)
        if frames is None:
            n_missing += 1
            if verbose:
                print(f"  [warn] no info.json::total_frames for {path}")
            continue
        if _is_cmr(emb):
            cmr_frames += frames
            windows = None
            if from_filter_cache:
                windows = _read_filter_cache_count(path, num_frames)
            if windows is None:
                windows = int(round(frames * cmr_keep_frac))
            cmr_windows += windows
            total += windows
        else:
            total += frames
    if n_missing:
        notes.append(
            f"{n_missing}/{n_specs} leaves missing info.json::total_frames "
            f"(excluded from D_dataset; re-run after staging completes)"
        )
    if cmr_frames:
        eff = (cmr_windows / cmr_frames) if cmr_frames else 0.0
        how = "filter-cache" if from_filter_cache else f"--cmr-keep-frac={cmr_keep_frac:g}"
        notes.append(
            f"CMR windows = {cmr_windows:,} of {cmr_frames:,} frames "
            f"(effective keep {eff:.3f}, via {how})"
        )
    return total, notes


# JobIDRaw + -X exclude .batch/.extern/.<step> subrows so each allocation is
# counted once. ElapsedRaw is integer seconds.
_SACCT_FIELDS = "JobID,JobIDRaw,Elapsed,ElapsedRaw,NNodes,AllocNodes,State"


def _run_sacct(selector: list[str], what: str) -> str:
    """Run ``sacct -X -n -P`` with the given selector flags; return stdout."""
    try:
        return subprocess.check_output(
            ["sacct", *selector, "-X", "-n", "-P", "-o", _SACCT_FIELDS],
            stderr=subprocess.STDOUT,
        ).decode()
    except FileNotFoundError as e:  # noqa: BLE001
        raise SystemExit(f"sacct not found ({e}); run on a Slurm login node or pass --gpu-hours")
    except subprocess.CalledProcessError as e:  # noqa: BLE001
        raise SystemExit(f"sacct failed for {what}: {e.output.decode().strip()}")


def _sum_sacct_rows(out: str, gpus_per_node: int) -> tuple[float, int, list[str]]:
    """Parse ``sacct`` rows -> (total_gpu_hours, n_rows, per-row detail lines)."""
    total_gpu_hours = 0.0
    rows = 0
    detail: list[str] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jobid, _raw, elapsed, elapsed_raw, nnodes, alloc_nodes, state = parts[:7]
        try:
            secs = int(elapsed_raw)
        except ValueError:
            continue
        # Prefer AllocNodes (actually-allocated); fall back to NNodes.
        try:
            nodes = int(alloc_nodes) if alloc_nodes and alloc_nodes != "0" else int(nnodes)
        except ValueError:
            nodes = int(nnodes) if nnodes.isdigit() else 0
        gh = (secs / 3600.0) * nodes * gpus_per_node
        total_gpu_hours += gh
        rows += 1
        detail.append(f"    {jobid}: {elapsed} x {nodes} nodes x {gpus_per_node} = {gh:,.1f} GPU-h [{state}]")
    return total_gpu_hours, rows, detail


def gpu_hours_from_sacct(job_id: str, gpus_per_node: int) -> tuple[float, str]:
    """Sum GPU-hours for a Slurm job id (and all its requeued/array tasks).

    GPU-hours = sum over allocation rows of ElapsedRaw(s)/3600 * AllocNodes *
    gpus_per_node. A requeued run shares one job id and shows multiple rows; a
    resubmit chain gets new ids (pass them comma-separated, or use
    ``gpu_hours_from_sacct_name``). Returns (gpu_hours, human_description).
    """
    out = _run_sacct(["-j", str(job_id)], f"job {job_id}")
    total, rows, detail = _sum_sacct_rows(out, gpus_per_node)
    if rows == 0:
        raise SystemExit(f"sacct returned no usable rows for job {job_id}")
    desc = f"GPU-hours from sacct (job {job_id}, {rows} alloc row(s)):\n" + "\n".join(detail)
    return total, desc


def gpu_hours_from_sacct_name(
    job_name: str,
    gpus_per_node: int,
    *,
    since: str | None = None,
    states: str | None = None,
) -> tuple[float, str]:
    """Sum GPU-hours across ALL jobs matching a Slurm job NAME.

    This is the no-id workflow for resubmit chains: every ``sbatch`` in the run
    shares ``--job-name``, so one query captures the whole chain. ``since`` maps
    to ``sacct -S`` (e.g. ``2026-06-20``) and ``states`` to ``-s`` (e.g.
    ``COMPLETED,TIMEOUT``) to fence off unrelated same-name jobs.

    NOTE: ``sacct --name`` only searches YOUR jobs within the accounting window;
    pass ``--sacct-since`` if the run started beyond the default lookback.
    Returns (gpu_hours, human_description).
    """
    selector = ["--name", job_name]
    if since:
        selector += ["-S", since]
    if states:
        selector += ["-s", states]
    out = _run_sacct(selector, f"name={job_name}")
    total, rows, detail = _sum_sacct_rows(out, gpus_per_node)
    if rows == 0:
        raise SystemExit(
            f"sacct returned no rows for name={job_name}"
            f"{f' since {since}' if since else ''}. Check the name/--sacct-since, "
            f"or pass explicit ids via --from-sacct."
        )
    filt = job_name + (f", since {since}" if since else "") + (f", states {states}" if states else "")
    desc = f"GPU-hours from sacct (name={filt}; {rows} job(s)):\n" + "\n".join(detail)
    return total, desc


def estimate_energy_co2(
    *,
    gpu_hours: float,
    gpu_tdp_w: float,
    gpu_util: float,
    pue: float,
    carbon_kg_per_kwh: float,
) -> dict:
    """Energy-first estimate: kWh and CO2e from GPU-hours and factors.

    Energy(kWh) = GPU_hours x (TDP_kW x util) x PUE
    CO2e(kg)    = Energy(kWh) x carbon_intensity(kgCO2e/kWh)
    """
    gpu_kw = (gpu_tdp_w / 1000.0) * gpu_util
    energy_kwh = gpu_hours * gpu_kw * pue
    co2_kg = energy_kwh * carbon_kg_per_kwh
    return {
        "gpu_hours": gpu_hours,
        "gpu_kw": gpu_kw,
        "energy_kwh": energy_kwh,
        "energy_mwh": energy_kwh / 1000.0,
        "co2_kg": co2_kg,
        "co2_t": co2_kg / 1000.0,
    }


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Model size.
    p.add_argument("--n-params", type=float, default=DEFAULT_N_PARAMS,
                   help=f"total model parameters N for 6ND (default {DEFAULT_N_PARAMS:.0e})")
    # Run shape -> D_seen.
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help=f"optimizer steps (default {DEFAULT_MAX_ITER})")
    p.add_argument("--nodes", type=int, default=DEFAULT_NODES,
                   help=f"number of nodes (default {DEFAULT_NODES})")
    p.add_argument("--gpus-per-node", type=int, default=DEFAULT_GPUS_PER_NODE,
                   help=f"GPUs per node (default {DEFAULT_GPUS_PER_NODE})")
    p.add_argument("--batch-per-rank", type=int, default=DEFAULT_BATCH_PER_RANK,
                   help=f"examples per rank per step = max_samples_per_batch "
                        f"(default {DEFAULT_BATCH_PER_RANK})")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="gradient accumulation steps (default 1)")
    # Direct D override (pure formula, no framework import).
    p.add_argument("--examples", type=float, default=None,
                   help="use this D directly (training examples); skips run-shape math")
    # D_dataset (epoch size) controls.
    p.add_argument("--root", default=None,
                   help="OPENH_SURGICAL_ROOT to re-root specs and read info.json totals")
    p.add_argument("--cmr-keep-frac", type=float, default=1.0,
                   help="approx fraction of CMR frames kept after clutch filtering "
                        "(default 1.0 = upper bound; set ~0.5 for a realistic guess)")
    p.add_argument("--from-filter-cache", action="store_true",
                   help="read exact CMR post-filter window counts from meta/ caches")
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                   help=f"action/video frames per window (default {DEFAULT_NUM_FRAMES})")
    p.add_argument("--no-dataset", action="store_true",
                   help="skip D_dataset computation (don't read specs/info.json)")
    # --- Combined mode ---
    p.add_argument("--all", dest="all_sections", action="store_true",
                   help="print EVERYTHING at once: training compute (6ND) + "
                        "D_dataset/epochs + energy (kWh/MWh) + CO2e. Implies --energy.")
    # --- Energy / CO2 ---
    p.add_argument("--energy", action="store_true",
                   help="also estimate training energy (kWh/MWh) and CO2e (kg/t)")
    p.add_argument("--gpu-hours", type=float, default=None,
                   help="measured total GPU-hours (overrides the from-config estimate)")
    p.add_argument("--from-sacct", metavar="JOBID", default=None,
                   help="query Slurm sacct for this job id and use its GPU-hours "
                        "(ElapsedRaw x AllocNodes x gpus_per_node). For resubmitted "
                        "runs, pass a comma-separated list of ids to sum them.")
    p.add_argument("--sacct-name", metavar="NAME", nargs="?", default=None,
                   const=DEFAULT_SACCT_NAME,
                   help="sum GPU-hours across ALL Slurm jobs with this --job-name "
                        "(captures the whole resubmit chain; NO ids needed). Bare "
                        f"--sacct-name uses '{DEFAULT_SACCT_NAME}'.")
    p.add_argument("--sacct-since", metavar="YYYY-MM-DD", default=None,
                   help="sacct -S lookback start (use with --sacct-name if the run "
                        "predates the default accounting window)")
    p.add_argument("--sacct-states", metavar="STATES", default=None,
                   help="sacct -s state filter for --sacct-name, e.g. "
                        "COMPLETED,TIMEOUT (excludes FAILED/CANCELLED retries)")
    p.add_argument("--sec-per-iter", type=float, default=DEFAULT_SEC_PER_ITER,
                   help=f"wall-clock seconds per optimizer step for the from-config "
                        f"GPU-hours estimate (default {DEFAULT_SEC_PER_ITER}; "
                        f"ASSUMPTION - replace with a measured value)")
    p.add_argument("--gpu-tdp-w", type=float, default=DEFAULT_GPU_TDP_W,
                   help=f"per-GPU board power in W (default {DEFAULT_GPU_TDP_W}, H100 SXM)")
    p.add_argument("--gpu-util", type=float, default=DEFAULT_GPU_UTIL,
                   help=f"avg fraction of TDP drawn (default {DEFAULT_GPU_UTIL})")
    p.add_argument("--pue", type=float, default=DEFAULT_PUE,
                   help=f"datacenter PUE (default {DEFAULT_PUE})")
    p.add_argument("--carbon-intensity", type=float, default=DEFAULT_CARBON_KG_PER_KWH,
                   help=f"grid carbon intensity kgCO2e/kWh (default "
                        f"{DEFAULT_CARBON_KG_PER_KWH}, ~world avg)")
    args = p.parse_args()

    # --all is a convenience that turns on every section at once.
    if args.all_sections:
        args.energy = True
        args.no_dataset = False

    N = args.n_params

    # --- D_seen: examples the configured run processes ---
    world_size = args.nodes * args.gpus_per_node
    if args.examples is not None:
        d_seen = args.examples
        seen_desc = f"D given directly = {d_seen:,.0f} examples"
    else:
        ex_per_step = args.batch_per_rank * world_size * args.grad_accum
        d_seen = ex_per_step * args.max_iter
        seen_desc = (
            f"D_seen = batch_per_rank({args.batch_per_rank}) x world_size({world_size}) "
            f"x grad_accum({args.grad_accum}) x steps({args.max_iter:,})\n"
            f"       = {ex_per_step:,} examples/step x {args.max_iter:,} steps"
        )

    c_seen = FLOP_PER_PARAM_PER_EXAMPLE * N * d_seen

    print("=" * 70)
    print("Cumulative training compute  (EU AI Act 6ND approximation)")
    print("=" * 70)
    print(f"N (total params) : {N:,.0f}  ({_fmt(N)})")
    print()
    print("[Compute basis: D_seen = examples the run actually processes]")
    print(f"  {seen_desc}")
    print(f"  D_seen           : {d_seen:,.0f}  ({_fmt(d_seen)}) examples")
    print(f"  C = 6 x N x D    : {_fmt(c_seen)} FLOP")
    print()

    # --- D_dataset: unique windows per epoch (optional) ---
    if not args.no_dataset and args.examples is None:
        try:
            d_dataset, notes = compute_d_dataset(
                args.root,
                cmr_keep_frac=args.cmr_keep_frac,
                from_filter_cache=args.from_filter_cache,
                num_frames=args.num_frames,
                verbose=True,
            )
        except ImportError as e:
            print(f"[note] could not import cosmos_framework specs ({e}); "
                  f"skipping D_dataset. Run inside the patched venv, or pass "
                  f"--no-dataset / --examples.")
            d_dataset, notes = 0, []
        if d_dataset:
            c_dataset = FLOP_PER_PARAM_PER_EXAMPLE * N * d_dataset
            epochs = d_seen / d_dataset
            print("[Epoch basis: D_dataset = unique training windows in the mixture]")
            print(f"  D_dataset        : {d_dataset:,.0f}  ({_fmt(d_dataset)}) windows")
            print(f"  C for 1 epoch    : {_fmt(c_dataset)} FLOP")
            print(f"  run covers       : {epochs:.3f} epochs over the mixture")
            for n in notes:
                print(f"  note: {n}")
            print()

    # --- Energy / CO2 (optional) ---
    if args.energy:
        world_size_e = args.nodes * args.gpus_per_node
        measured = True
        if args.sacct_name is not None:
            gpu_hours, gh_desc = gpu_hours_from_sacct_name(
                args.sacct_name,
                args.gpus_per_node,
                since=args.sacct_since,
                states=args.sacct_states,
            )
        elif args.from_sacct is not None:
            job_ids = [j.strip() for j in str(args.from_sacct).split(",") if j.strip()]
            gpu_hours = 0.0
            descs: list[str] = []
            for jid in job_ids:
                gh, d = gpu_hours_from_sacct(jid, args.gpus_per_node)
                gpu_hours += gh
                descs.append(d)
            gh_desc = "\n".join(descs)
            if len(job_ids) > 1:
                gh_desc += f"\n  total across {len(job_ids)} jobs = {gpu_hours:,.1f} GPU-hours"
        elif args.gpu_hours is not None:
            gpu_hours = args.gpu_hours
            gh_desc = f"measured GPU-hours = {gpu_hours:,.1f}"
        else:
            measured = False
            wall_hours = args.sec_per_iter * args.max_iter / 3600.0
            gpu_hours = wall_hours * world_size_e
            gh_desc = (
                f"GPU-hours = sec_per_iter({args.sec_per_iter:g}) x "
                f"steps({args.max_iter:,}) / 3600 x world_size({world_size_e})\n"
                f"             = {wall_hours:,.1f} wall-hours x {world_size_e} GPUs"
            )
        e = estimate_energy_co2(
            gpu_hours=gpu_hours,
            gpu_tdp_w=args.gpu_tdp_w,
            gpu_util=args.gpu_util,
            pue=args.pue,
            carbon_kg_per_kwh=args.carbon_intensity,
        )
        print("[Energy & CO2  (energy-first: GPU-hours x power x PUE x intensity)]")
        print(f"  {gh_desc}")
        print(f"  per-GPU power    : {args.gpu_tdp_w:.0f} W TDP x {args.gpu_util:.2f} util "
              f"= {e['gpu_kw'] * 1000:.0f} W ({e['gpu_kw']:.3f} kW)")
        print(f"  PUE              : {args.pue:g}")
        print(f"  carbon intensity : {args.carbon_intensity:g} kgCO2e/kWh")
        print(f"  -> energy        : {e['energy_kwh']:,.0f} kWh  ({e['energy_mwh']:.2f} MWh)")
        print(f"  -> emissions     : {e['co2_kg']:,.0f} kgCO2e  ({e['co2_t']:.3f} tCO2e)")
        if not measured:
            print("  note: GPU-hours derived from an ASSUMED sec_per_iter; replace with "
                  "--from-sacct <jobid> or --gpu-hours <measured> for a reportable figure.")
        print()

    # --- Comparison + threshold context ---
    ratio = c_seen / PREDICT25_FLOP
    print("[Context]")
    print(f"  Cosmos-Predict2.5 Open-H ref : {_fmt(PREDICT25_FLOP)} FLOP "
          f"(N={PREDICT25_N:.0e}, D~={PREDICT25_FLOP / (6 * PREDICT25_N):,.0f} examples)")
    print(f"  this run / Predict2.5        : {ratio:.2f}x")
    print(f"  EU AI Act GPAISR threshold   : {_fmt(EU_AI_ACT_THRESHOLD)} FLOP")
    print(f"  this run / threshold         : {c_seen / EU_AI_ACT_THRESHOLD:.2e}  "
          f"({'BELOW' if c_seen < EU_AI_ACT_THRESHOLD else 'AT/ABOVE'} threshold)")
    print("=" * 70)


if __name__ == "__main__":
    main()
