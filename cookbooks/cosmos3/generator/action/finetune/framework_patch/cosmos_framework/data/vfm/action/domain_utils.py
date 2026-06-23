# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Domain ID helpers for cross-embodiment action datasets.

This is the Cosmos Framework ``domain_utils`` with the Open-H surgical
embodiments added (ported from the Cosmos3 internal surgical stack).  All
surgical embodiments share a single domain id (31, with CMR Versius at 30)
so the model treats them as one cross-embodiment surgical family — mirroring
the Cosmos-H-Surgical-Simulator training setup where the 44D action space is
unified across all Open-H subsets.  The non-surgical entries are unchanged
from the upstream Cosmos Framework table.
"""

EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "no_action": 0,
    "av": 1,
    "camera_pose": 2,
    "hand_pose": 3,
    "pusht": 4,
    "libero": 5,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,  # Both Droid and RoboMIND-Franka are using robotiq and franka
    "embodiment_b": 9,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "embodiment_c_gripper": 15,
    "embodiment_c_gripper_ext": 15,
    "fractal": 20,
    # =====================================================================
    # Open-H surgical embodiments (ported from the Cosmos3 internal
    # gr00t_dreams EmbodimentTag registry). All surgical embodiments share
    # a single domain id so the model treats them as one cross-embodiment
    # surgical family, mirroring the Cosmos-H-Surgical-Simulator training
    # setup where the 44D action space is unified across all Open-H subsets.
    # =====================================================================
    "cmr_versius": 30,
    "dvrk": 31,  # Deprecated alias for jhu_dvrk_mono — preserved for pickle compat
    "suturebot": 31,
    "jhu_dvrk_mono": 31,
    "dvrk_ucb": 31,
    "hamlyn_30hz": 31,
    "dvrk_ucsd": 31,
    "ustc_torin": 31,
    "dvrk_obuda": 31,
    "rob_surgical": 31,
    "dvrk_stanford_real": 31,
    "polyu_sim": 31,
    "moon": 31,
    "jhu_lscr_miracle": 31,
    "jhu_lscr_smarts": 31,
    "tud_tundra": 31,
    "turin_mitic_ex_vivo": 31,
    # Added for the cosmos3-h-s-s Open-H mixture (union with the sean repo
    # surgical dataset list): single-arm KUKA (STAR-IL / IMERSE) and the
    # Virtual Incision MIRA delta-command embodiment.
    "jhu_imerse": 31,
    "virtual_incision_mira": 31,
}


EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "embodiment_b": 30,
    "agibotworld": 29,
    "embodiment_c_gripper": 29,
    "embodiment_c_gripper_ext": 29,
    "fractal": 10,
    # NOTE: ``libero`` (7/10/13 depending on ``rotation_space``) and ``hand_pose``
    # (variable with ``keypoint_option`` and ``rotation_format``) are absent
    # because their raw width is set per-dataset at construction time. Inference
    # in inverse_dynamics/policy modes is not supported for these domains until
    # canonical widths are added here.
    #
    # NOTE: the Open-H surgical embodiments are likewise absent here: their raw
    # action width varies per embodiment (6D Moon … 36D Rob Surgical) and is
    # resolved by ``OpenHMixedLeRobotDataset`` / ``construct_modality_config_and_transforms``
    # at construction time, then zero-padded to the unified 44D ceiling.
}


def get_domain_id(embodiment_type: str) -> int:
    """Get the domain ID for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_DOMAIN_ID.keys())}"
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_action_dim(embodiment_type: str) -> int:
    """Get the raw action dimension for a given embodiment type."""
    key = embodiment_type.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise KeyError(
            f"Unknown embodiment type: {embodiment_type!r}. "
            f"Available embodiments: {sorted(EMBODIMENT_TO_RAW_ACTION_DIM.keys())}"
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def is_valid_domain_name(embodiment_type: str) -> bool:
    """Check if the given embodiment type is recognized."""
    key = embodiment_type.lower().strip()
    return key in EMBODIMENT_TO_RAW_ACTION_DIM
