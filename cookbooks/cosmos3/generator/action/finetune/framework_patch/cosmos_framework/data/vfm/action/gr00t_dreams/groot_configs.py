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

from cosmos_framework.data.vfm.action.gr00t_dreams.data.dataset import ModalityConfig
from cosmos_framework.data.vfm.action.gr00t_dreams.data.embodiment_tags import EmbodimentTag
from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.base import ComposedModalityTransform
from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.concat import ConcatTransform
from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.state_action import (
    ActionKeyConfig,
    CMRVersiusRelativeActionTransform,
    GenericRelativeActionTransform,
    RelativeActionTransform,
    StateActionToTensor,
    StateActionTransform,
)
from cosmos_framework.data.vfm.action.gr00t_dreams.data.transform.video import (
    VideoCrop,
    VideoResize,
    VideoToTensor,
)

# =============================================================================
# Open-H Multi-Embodiment Registry
# =============================================================================
# Each entry defines the dataset-specific parameters for a single embodiment.
# Action keys match the gr00t-H action space definitions. State keys are loaded
# for the model's __key__ field but are NOT appended to the action vector
# (except for CMR Versius which has its own CMR-specific path above).
#
# Video: Single camera per dataset (monocular) for video generation.
# Action: Raw parquet dimensions (no rot6d conversion; model learns raw format).
# State: Reference state at t=0 for context.
#
# The max unified action dimension is 44D (CMR Versius with state conditioning).
# All other datasets are zero-padded to 44D by MixedLeRobotDataset.
#
# !!! SETUP-TIME AUDIT REQUIRED (this registry is authored OFFLINE) !!!
# This registry was first written against the C-H-S-S internally re-converted
# LeRobot datasets. The PUBLIC open-h-embodiment release uses DIFFERENT
# modality keys. The ``video_keys`` below have been updated to the public
# on-disk camera folders (e.g. ``video.endoscope.left`` for
# ``observation.images.endoscope.left``), but the ``state_keys`` /
# ``action_keys`` / ``action_key_configs`` may still differ from each
# dataset's actual ``meta/modality.json`` (key names, quaternion order,
# gripper presence). Run ``scripts/audit_openh_action_schemas.py`` and FIX
# any mismatch BEFORE training — a wrong key fails at the first batch, and a
# wrong quaternion order silently corrupts the action representation.
# =============================================================================


# Helper: create a standard dual-arm EEF + gripper action config (most common pattern)
def _dual_arm_eef_configs(
    pose1_key: str,
    grip1_key: str,
    pose2_key: str,
    grip2_key: str,
    state_pose1: str,
    state_pose2: str,
    input_rot: str = "quat",
    ref_rot: str = "quat",
    input_quat: str = "xyzw",
    ref_quat: str = "xyzw",
) -> dict[str, ActionKeyConfig]:
    return {
        pose1_key: ActionKeyConfig(
            rep="rel_xyz_rot6d",
            state_key=state_pose1,
            input_rotation_format=input_rot,
            reference_rotation_format=ref_rot,
            input_quat_order=input_quat,
            reference_quat_order=ref_quat,
        ),
        grip1_key: ActionKeyConfig(rep="absolute"),
        pose2_key: ActionKeyConfig(
            rep="rel_xyz_rot6d",
            state_key=state_pose2,
            input_rotation_format=input_rot,
            reference_rotation_format=ref_rot,
            input_quat_order=input_quat,
            reference_quat_order=ref_quat,
        ),
        grip2_key: ActionKeyConfig(rep="absolute"),
    }


EMBODIMENT_REGISTRY: dict[str, dict] = {
    # -----------------------------------------------------------------
    # dVRK JHU Monocular  (left endoscope only, 30Hz storage → stride 3
    # for 10fps effective training rate).
    #
    # Action: REL_XYZ_ROT6D for poses (quat → 9D), ABSOLUTE for grippers
    # Raw per-arm channels: xyz(3) + quat_xyzw(4) + gripper(1) = 8D
    # Post-transform per-arm channels: xyz_rel(3) + rot6d_rel(6) + gripper(1) = 10D
    # Dual-arm → 20D concatenated action, zero-padded to MAX_ACTION_DIM=44.
    #
    # NOTE: this single "jhu_dvrk_mono" entry subsumes the previous
    # stereo-oriented "dvrk" registry entry. Cosmos is monocular-only
    # (reads only ``video.endoscope_left``), and the two entries were
    # byte-identical aside from comments. ``EmbodimentTag.DVRK`` is
    # preserved in the enum for pickle/checkpoint compatibility but
    # should be treated as an alias for ``JHU_DVRK_MONO``.
    # -----------------------------------------------------------------
    "jhu_dvrk_mono": {
        "timestep_interval": 3,
        # On-disk (open-h-embodiment): observation.images.endoscope.left
        # (IMERSE / LCSR ARCADE are dVRK-Si stereo; we read the left endoscope).
        # AUDIT: confirm the modality key + psm1/psm2 pose/gripper keys per dataset.
        "video_keys": ["video.endoscope.left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        # Training/inference resolution: 544 H x 960 W.
        #
        # WHY 544 (NOT 540) for height
        # -----------------------------
        # The source data on disk is 540 H x 960 W
        # (``LeRobot_540x960`` in ``/lustre/.../JHU_data_jpeg100_noacc_clean++/``),
        # but Cosmos-Predict2.5's pipeline requires both H and W to be
        # divisible by 16 (8x VAE compression × 2x DiT spatial patch
        # size).  540 % 16 == 12 → fails.  Picking 544 = 16 × 34 (the
        # nearest valid grid above 540) gives:
        #   * 16:9-ish aspect 17:30 ≈ 0.567 (vs source 9:16 = 0.5625);
        #     0.7 % aspect drift, visually imperceptible.
        #   * VideoResize stretches the 95 %-cropped frame from
        #     513 × 912 to 544 × 960 (asymmetric 1.060 × 1.053);
        #     stretch is uniform within each axis, no center crop.
        # The alternative 528 (squish 540 → 528 = -2.2 %) introduces
        # more drift and was not chosen.
        #
        # NOTE on memory: 544 x 960 = 8160 latent tokens (after VAE +
        # patch) per frame, ~3.5x more than the prior 288x512 setup
        # (attention memory ~12x).  Per-GPU batch sizes in the
        # experiment configs are tuned down accordingly; see the
        # per-experiment comments in
        # ``exp_2B_action_conditioned_rectify_flow_gr00t.py``,
        # ``exp_action_warmup.py``, and ``exp_action_self_forcing.py``.
        #
        # NOTE on Cosmos-Predict2.5 alignment: 544x960 is *not* a
        # registered preset of ``VIDEO_RES_SIZE_INFO``; the closest
        # 9:16 preset is ``720p / 9,16`` = (720, 1280).  We stay at
        # 544x960 deliberately to minimise resize cost vs the source —
        # see ``agent_chats/cursor_resize_investigation_720x960.md``
        # for the trade-off analysis.
        "video_width": 960,
        "video_height": 544,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # dVRK UCB Debridement (~30Hz → stride 3)
    # modality.json video key: "camera_left" (not "endoscope")
    # Updated (gr00t-H 9e25db4): Cartesian EEF pose actions with REL_XYZ_ROT6D,
    # joint-angle state channels for normalization, and cartesian pose as
    # pass-through for the action reference frame (not normalized/concatenated).
    # -----------------------------------------------------------------
    "dvrk_ucb": {
        "timestep_interval": 3,
        # On-disk: observation.images.left (UCB debridement; cams 'left'/'right').
        "video_keys": ["video.left"],
        "state_keys": [
            # Joint-angle + gripper state (normalized & concatenated as model input)
            "state.psm1_joints",
            "state.psm1_gripper",
            "state.psm2_joints",
            "state.psm2_gripper",
            # Cartesian pose (pass-through for REL_XYZ_ROT6D reference only)
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        # Pass-through state keys: loaded and used by GenericRelativeActionTransform
        # as the reference frame for REL_XYZ_ROT6D conversion, but NOT normalized
        # or concatenated into the model's state input vector.
        "pass_through_state_keys": [
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Hamlyn 30Hz (30Hz → stride 3, wxyz quaternions)
    # modality.json video key: "endoscope" (not "endoscope_left")
    # -----------------------------------------------------------------
    "hamlyn_30hz": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.left_arm_pose",
            "state.left_arm_gripper",
            "state.right_arm_pose",
            "state.right_arm_gripper",
        ],
        "action_keys": [
            "action.left_arm_pose",
            "action.left_arm_gripper",
            "action.right_arm_pose",
            "action.right_arm_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.left_arm_pose",
            "action.left_arm_gripper",
            "action.right_arm_pose",
            "action.right_arm_gripper",
            "state.left_arm_pose",
            "state.right_arm_pose",
            input_quat="wxyz",
            ref_quat="wxyz",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # UCSD Surgical Learning (30Hz → stride 3, wxyz quaternions)
    # modality.json video key: "camera_left" (not "endoscope")
    # -----------------------------------------------------------------
    "dvrk_ucsd": {
        "timestep_interval": 3,
        # On-disk: observation.images.left (UCSD; cams 'left'/'right').
        "video_keys": ["video.left"],
        "state_keys": [
            "state.psm_retraction_pose",
            "state.psm_retraction_gripper",
            "state.psm_cutter_pose",
            "state.psm_cutter_gripper",
        ],
        "action_keys": [
            "action.psm_retraction_pose",
            "action.psm_retraction_gripper",
            "action.psm_cutter_pose",
            "action.psm_cutter_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm_retraction_pose",
            "action.psm_retraction_gripper",
            "action.psm_cutter_pose",
            "action.psm_cutter_gripper",
            "state.psm_retraction_pose",
            "state.psm_cutter_pose",
            input_quat="wxyz",
            ref_quat="wxyz",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # USTC Torin (24Hz → stride 2)
    # NOTE: The old knot_tying_all / needle_handover_all / needle_pickup_all
    # datasets have joint-space data (left_joints, right_joints) in modality.json,
    # NOT pose data (left_pose, right_pose). The newer quat_merged datasets
    # (exp86) have pose keys. Using joint keys for backward compatibility.
    # -----------------------------------------------------------------
    "ustc_torin": {
        "timestep_interval": 2,
        "video_keys": ["video.endoscope_left"],
        "state_keys": [
            "state.left_joints",
            "state.right_joints",
        ],
        "action_keys": [
            "action.left_joints",
            "action.right_joints",
        ],
        "action_key_configs": {
            "action.left_joints": ActionKeyConfig(rep="relative", state_key="state.left_joints"),
            "action.right_joints": ActionKeyConfig(rep="relative", state_key="state.right_joints"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Obuda dVRK (50Hz → stride 5)
    # -----------------------------------------------------------------
    "dvrk_obuda": {
        "timestep_interval": 5,
        # On-disk: observation.images.endoscope.left (Obuda dVRK stereo -> left).
        "video_keys": ["video.endoscope.left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Rob Surgical (4-arm, pose only, Euler rotation, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "rob_surgical": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.left_pose",
            "state.right_pose",
            "state.lap_pose",
            "state.aux_pose",
        ],
        "action_keys": [
            "action.left_pose",
            "action.right_pose",
            "action.lap_pose",
            "action.aux_pose",
        ],
        "action_key_configs": {
            "action.left_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.left_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.right_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.right_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.lap_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.lap_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
            "action.aux_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.aux_pose",
                input_rotation_format="euler",
                reference_rotation_format="euler",
            ),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Stanford Real dVRK (Euler rotation, 30Hz → stride 3)
    # -----------------------------------------------------------------
    "dvrk_stanford_real": {
        "timestep_interval": 3,
        # On-disk: observation.images.camera_left (Stanford real dVRK).
        "video_keys": ["video.camera_left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_rot="euler",
            ref_rot="euler",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # PolyU Simulated (single arm, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "polyu_sim": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.psm_cartesian_pose",
            "state.psm_gripper",
        ],
        "action_keys": [
            "action.psm_cartesian_pose",
            "action.psm_gripper",
        ],
        "action_key_configs": {
            "action.psm_cartesian_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm_cartesian_pose"),
            "action.psm_gripper": ActionKeyConfig(rep="absolute"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Moon Surgical (DELTA xyz only, ~30Hz → stride 3)
    # modality.json video key: "scope" (not "endoscope")
    # modality.json state keys: "right_arm_joints", "left_arm_joints" (not delta_xyz)
    # -----------------------------------------------------------------
    "moon": {
        "timestep_interval": 3,
        "video_keys": ["video.scope"],
        "state_keys": [
            "state.right_arm_joints",
            "state.left_arm_joints",
        ],
        "action_keys": [
            "action.right_arm_delta_xyz",
            "action.left_arm_delta_xyz",
        ],
        "action_key_configs": {
            "action.right_arm_delta_xyz": ActionKeyConfig(rep="delta"),
            "action.left_arm_delta_xyz": ActionKeyConfig(rep="delta"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # JHU LSCR MIRACLE (15Hz → stride 1, explicit xyzw quaternions)
    # -----------------------------------------------------------------
    "jhu_lscr_miracle": {
        "timestep_interval": 1,
        # On-disk: observation.images.left (LCSR MIRACLE; cams 'left'/'right').
        "video_keys": ["video.left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_quat="xyzw",
            ref_quat="xyzw",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # JHU LSCR SMARTS (10Hz → stride 1, explicit xyzw quaternions)
    # -----------------------------------------------------------------
    "jhu_lscr_smarts": {
        "timestep_interval": 1,
        # On-disk: observation.images.left (LCSR SMARTS; cams 'left'/'right').
        "video_keys": ["video.left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm1_gripper",
            "state.psm2_pose",
            "state.psm2_gripper",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
        ],
        "action_key_configs": _dual_arm_eef_configs(
            "action.psm1_pose",
            "action.psm1_gripper",
            "action.psm2_pose",
            "action.psm2_gripper",
            "state.psm1_pose",
            "state.psm2_pose",
            input_quat="xyzw",
            ref_quat="xyzw",
        ),
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # TUD TUNDRA (single arm UR5e, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "tud_tundra": {
        "timestep_interval": 3,
        "video_keys": ["video.laparoscope_left"],
        "state_keys": [
            "state.eef_pose",
            "state.gripper",
        ],
        "action_keys": [
            "action.eef_pose",
            "action.gripper",
        ],
        "action_key_configs": {
            "action.eef_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d", state_key="state.eef_pose", reference_quat_order="xyzw"
            ),
            "action.gripper": ActionKeyConfig(rep="absolute"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Turin MITIC (dual arm pose only, no grippers, ~30Hz → stride 3)
    # -----------------------------------------------------------------
    "turin_mitic_ex_vivo": {
        "timestep_interval": 3,
        # On-disk: observation.images.endoscope.left (Turin MITIC stereo -> left).
        "video_keys": ["video.endoscope.left"],
        "state_keys": [
            "state.psm1_pose",
            "state.psm2_pose",
        ],
        "action_keys": [
            "action.psm1_pose",
            "action.psm2_pose",
        ],
        "action_key_configs": {
            "action.psm1_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm1_pose"),
            "action.psm2_pose": ActionKeyConfig(rep="rel_xyz_rot6d", state_key="state.psm2_pose"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # JHU IMERSE STAR-IL (single KUKA arm, pose only, no gripper)
    # -----------------------------------------------------------------
    # Added for the cosmos3-h-s-s Open-H mixture (union with the sean repo
    # surgical dataset list). The sean repo's audited schema
    # (``surgical_action_schemas._star_il_layout``) models STAR-IL as a
    # single KUKA EEF pose (action[0:7], quat xyzw) with NO gripper action
    # (zero-filled). Here it maps to the registry's single-arm pattern:
    # ``rel_xyz_rot6d`` pose (xyz_rel(3) + rot6d_rel(6) = 9D), no gripper,
    # zero-padded to MAX_ACTION_DIM=44.
    #
    # ASSUMPTION (validate at setup via audit_openh_action_schemas.py): the
    # LeRobot ``modality.json`` exposes a named ``action.eef_pose`` /
    # ``state.eef_pose`` (xyz + quat_xyzw, 7D). If the converted dataset
    # uses a different key (e.g. ``action.action`` raw vector), adjust the
    # keys below to match the actual modality file before training.
    "jhu_imerse": {
        "timestep_interval": 3,
        # On-disk: observation.images.endoscope.left (STAR-IL KUKA, mono left).
        "video_keys": ["video.endoscope.left"],
        "state_keys": [
            "state.eef_pose",
        ],
        "action_keys": [
            "action.eef_pose",
        ],
        "action_key_configs": {
            "action.eef_pose": ActionKeyConfig(
                rep="rel_xyz_rot6d",
                state_key="state.eef_pose",
                input_quat_order="xyzw",
                reference_quat_order="xyzw",
            ),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
    # -----------------------------------------------------------------
    # Virtual Incision MIRA (delta-command pose + gripper, single arm)
    # -----------------------------------------------------------------
    # Added for the cosmos3-h-s-s Open-H mixture (union with the sean repo
    # surgical dataset list). The sean repo treats MIRA as a bespoke
    # ``mira_delta_single`` schema (``surgical_action_schemas.mira_delta_layout``):
    # the action is already a haptic delta-pose command (xyz + euler deltas)
    # plus an absolute gripper, NOT an absolute pose to be differenced.
    # We therefore use ``rep="delta"`` (pass-through, like Moon) for the
    # delta-pose channels and ``rep="absolute"`` for the gripper. No
    # ``state_key`` reference is needed for a delta channel.
    #
    # ASSUMPTION (validate at setup): the LeRobot ``modality.json`` exposes
    # ``action.delta_pose`` (xyz + euler/rot deltas) and ``action.gripper``
    # (1D, absolute). The exact width of the delta-pose channel must be
    # confirmed against the modality file; the action is zero-padded to
    # MAX_ACTION_DIM=44 regardless.
    "virtual_incision_mira": {
        "timestep_interval": 3,
        "video_keys": ["video.endoscope"],
        "state_keys": [
            "state.eef_pose",
        ],
        "action_keys": [
            "action.delta_pose",
            "action.gripper",
        ],
        "action_key_configs": {
            "action.delta_pose": ActionKeyConfig(rep="delta"),
            "action.gripper": ActionKeyConfig(rep="absolute"),
        },
        "video_width": 512,
        "video_height": 288,
        "modality_filename": "meta/modality.json",
        "normalization_mode": "mean_std",
    },
}

# Maximum unified action dimension across all embodiments.
# CMR Versius has 44D (30D actions + 14D state conditioning) = the largest.
# All other datasets are zero-padded to this dimension by MixedLeRobotDataset.
MAX_ACTION_DIM = 44


# =============================================================================
# Open-H Dataset Specifications  —  SINGLE SOURCE OF TRUTH
# =============================================================================
# This list defines EVERY Open-H dataset: path, embodiment tag, and mix_ratio.
# Everything else (the set of Open-H embodiment tags, the stats-file check in
# dataset.py, the experiment dataloader config) is DERIVED from here.
#
# cosmos3-h-s-s mixture — grounded in the PUBLIC Open-H-Embodiment tree
# --------------------------------------------------------------------
# Every path below points at the actual on-disk layout under
# ``_OPENH_SURGICAL_ROOT`` (verified against
# ``doc/open-h-embodiment_dataset_folder_structure.txt``), NOT the C-H-S-S
# internal re-converted mirror. Unified 44D action space (CMR Versius
# ceiling; smaller embodiments zero-padded).
#
# Scope decisions (see cookbook README for the full audit):
#   - Target: maximum non-synthetic surgical coverage that fits the 1-/2-arm
#     44D Cartesian-pose contract and has a usable endoscope/stereo/scope view.
#   - INCLUDED groups: CMR Versius (clinical), JHU (IMERSE + LCSR
#     ARCADE/MIRACLE/SMARTS + STAR-IL), Obuda dVRK, Stanford real dVRK,
#     Turin MITIC, UC Berkeley, UCSD, TUD TUNDRA (grasping_retraction),
#     Virtual Incision MIRA.
#   - REMOVED vs the earlier draft:
#       * USTC/Tuodao — NOT present in the public open-h-embodiment tree
#         (was only in C-H-S-S v1's internal mirror).
#       * Moon Surgical — dropped (delta-xyz only, no verified rotation; its
#         presence in C-H-S-S v1 does not by itself justify re-inclusion).
#       * Hamlyn/Imperial — dropped: the public release exposes only
#         ``color``/``depth``/``wrist_{left,right}`` cameras, NO endoscope
#         view, so it does not fit the endoscope-conditioned FD setup.
#   - EXCLUDED (unchanged): UTenn (video/seg/label only), Rob Surgical
#     (3-arm/27D), UIC (joint-only), Semaphor (no robot kinematics / TPV
#     only), HK PolyU (synthetic), UT Austin (colonoscopy), TUM/Balgrist/
#     CUHK/HKBU/ImFusion (ultrasound), SanoScience (synthetic), CMR
#     dry_box/peg_transfer (benchtop/unverified), UCSD retraction
#     dataset3/failurecase (unverified).
#
# Weighting strategy (cosmos3-internal recipe):
#   - CMR Versius: ~50% of training (mix_ratio sum = 4.0 across 4 procedures).
#   - All other embodiments: ~50% (mix_ratio sum ≈ 4.0), step-weighted by
#     frame count.
#
# IMPORTANT — two setup-time gates (this file is authored OFFLINE):
#   1. SCHEMA: the public open-h-embodiment LeRobot datasets use different
#      modality keys than the C-H-S-S re-converted mirror this registry was
#      first written for (e.g. video ``observation.images.endoscope.left``
#      -> ``video.endoscope.left``; action/state key names also differ).
#      ``video_keys`` below are set from the on-disk camera folders, but the
#      action/state keys in EMBODIMENT_REGISTRY MUST be validated and fixed
#      against each dataset's ``meta/modality.json`` via
#      ``scripts/audit_openh_action_schemas.py`` before training.
#   2. RATIOS: non-CMR ``mix_ratio``s are frame-count estimates from Table S1;
#      recompute from each dataset's ``meta/info.json::total_frames`` with
#      ``scripts/compute_openh_action_stats.py`` and re-normalize so the
#      non-CMR pool sums to ~4.0 (CMR stays at 50%).
# =============================================================================

# Base path — the public Open-H-Embodiment surgical tree on the EOS cluster.
#
#   /lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical
#
# Every spec below uses this single root with the ACTUAL on-disk leaf paths
# and directory names (verified against
# ``doc/open-h-embodiment_dataset_folder_structure.txt``). Override the root
# via ``base_path`` (``DATASET_PATH`` / ``OPENH_SURGICAL_ROOT``) if you stage
# the data elsewhere — note ``_rebase_specs`` re-roots by the FULL relative
# path under this constant (see ``get_open_h_multi_train_specs``).
_OPENH_SURGICAL_ROOT = "/lustre/fsw/healthcareeng_holoscan/datasets/open-h-embodiment/Surgical"
# Stanford real dVRK tree (on-disk path under _OPENH_SURGICAL_ROOT).
_STANFORD_BASE = f"{_OPENH_SURGICAL_ROOT}/stanford/collaborative_haptics_and_robotics_in_medicine_lab/real_robot_dvrk"
# JHU sub-trees.
_JHU_IMERSE = f"{_OPENH_SURGICAL_ROOT}/jhu/imerse"
_JHU_LCSR = f"{_OPENH_SURGICAL_ROOT}/jhu/lcsr"

# Frame counts in the per-spec comments are from the paper's Table S1
# (Complete Open-H-Embodiment dataset inventory). ``mix_ratio``s for the
# non-CMR pool are frame-proportional, normalized so the pool sums to ~4.0
# (norm factor ≈ 2,276,756 = total_non_cmr_frames / 4.0). Recompute against
# each dataset's actual ``meta/info.json::total_frames`` at setup.
OPEN_H_DATASET_SPECS: list[dict] = [
    # ===== CMR Versius (50% of total, 4 clinical procedures) =====
    # On-disk: cmr_surgical/{cholecystectomy,hysterectomy,inguinal_hernia,prostatectomy}
    # (dry_box / peg_transfer benchtop leaves intentionally excluded).
    # mix_ratio: 1.0 each x 4 = 4.0 total (50%).
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/cmr_surgical/cholecystectomy",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },  # 16,999,777 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/cmr_surgical/hysterectomy",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },  # 26,374,851 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/cmr_surgical/inguinal_hernia",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },  # 25,807,467 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/cmr_surgical/prostatectomy",
        "embodiment": EmbodimentTag.CMR_VERSIUS,
        "mix_ratio": 1.0,
    },  # 36,516,224 fr
    # ===== JHU IMERSE (dVRK-Si stereo; we read the left endoscope) =====
    # AUDIT: confirm action/state keys (psm1/psm2 pose+gripper) and the
    # video key (observation.images.endoscope.left) in each meta/modality.json.
    {
        "path": f"{_JHU_IMERSE}/srth_porcine_chole",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.825,
    },  # 1,878,393 fr
    {
        "path": f"{_JHU_IMERSE}/wound_closure/point_labeled/fausto_0_1_jesse_0_1_2_labeled",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.827,
    },  # 1,883,971 fr
    {
        "path": f"{_JHU_IMERSE}/suturebot",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.227,
    },  # 516,334 fr
    {
        "path": f"{_JHU_IMERSE}/nephfat/nephfat",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.217,
    },  # 494,525 fr
    {
        "path": f"{_JHU_IMERSE}/srt_needle_pickup_handover",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.026,
    },  # 58,305 fr
    {
        "path": f"{_JHU_IMERSE}/cao_cautery_combined",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.023,
    },  # 52,748 fr
    {
        "path": f"{_JHU_IMERSE}/srt_tissue_lift",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.012,
    },  # 27,487 fr
    # ===== JHU LCSR ARCADE (dVRK-Si stereo; left endoscope) =====
    {
        "path": f"{_JHU_LCSR}/arcade/cholecystectomy",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.080,
    },  # 181,021 fr
    {
        "path": f"{_JHU_LCSR}/arcade/cautery",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 0.002,
    },  # 5,288 fr
    # ===== JHU LCSR MIRACLE (dedicated embodiment; cam keys 'left'/'right') =====
    {
        "path": f"{_JHU_LCSR}/miracle/prepare_to_pierce",
        "embodiment": EmbodimentTag.JHU_LSCR_MIRACLE,
        "mix_ratio": 0.001,
    },  # 582 fr
    # ===== JHU LCSR SMARTS (dedicated embodiment; per-participant leaves) =====
    {
        "path": f"{_JHU_LCSR}/smarts/SurgSync-stitch-coldcut/P1",
        "embodiment": EmbodimentTag.JHU_LSCR_SMARTS,
        "mix_ratio": 0.023,
    },  # 53,114 fr
    {
        "path": f"{_JHU_LCSR}/smarts/SurgSync-stitch-coldcut/P2",
        "embodiment": EmbodimentTag.JHU_LSCR_SMARTS,
        "mix_ratio": 0.014,
    },  # 32,426 fr
    {
        "path": f"{_JHU_LCSR}/smarts/SurgSync-stitch-coldcut/P3",
        "embodiment": EmbodimentTag.JHU_LSCR_SMARTS,
        "mix_ratio": 0.008,
    },  # 17,485 fr
    # NOTE: smarts/SurgSync-multitask/{P1..P4} also exist on disk; add once
    # their action schema is audited (Table S1 lists them under SMARTS).
    # ===== JHU IMERSE STAR-IL (single KUKA arm, no gripper) =====
    {
        "path": f"{_JHU_IMERSE}/star_il/star_il",
        "embodiment": EmbodimentTag.JHU_IMERSE,
        "mix_ratio": 0.095,
    },  # 216,140 fr
    # ===== Obuda dVRK (stereo; left endoscope) — all task leaves =====
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/frs_dome_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.062,
    },  # 141,078 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/pork_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.073,
    },  # 165,486 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/pegtransfer_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.059,
    },  # 134,832 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/rollercoaster_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.057,
    },  # 130,268 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/needlethreading_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.045,
    },  # 103,067 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/needlethreading_2",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.045,
    },  # 102,221 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/seaspike_3",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.045,
    },  # 102,948 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/seaspike_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.039,
    },  # 89,269 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/pegtransfer_2",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.034,
    },  # 78,140 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/seaspike_2",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.030,
    },  # 67,658 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/obuda/skinphantom_1",
        "embodiment": EmbodimentTag.DVRK_OBUDA,
        "mix_ratio": 0.018,
    },  # 41,979 fr
    # ===== Stanford real dVRK (Euler RPY; cam keys 'camera_left'/'camera_right') =====
    {
        "path": f"{_STANFORD_BASE}/needle_transfer",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.138,
    },  # 313,882 fr
    {
        "path": f"{_STANFORD_BASE}/tissue_retraction",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.128,
    },  # 291,826 fr
    {
        "path": f"{_STANFORD_BASE}/peg_transfer",
        "embodiment": EmbodimentTag.DVRK_STANFORD_REAL,
        "mix_ratio": 0.118,
    },  # 268,729 fr
    # ===== Turin MITIC (stereo; left endoscope; no grippers) — all leaves =====
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/turin/mitic_lerobot_ex_vivo",
        "embodiment": EmbodimentTag.TURIN_MITIC_EX_VIVO,
        "mix_ratio": 0.171,
    },  # 388,690 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/turin/mitic_lerobot_plastic_pad_3dmed",
        "embodiment": EmbodimentTag.TURIN_MITIC_EX_VIVO,
        "mix_ratio": 0.107,
    },  # 243,229 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/turin/mitic_lerobot_plastic_tube",
        "embodiment": EmbodimentTag.TURIN_MITIC_EX_VIVO,
        "mix_ratio": 0.095,
    },  # 216,070 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/turin/mitic_lerobot_plastic_pad",
        "embodiment": EmbodimentTag.TURIN_MITIC_EX_VIVO,
        "mix_ratio": 0.066,
    },  # 149,846 fr
    # ===== UCSD (stereo; cam keys 'left'/'right') =====
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/ucsd/surgical_learning_dataset",
        "embodiment": EmbodimentTag.DVRK_UCSD,
        "mix_ratio": 0.127,
    },  # 288,604 fr
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/ucsd/surgical_learning_dataset2",
        "embodiment": EmbodimentTag.DVRK_UCSD,
        "mix_ratio": 0.012,
    },  # 26,313 fr
    # ===== UC Berkeley debridement (stereo; cam keys 'left'/'right') =====
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/ucberkeley/debridement_lerobot",
        "embodiment": EmbodimentTag.DVRK_UCB,
        "mix_ratio": 0.097,
    },  # 221,950 fr
    # ===== TUD TUNDRA grasping_retraction (single-arm UR5e; laparoscope_left) =====
    # endoscope_guidance leaf excluded (4D delta-tip schema).
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/tud/260131_tundra_dataset/grasping_retraction",
        "embodiment": EmbodimentTag.TUD_TUNDRA,
        "mix_ratio": 0.037,
    },  # 83,159 fr (35,406 + 47,753)
    # ===== Virtual Incision MIRA (delta-command pose + gripper; 'endoscope') =====
    {
        "path": f"{_OPENH_SURGICAL_ROOT}/virtual_incision/150_episodes_mira_needle_lift",
        "embodiment": EmbodimentTag.VIRTUAL_INCISION_MIRA,
        "mix_ratio": 0.018,
    },  # 39,960 fr
]
# Derived: the set of all Open-H embodiment tag strings.
# Used by dataset.py to enforce stats_cosmos.json requirement.
# Includes both EMBODIMENT_REGISTRY keys (non-CMR) and all tags from the specs.
OPEN_H_EMBODIMENT_TAGS: frozenset[str] = frozenset(
    {
        (spec["embodiment"].value if isinstance(spec["embodiment"], EmbodimentTag) else spec["embodiment"])
        for spec in OPEN_H_DATASET_SPECS
    }
    | set(EMBODIMENT_REGISTRY.keys())
)


# =============================================================================
# JHU dVRK Mono Downstream Fine-Tune Mixture (frame-proportional, exp_605-style)
# =============================================================================
# Dataset mixture for the downstream fine-tune of the pretrained
# Cosmos-H-Surgical-Simulator checkpoint on JHU dVRK tabletop data only.
#
# All 9 datasets use the unified ``EmbodimentTag.JHU_DVRK_MONO`` embodiment
# (20D dual-arm EEF + gripper action, post-transform; zero-padded to
# MAX_ACTION_DIM=44 by MixedLeRobotDataset). Effective training rate is 10 Hz
# (30 Hz raw storage × timestep_interval=3, set on the registry entry above).
#
# Composition:
#   - ``hf_suturebot``: the original JHU SutureBot LeRobot bundle from the
#     public Open-H release (1,452 episodes / 516,334 frames).
#   - 8 newly-converted subsets under ``Open-H_failures_ood``: successful
#     demos, OOD demos, and failure-case demos converted from the JHU zarr
#     archives via ``scripts/convert_jhu_zarr_to_lerobot.py``
#     (1,483 episodes / 557,172 frames across 8 subsets).
# Total: 2,935 episodes / 1,073,506 frames.
#
# Weighting: ``mix_ratio_i = total_frames_i`` per subset, mirroring exp_605's
# 1× ``jhu_train`` mixture from the prior Cosmos-Surg-dVRK work
# (cf. ``exp605-nigeln-14_cosmos_predict2/exp605-data-mixture.md``):
# **each subset contributes proportionally to its number of frames** — no
# minority oversampling, no size-equalizing.
#
# Why frames-proportional instead of equal ``mix_ratio=1.0``: equal weighting
# would give each of the 9 subsets 1/9 ≈ 11.1% of training samples,
# upsampling the smallest subset (``suture_bot_success``, 1,557 frames) by
# >300× to match ``hf_suturebot`` (516,334 frames). That is undesirable here
# — exp_605 trained with frame-proportional sampling and produced the
# checkpoints used by the downstream Cosmos-Surg-dVRK pipeline.
#
# Train / val split policy (per-spec overrides; Cosmos-predict2.5's trainer
# barely consumes ``dataloader_val`` in practice — see
# ``imaginaire/config.py::validation_iter`` default 999_999_999 — so the held-
# out test data is mostly inert plumbing; we minimize what we give up):
#
#     subset                    test_split_ratio  rationale
#     -----------------------   ----------------  ---------------------------
#     hf_suturebot              0.01              very large; 1% is plenty
#     knot_tying                0.01              large; 1% is plenty
#     cosmos_throw_fail_demo    0.01              medium; 1% is plenty
#     suture_bot_success        0.02 (default)    tiny; keep symmetric 2%
#     suture_bot_failure        0.02 (default)
#     cosmos_fail_filtered      0.02 (default)
#     cosmos_knot_fail_demo     0.02 (default)
#     suturebot_act_throw_eval  0.02 (default)
#     ood                       0.00 (full)       no holdout - all 100% in train
#
# Math: ``MixedLeRobotDataset._compute_repeat_factors`` does
# ``per_sample_weight_i = mix_ratio_i / len(ds_i)``. ``LeRobotSingleDataset``
# enumerates one base index per frame (every ``(traj_id, base_index)`` pair,
# see ``dataset.py::_get_all_steps``) and ``WrappedLeRobotSingleDataset``
# splits ``test_split_ratio`` of those off as the test partition. With
# ``mix_ratio_i = total_frames_i`` and the per-spec test ratios above, every
# subset's per_sample_weight in train mode rounds to repeat_factor=1, so no
# upsampling is applied and each subset contributes its actual frame share.
# Total drift from "exact frame-proportional" is < 0.2 percentage points
# (driven entirely by ``ood`` keeping its full 21.42% pool share instead of
# 21.24%). All other subsets drift down by < 0.08 pp.
#
# To re-balance later (e.g., re-introduce exp_606's 2× failure/OOD
# oversampling), multiply the relevant ``mix_ratio`` by the desired factor.
# =============================================================================

# All 9 JHU dVRK mono subsets now live under one parent directory:
# ``LeRobot_540x960`` (the rebuild produced by
# ``scripts/convert_jhu_zarr_to_lerobot_highres.py`` for the 8 zarr-only
# subsets and ``scripts/dvrk_to_lerobot_highres.py`` for ``hf_suturebot``).
# All 9 share a uniform mono schema (``observation.images.endoscope_left``,
# ``robot_type='jhu_dvrk_mono'``) at native 540 H x 960 W.  The dataloader
# resizes 540 -> 720 at runtime per the ``video_height/video_width=720/960``
# entry in ``EMBODIMENT_REGISTRY['jhu_dvrk_mono']`` above.
_JHU_DVRK_MONO_540X960_BASE = (
    "/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/"
    "datasets/JHU_data_jpeg100_noacc_clean++/LeRobot_540x960"
)
# Backward-compatible aliases: legacy code that still imports
# ``_JHU_DVRK_MONO_OPEN_H_BASE`` or ``_JHU_DVRK_MONO_FAILURES_OOD_BASE``
# now resolves to the same unified ``LeRobot_540x960`` root.  Kept as
# distinct names only to avoid breaking existing imports — feel free to
# inline + delete these once the call sites are migrated.
_JHU_DVRK_MONO_OPEN_H_BASE = _JHU_DVRK_MONO_540X960_BASE
_JHU_DVRK_MONO_FAILURES_OOD_BASE = _JHU_DVRK_MONO_540X960_BASE

# Per-spec overrides for the 8 non-ood subsets. Frame counts re-measured
# 2026-05-07 against each subset's ``meta/info.json::total_frames`` in
# ``LeRobot_540x960``.  ``hf_suturebot``'s count is +2,149 vs. the
# original Open-H reference because our mono converter accepts partially-
# recorded episodes (``ee_csv.csv`` + ``left_img_dir/`` only) that the
# original 4-cam converter dropped — see
# ``scripts/dvrk_to_lerobot_highres.py::_is_raw_episode_dir`` for the
# rationale.  Both train and val mixtures share these specs (the val
# mixture excludes ``ood``, and applies ``data_split="test"`` at the
# constructor level).
_JHU_DVRK_MONO_FINETUNE_NON_OOD_SPECS: list[dict] = [
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/hf_suturebot",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 518483.0,
        "test_split_ratio_override": 0.01,
    },  # 1,458 episodes / 518,483 frames -> 513,298 train / 5,185 val
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/knot_tying",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 209253.0,
        "test_split_ratio_override": 0.01,
    },  # 512 episodes / 209,253 frames -> 207,161 train / 2,092 val
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/suture_bot_success",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 1557.0,
    },  # 10 episodes / 1,557 frames -> 1,526 train / 31 val (default 2%)
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/suture_bot_failure",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 8793.0,
    },  # 46 episodes / 8,793 frames -> 8,618 train / 175 val (default 2%)
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/cosmos_fail_filtered",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 12948.0,
    },  # 163 episodes / 12,948 frames -> 12,690 train / 258 val (default 2%)
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/cosmos_throw_fail_demo",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 54581.0,
        "test_split_ratio_override": 0.01,
    },  # 158 episodes / 54,581 frames -> 54,036 train / 545 val
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/cosmos_knot_fail_demo",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 30502.0,
    },  # 151 episodes / 30,502 frames -> 29,892 train / 610 val (default 2%)
    {
        "path": f"{_JHU_DVRK_MONO_540X960_BASE}/suturebot_act_throw_eval",
        "embodiment": EmbodimentTag.JHU_DVRK_MONO,
        "mix_ratio": 11548.0,
    },  # 30 episodes / 11,548 frames -> 11,318 train / 230 val (default 2%)
]

# ``ood`` spec — train-only with a hard ``data_split_override="full"`` so 100%
# of its 227,990 frames participate in training and zero are held out for the
# test partition. Pinned in a separate constant so it can be referenced (or
# removed) explicitly from the train mixture below.
_JHU_DVRK_MONO_FINETUNE_OOD_SPEC: dict = {
    "path": f"{_JHU_DVRK_MONO_540X960_BASE}/ood",
    "embodiment": EmbodimentTag.JHU_DVRK_MONO,
    "mix_ratio": 227990.0,
    "data_split_override": "full",
}  # 413 episodes / 227,990 frames -> 227,990 train / 0 val

# Train mixture: 8 non-ood subsets + ``ood`` (full split, no holdout).
JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS: list[dict] = [
    *_JHU_DVRK_MONO_FINETUNE_NON_OOD_SPECS,
    _JHU_DVRK_MONO_FINETUNE_OOD_SPEC,
]

# Val mixture: 8 non-ood subsets only. ``ood`` is intentionally excluded so its
# frames are never used as validation. The constructor-level ``data_split="test"``
# argument applied at instantiation time then selects the trailing test partition
# per spec (sized by each spec's ``test_split_ratio_override`` or the default).
JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS: list[dict] = list(_JHU_DVRK_MONO_FINETUNE_NON_OOD_SPECS)


def _build_generic_config_and_transforms(
    num_frames: int,
    reg: dict,
    downscaled_res: bool = False,
) -> tuple[dict, ComposedModalityTransform, ComposedModalityTransform]:
    """Build modality config and transforms for a generic (non-CMR) Open-H embodiment.

    This creates the standard pipeline:
      Video: ToTensor → Crop (train only) → Resize
      State/Action: ToTensor → mean_std normalization → Concat

    Supports ``pass_through_state_keys`` for embodiments where certain state
    keys are needed as references for action transforms (e.g. REL_XYZ_ROT6D)
    but should NOT be normalized or concatenated into the model's state input.
    For example, dVRK UCB uses cartesian pose as the action reference frame
    while joint angles are the actual state input.

    Args:
        num_frames: Total number of video frames (1 context + N-1 prediction).
        reg: Registry entry dict from EMBODIMENT_REGISTRY.  May include:
            - ``pass_through_state_keys``: list of state keys that are loaded
              and converted to tensors (so the action transform can read them)
              but excluded from normalization and concatenation.
        downscaled_res: If True, use 256×256 resolution.

    Returns:
        Tuple of (modality_config_dict, train_transform, test_transform).
    """
    timestep_interval = reg["timestep_interval"]

    # Video: all num_frames frames
    video_delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))

    # Action: num_frames - 1 action timesteps (prediction frames only)
    num_action_frames = num_frames - 1
    action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))

    config = {
        "video": ModalityConfig(
            delta_indices=video_delta_indices,
            modality_keys=reg["video_keys"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=reg["state_keys"],
        ),
        "action": ModalityConfig(
            delta_indices=action_delta_indices,
            modality_keys=reg["action_keys"],
        ),
        # Extra metadata for dataset initialization
        "modality_filename": reg.get("modality_filename", "meta/modality.json"),
    }

    width = reg["video_width"] if not downscaled_res else 256
    height = reg["video_height"] if not downscaled_res else 256
    norm_mode = reg.get("normalization_mode", "mean_std")

    video_keys = reg["video_keys"]
    state_keys = reg["state_keys"]
    action_keys = reg["action_keys"]
    action_key_configs = reg.get("action_key_configs", {})

    # Pass-through state keys: loaded and converted to tensors (so the action
    # transform can read them as reference frames), but NOT normalized or
    # concatenated into the model's state input vector.
    # Example: dVRK UCB uses cartesian pose (psm1_pose, psm2_pose) as the
    # REL_XYZ_ROT6D reference frame, while joint angles are the actual state input.
    pass_through_state_keys = set(reg.get("pass_through_state_keys", []))

    # State keys that get normalized and concatenated (excluding pass-through)
    normalizable_state_keys = [k for k in state_keys if k not in pass_through_state_keys]

    # Build the relative action transform (runs BEFORE normalization)
    # This converts raw absolute actions to deltas using the state reference.
    # IMPORTANT: Must run before StateActionTransform because normalization
    # would corrupt the reference state (e.g., quaternions) needed for
    # the relative conversion.
    rel_action_transform = GenericRelativeActionTransform(
        apply_to=action_keys,
        action_key_configs=action_key_configs,
    )

    train_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_keys),
            VideoCrop(apply_to=video_keys, scale=0.95),
            VideoResize(apply_to=video_keys, height=height, width=width, interpolation="linear"),
            # Convert ALL state keys to tensors (including pass-through keys
            # needed by the action transform for reference frame lookups)
            StateActionToTensor(apply_to=state_keys),
            StateActionToTensor(apply_to=action_keys),
            # Delta action conversion BEFORE normalization
            # (pass-through state keys are read here as reference frames)
            rel_action_transform,
            # Normalization AFTER delta conversion
            # Only normalize non-pass-through state keys
            StateActionTransform(
                apply_to=normalizable_state_keys,
                normalization_modes={k: norm_mode for k in normalizable_state_keys},
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={k: norm_mode for k in action_keys},
            ),
            # Only concatenate non-pass-through state keys into model input
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=normalizable_state_keys,
                action_concat_order=action_keys,
            ),
        ]
    )

    test_transform = ComposedModalityTransform(
        transforms=[
            VideoToTensor(apply_to=video_keys),
            VideoResize(apply_to=video_keys, height=height, width=width, interpolation="linear"),
            # Convert ALL state keys to tensors (including pass-through)
            StateActionToTensor(apply_to=state_keys),
            StateActionToTensor(apply_to=action_keys),
            # Delta action conversion BEFORE normalization
            rel_action_transform,
            # Normalization AFTER delta conversion
            StateActionTransform(
                apply_to=normalizable_state_keys,
                normalization_modes={k: norm_mode for k in normalizable_state_keys},
            ),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={k: norm_mode for k in action_keys},
            ),
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=normalizable_state_keys,
                action_concat_order=action_keys,
            ),
        ]
    )

    return config, train_transform, test_transform


def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_res=False):
    if embodiment == "gr1":
        timestep_interval = 2
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.ego_view_freq20" if not downscaled_res else "video.ego_view_bg_crop_pad_res256_freq20"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
        }
    elif embodiment == "gr1_video_only":
        timestep_interval = 1
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.ego_view_bg_crop_pad_res256_freq20"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm",
                    "state.right_arm",
                    "state.left_hand",
                    "state.right_hand",
                    "state.waist",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm",
                    "action.right_arm",
                    "action.left_hand",
                    "action.right_hand",
                    "action.waist",
                ],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.coarse_action"]),
        }
    elif embodiment == "agibot":
        timestep_interval = 4
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        video_key = "video.top_head" if not downscaled_res else "video.top_head_pad_res256_freq10"
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[video_key],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=[
                    "state.left_arm_joint_position",
                    "state.right_arm_joint_position",
                    "state.left_effector_position",
                    "state.right_effector_position",
                    "state.head_position",
                    "state.waist_position",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=[
                    "action.left_arm_joint_position",
                    "action.right_arm_joint_position",
                    "action.left_effector_position",
                    "action.right_effector_position",
                    "action.head_position",
                    "action.waist_position",
                    "action.robot_velocity",
                ],
            ),
        }
    elif embodiment == "cmr_versius":
        # CMR Versius surgical robot configuration
        # Original data is 60Hz, using FRAME_STRIDE=6 for 10fps effective rate
        timestep_interval = 6

        # Video: 13 frames (1 context + 12 prediction)
        video_delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))

        # Action: 12 timesteps (only for the 12 prediction frames, not context)
        # The model expects num_actions to be divisible by temporal_compression_ratio (4)
        # 12 actions / 4 = 3 latent temporal positions, each getting action embedding
        # Note: action timesteps start from index 0 (same as video context frame) because
        # the action at t=0 represents the transition FROM frame 0 TO frame 1
        num_action_frames = num_frames - 1  # 12 action timesteps for 13 video frames
        action_delta_indices = list(range(0, num_action_frames * timestep_interval, timestep_interval))

        config = {
            "video": ModalityConfig(
                delta_indices=video_delta_indices,
                modality_keys=["video.endoscope"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],  # Single reference state for hybrid-relative
                modality_keys=[
                    # State pose for reference (xyz + quat = 7D each arm)
                    "state.left_pose",
                    "state.left_gripper",
                    "state.right_pose",
                    "state.right_gripper",
                    # Engagement status for clutch-aware processing
                    "state.hapticengaged_left",
                    "state.hapticengaged_right",
                    # Motion scaling factors (pass-through for hybrid-relative conversion)
                    "state.translation_scaling",
                    "state.rotation_scaling",
                ],
            ),
            "action": ModalityConfig(
                delta_indices=action_delta_indices,
                modality_keys=[
                    # Left arm: pose (xyz + quat = 7D raw, converted to xyz + rot6d = 9D hybrid-relative)
                    "action.left_pose",
                    "action.left_gripper",
                    # Right arm: pose (xyz + quat = 7D raw, converted to xyz + rot6d = 9D hybrid-relative)
                    "action.right_pose",
                    "action.right_gripper",
                    # Energy buttons (binary)
                    "action.left_energy",
                    "action.right_energy",
                    # Thumbstick controls (for endoscope control and instrument straighten function)
                    "action.thumbstick_x_left",
                    "action.thumbstick_x_right",
                    "action.thumbstick_y_left",
                    "action.thumbstick_y_right",
                    "action.thumbstickBtn_left",
                    "action.thumbstickBtn_right",
                    # Clutch button inputs (engage/disengage arm control)
                    "action.clutchBtn_left",
                    "action.clutchBtn_right",
                    # Engagement status (pass-through for clutch-aware processing)
                    # Note: hapticengaged keys (without cond_ prefix) are used for clutch-aware processing but removed after
                    "action.hapticengaged_left",
                    "action.hapticengaged_right",
                    # =====================================================
                    # STATE CONDITIONING VARIABLES (sampled at action timesteps)
                    # These are from observation.state but sampled at action delta_indices
                    # for MLP conditioning. They're passed through as absolute values.
                    # =====================================================
                    # Haptic engagement state (persistent, unlike clutchBtn which is momentary)
                    "action.cond_hapticengaged_left",
                    "action.cond_hapticengaged_right",
                    # Which physical arm (0-3) each controller is linked to
                    "action.cond_armlinkedtohaptic_left",
                    "action.cond_armlinkedtohaptic_right",
                    # Instrument type for each arm (0-3)
                    "action.cond_arm_0_instrtype",
                    "action.cond_arm_1_instrtype",
                    "action.cond_arm_2_instrtype",
                    "action.cond_arm_3_instrtype",
                    # HUD color assignment for each arm (0-3)
                    "action.cond_arm_0_color",
                    "action.cond_arm_1_color",
                    "action.cond_arm_2_color",
                    "action.cond_arm_3_color",
                    # Electrosurgery mode (CUT/COAG) selected on each controller
                    "action.cond_electroSurgeryMode_left",
                    "action.cond_electroSurgeryMode_right",
                ],
            ),
        }

    # =========================================================================
    # SutureBot (JHU dVRK, pre-concatenated LeRobot format)
    # =========================================================================
    # SutureBot uses a single concatenated action key 'action.action' (20D)
    # with dual-arm dVRK data: [arm1: xyz(3)+rot6d(6)+gripper(1)] × 2.
    # Uses RelativeActionTransform for delta conversion (different from the
    # per-key GenericRelativeActionTransform used by EMBODIMENT_REGISTRY entries).
    # =========================================================================
    elif embodiment == "suturebot":
        timestep_interval = 3
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.observation.images.main"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state.observation.state"],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["action.action"],
            ),
        }

    # =========================================================================
    # Registry-based embodiments (all non-CMR Open-H datasets)
    # =========================================================================
    if embodiment in EMBODIMENT_REGISTRY:
        return _build_generic_config_and_transforms(num_frames, EMBODIMENT_REGISTRY[embodiment], downscaled_res)

    video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
    if embodiment == "gr1" or embodiment == "gr1_video_only":
        width = 832 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256
    elif embodiment == "agibot":
        width = 640 if not downscaled_res else 256
        height = 480 if not downscaled_res else 256
    elif embodiment == "cmr_versius":
        # CMR Versius endoscope video resolution (original: 1920x1080, 16:9 aspect ratio)
        #
        # IMPORTANT: Resolution must be divisible by 16 (8x VAE compression × 2 patch size)
        # Valid 16:9 options: 512x288, 768x432, 1024x576, 1280x720
        # Invalid: 384x216 (216/16=13.5), 320x180 (180/16=11.25)
        #
        # Using 512x288 for fast PoC training while maintaining 16:9 aspect ratio
        # For production: consider 768x432 or 1280x720 (matches Cosmos 720p pretrain)
        # cf. https://docs.google.com/presentation/d/1G0mqiQRBQohDAMjMG6hpLzPVCZi3KbJxHSi4LXMJl5A/edit?slide=id.g3b869a60288_1_50#slide=id.g3b869a60288_1_50
        width = 512 if not downscaled_res else 256
        height = 288 if not downscaled_res else 256
    elif embodiment == "suturebot":
        # SutureBot: same resolution as CMR Versius (512x288, 16:9)
        width = 512 if not downscaled_res else 256
        height = 288 if not downscaled_res else 256

    # Build embodiment-specific transforms
    if embodiment == "cmr_versius":
        # CMR Versius uses hybrid-relative actions with rot6d rotation format
        # Final conditioning: 44D = 30D actions + 14D state conditioning
        #   Actions (30D):
        #     - left(9D pose + 1D gripper) + right(9D pose + 1D gripper) = 20D
        #     - energy(2D) + thumbstick_x(2D) + thumbstick_y(2D) + thumbstickBtn(2D) + clutchBtn(2D) = 10D
        #   State conditioning (14D, sampled at action timesteps):
        #     - haptic_engaged(2D) + armlinkedtohaptic(2D) + instrtype(4D) + color(4D) + electroSurgeryMode(2D) = 14D
        # Note: hapticengaged keys (without cond_ prefix) are used for clutch-aware processing but removed after

        # Keys that get concatenated into final conditioning tensor (exclude pass-through keys)
        cmr_action_output_keys = [
            # === ACTIONS (30D) ===
            "action.left_pose",
            "action.left_gripper",
            "action.right_pose",
            "action.right_gripper",
            "action.left_energy",
            "action.right_energy",
            "action.thumbstick_x_left",
            "action.thumbstick_x_right",
            "action.thumbstick_y_left",
            "action.thumbstick_y_right",
            "action.thumbstickBtn_left",
            "action.thumbstickBtn_right",
            "action.clutchBtn_left",
            "action.clutchBtn_right",
            # === STATE CONDITIONING (12D) ===
            "action.cond_hapticengaged_left",
            "action.cond_hapticengaged_right",
            "action.cond_armlinkedtohaptic_left",
            "action.cond_armlinkedtohaptic_right",
            "action.cond_arm_0_instrtype",
            "action.cond_arm_1_instrtype",
            "action.cond_arm_2_instrtype",
            "action.cond_arm_3_instrtype",
            "action.cond_arm_0_color",
            "action.cond_arm_1_color",
            "action.cond_arm_2_color",
            "action.cond_arm_3_color",
            # Electrosurgery mode
            "action.cond_electroSurgeryMode_left",
            "action.cond_electroSurgeryMode_right",
        ]

        # Thumbstick, clutch button, and state conditioning keys (ABSOLUTE values, pass-through)
        # These are NOT converted to deltas - they pass through as raw absolute values.
        # - Thumbstick: endoscope/instrument control (continuous)
        # - ClutchBtn: engage/disengage arm control (binary button press)
        # - State conditioning: system state sampled at action timesteps for MLP conditioning
        cmr_passthrough_action_keys = [
            # Thumbstick controls
            "action.thumbstick_x_left",
            "action.thumbstick_x_right",
            "action.thumbstick_y_left",
            "action.thumbstick_y_right",
            "action.thumbstickBtn_left",
            "action.thumbstickBtn_right",
            # Clutch buttons
            "action.clutchBtn_left",
            "action.clutchBtn_right",
            # State conditioning (from observation.state, sampled at action timesteps)
            "action.cond_hapticengaged_left",
            "action.cond_hapticengaged_right",
            "action.cond_armlinkedtohaptic_left",
            "action.cond_armlinkedtohaptic_right",
            "action.cond_arm_0_instrtype",
            "action.cond_arm_1_instrtype",
            "action.cond_arm_2_instrtype",
            "action.cond_arm_3_instrtype",
            "action.cond_arm_0_color",
            "action.cond_arm_1_color",
            "action.cond_arm_2_color",
            "action.cond_arm_3_color",
            # Electrosurgery mode
            "action.cond_electroSurgeryMode_left",
            "action.cond_electroSurgeryMode_right",
        ]

        # Keys that don't need normalization (engagement status, scaling factors, pass-through)
        cmr_state_passthrough_keys = [
            "state.hapticengaged_left",
            "state.hapticengaged_right",
            "state.translation_scaling",
            "state.rotation_scaling",
        ]
        cmr_action_passthrough_keys = ["action.hapticengaged_left", "action.hapticengaged_right"]

        # State keys to include in final output (excluding pass-through only keys)
        cmr_state_output_keys = [k for k in state_modality.modality_keys if k not in cmr_state_passthrough_keys]

        # NOTE: Normalization uses stats_cosmos.json (not stats.json) for CMR Versius.
        # Run scripts/compute_cmr_action_stats.py to generate stats_cosmos.json with
        # correct statistics for the 9D hybrid-relative pose format.
        # Stats loading is handled in dataset.py which checks for stats_cosmos.json first.

        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                # CMR Versius relative action transform: converts absolute actions to hybrid-relative
                # IMPORTANT: Must run BEFORE state normalization because it uses raw state poses
                # (state.left_pose, state.right_pose) as reference for relative computation.
                # Normalizing quaternions would produce invalid rotation matrices!
                # CMR data stores poses as quaternions (7D: xyz + quat_xyzw), output is rot6d (9D)
                CMRVersiusRelativeActionTransform(
                    apply_to=["action.left_pose", "action.right_pose"],
                    pose_keys={
                        "action.left_pose": "state.left_pose",
                        "action.right_pose": "state.right_pose",
                    },
                    gripper_keys=["action.left_gripper", "action.right_gripper"],
                    energy_keys=["action.left_energy", "action.right_energy"],
                    thumbstick_keys=cmr_passthrough_action_keys,
                    engaged_left_key="state.hapticengaged_left",
                    engaged_right_key="state.hapticengaged_right",
                    input_rotation_format="quat",  # CMR data uses quaternions (xyzw)
                    reference_rotation_format="quat",  # State also uses quaternions (xyzw)
                    # Motion scaling (converts hand-controller-space to instrument-space):
                    translation_scaling_key="state.translation_scaling",
                    rotation_scaling_key="state.rotation_scaling",
                    # Remove passthrough keys after processing (not needed in final output)
                    action_passthrough_keys=cmr_action_passthrough_keys,
                    state_passthrough_keys=cmr_state_passthrough_keys,
                ),
                # State normalization (uses stats_cosmos.json) - AFTER CMRVersiusRelativeActionTransform
                # State poses remain as raw 7D (xyz + quat), normalized here for model input
                StateActionTransform(
                    apply_to=cmr_state_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_state_output_keys},
                ),
                # Action normalization (uses stats_cosmos.json with 9D hybrid-relative pose stats)
                StateActionTransform(
                    apply_to=cmr_action_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_action_output_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=cmr_state_output_keys,
                    action_concat_order=cmr_action_output_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                # CMR Versius relative action transform - BEFORE state normalization
                CMRVersiusRelativeActionTransform(
                    apply_to=["action.left_pose", "action.right_pose"],
                    pose_keys={
                        "action.left_pose": "state.left_pose",
                        "action.right_pose": "state.right_pose",
                    },
                    gripper_keys=["action.left_gripper", "action.right_gripper"],
                    energy_keys=["action.left_energy", "action.right_energy"],
                    thumbstick_keys=cmr_passthrough_action_keys,
                    engaged_left_key="state.hapticengaged_left",
                    engaged_right_key="state.hapticengaged_right",
                    input_rotation_format="quat",
                    reference_rotation_format="quat",
                    translation_scaling_key="state.translation_scaling",
                    rotation_scaling_key="state.rotation_scaling",
                    # Remove passthrough keys after processing (not needed in final output)
                    action_passthrough_keys=cmr_action_passthrough_keys,
                    state_passthrough_keys=cmr_state_passthrough_keys,
                ),
                # State normalization (uses stats_cosmos.json) - AFTER CMRVersiusRelativeActionTransform
                StateActionTransform(
                    apply_to=cmr_state_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_state_output_keys},
                ),
                # Action normalization (uses stats_cosmos.json with 9D hybrid-relative pose stats)
                StateActionTransform(
                    apply_to=cmr_action_output_keys,
                    normalization_modes={key: "mean_std" for key in cmr_action_output_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=cmr_state_output_keys,
                    action_concat_order=cmr_action_output_keys,
                ),
            ]
        )
    elif embodiment == "suturebot":
        # SutureBot uses pre-concatenated 20D actions with RelativeActionTransform.
        # Pipeline: ToTensor → RelativeAction (absolute→relative) → Normalize → Concat
        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                RelativeActionTransform(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                RelativeActionTransform(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
    else:
        # Default transforms for gr1, agibot, etc.
        train_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoCrop(apply_to=video_modality.modality_keys, scale=0.95),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                # VideoColorJitter(apply_to=video_modality.modality_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )
        test_transform = ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=video_modality.modality_keys),
                VideoResize(apply_to=video_modality.modality_keys, height=height, width=width, interpolation="linear"),
                StateActionToTensor(apply_to=state_modality.modality_keys),
                StateActionTransform(
                    apply_to=state_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in state_modality.modality_keys},
                ),
                StateActionToTensor(apply_to=action_modality.modality_keys),
                StateActionTransform(
                    apply_to=action_modality.modality_keys,
                    normalization_modes={key: "min_max" for key in action_modality.modality_keys},
                ),
                ConcatTransform(
                    video_concat_order=video_modality.modality_keys,
                    state_concat_order=state_modality.modality_keys,
                    action_concat_order=action_modality.modality_keys,
                ),
            ]
        )

    return config, train_transform, test_transform


# =============================================================================
# Dataset-spec builder helpers (Cosmos3 adapter)
# =============================================================================
# The static ``OPEN_H_DATASET_SPECS`` / ``JHU_DVRK_MONO_FINETUNE_*_SPECS``
# constants above pin absolute paths to the Lustre filesystem.  These
# builder helpers re-root every spec under a caller-provided ``base_path``
# so the Cosmos3 YAML can plug in ``$DATASET_PATH`` via
# ``${oc.env:DATASET_PATH}`` without editing the registry.
#
# Each spec's leaf path (the final subdir under the original base) is
# preserved — only the *root* changes:
#
#     original:  /lustre/.../LeRobot_540x960/hf_suturebot
#     rebased :  $DATASET_PATH/hf_suturebot
#
# When ``base_path`` is ``None`` or empty, the original absolute paths are
# returned unchanged (useful when running on the cluster where the Lustre
# defaults are already correct).
# =============================================================================
from pathlib import Path as _Path


def _rebase_specs(specs: list[dict], base_path: str | None) -> list[dict]:
    """Return a copy of ``specs`` with each entry's ``path`` re-rooted under ``base_path``.

    The Open-H surgical tree is NESTED (e.g.
    ``jhu/imerse/star_il/star_il``, ``stanford/.../real_robot_dvrk/needle_transfer``),
    so a leaf-name-only rebase would collide and lose structure. We therefore
    preserve the FULL relative path under ``_OPENH_SURGICAL_ROOT`` when the
    spec lives beneath it:

        original:  {_OPENH_SURGICAL_ROOT}/jhu/imerse/star_il/star_il
        rebased :  {base_path}/jhu/imerse/star_il/star_il

    For specs NOT under ``_OPENH_SURGICAL_ROOT`` (e.g. the JHU dVRK-mono
    downstream-finetune mixture, which uses its own flat ``LeRobot_540x960``
    root), we fall back to preserving only the trailing leaf name.

    When ``base_path`` is ``None`` or empty, the original absolute paths are
    returned unchanged (the cluster-default layout).
    """
    if not base_path:
        return [dict(spec) for spec in specs]
    base = _Path(str(base_path))
    root = _Path(_OPENH_SURGICAL_ROOT)
    rebased: list[dict] = []
    for spec in specs:
        new_spec = dict(spec)
        original = _Path(spec["path"])
        try:
            rel = original.relative_to(root)
            new_spec["path"] = str(base / rel)
        except ValueError:
            # Not under the surgical root — preserve the leaf name only.
            new_spec["path"] = str(base / original.name)
        rebased.append(new_spec)
    return rebased


def get_open_h_multi_train_specs(base_path: str | None = None) -> list[dict]:
    """Open-H multi-embodiment training mixture (CMR + 13 surgical subsets).

    See :data:`OPEN_H_DATASET_SPECS` for the spec list, weighting rationale,
    and per-subset frame counts.
    """
    return _rebase_specs(OPEN_H_DATASET_SPECS, base_path)


def get_jhu_dvrk_mono_finetune_train_specs(base_path: str | None = None) -> list[dict]:
    """JHU dVRK monocular downstream fine-tune mixture (train split).

    9 subsets — 8 with a small trailing-test partition plus ``ood`` at
    full split.  See :data:`JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS`
    for the mix-ratio rationale.
    """
    return _rebase_specs(JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS, base_path)


def get_jhu_dvrk_mono_finetune_val_specs(base_path: str | None = None) -> list[dict]:
    """JHU dVRK monocular downstream fine-tune mixture (val split).

    8 non-ood subsets.  ``ood`` is intentionally excluded — its 100% goes
    to training (see :data:`JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS`).
    """
    return _rebase_specs(JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS, base_path)
