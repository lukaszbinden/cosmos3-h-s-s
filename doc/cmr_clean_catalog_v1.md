# CMR clean-window catalog v1

Generated on DRACO on 2026-07-29 from the four
`cmr-surgical-60hz-fixed` clinical procedure leaves with a 13-video-frame,
12-action horizon (`[0, 6, ..., 66]`; every raw frame in the 67-frame span is
audited).

## Source-label limitation

The CMR source README says the clinical data is unlabeled and contains
instances of failure and recovery. Its four converted LeRobot leaves expose
only `splits.train`; they contain no failure/recovery, manual-intervention, or
tool-exchange labels. Therefore the telemetry artifact below is a **candidate
review catalog**, not a training catalog. The loader refuses to train from it.

## Telemetry gate

A candidate window survives only when, at every raw 60 Hz frame in its
horizon:

- both haptics are engaged;
- both linked arms are engageable (instrument-change mode is excluded);
- neither clutch button is pressed;
- arm-to-haptic mappings and all instrument-type fields are constant; and
- all required telemetry is finite.

This removes windows that overlap partial clutch/disengagement, bedside/manual
arm movement exposed by disengagement telemetry, arm remapping, and observable
tool-change transitions. It does not claim to infer unlabeled semantic
failure/recovery or visually hidden manual/tool activity.

## Candidate artifact

- DRACO:
  `/lustre/fs11/portfolios/healthcareeng/projects/healthcareeng_holoscan/users/shuver/catalogs/cmr-clean-candidate-v1`
- Mac mirror: `/Users/shuver/cmr-clean-candidate-v1`
- catalog ID:
  `c666a6877a8fba885b2ef6e142f761d57dc08e201775bd5d0cd7f7dade14688f`
- builder SHA-256:
  `22ba771800ca15be59025bf1a440dc731105838e0a70d49277acc5192f9d03c5`
- manifest SHA-256:
  `c5a5801837479b47fe86e6757ea0eb987de1938be9e666807b6d1c56f694a8b5`
- Slurm job: `11279288` (`COMPLETED`, exit `0:0`, elapsed `00:02:57`)

| Procedure | Candidate episodes | Candidate windows | Effective windows | Keep |
| --- | ---: | ---: | ---: | ---: |
| cholecystectomy | 4,232 | 12,893,237 | 16,683,505 | 77.281% |
| hysterectomy | 6,709 | 19,374,971 | 25,884,339 | 74.852% |
| inguinal hernia | 6,397 | 18,805,512 | 25,326,591 | 74.252% |
| prostatectomy | 9,976 | 30,286,735 | 35,834,510 | 84.518% |
| **Total** | **27,314** | **81,360,455** | **103,728,945** | **78.436%** |

`review_queue.jsonl` contains one fail-closed review record for each of the
27,314 candidate episodes.

## Strict promotion and 30% mixture

Review records must explicitly set `outcome`, `manual_activity`,
`tool_exchange`, and half-open raw-frame `clean_intervals`. Failure, recovery,
manual activity, tool exchange, and unknown status contribute no strict
windows. Build the strict artifact with
`scripts/build_cmr_clean_catalog.py --tier strict --review-labels ...`.

Only after the strict manifest reports `training_authorized: true`:

```bash
export COSMOS_OPENH_CMR_CLEAN_CATALOG=/path/to/cmr-clean-strict-v1
export COSMOS_OPENH_CMR_TARGET_SHARE=0.30
export COSMOS_OPENH_STATS_POSTFIX=c3hss-cmr-clean-v1
```

The recipe rescales the four CMR weights to a nominal 30% and preserves the
relative weights of the non-CMR 70%. Requesting a CMR share without a strict
catalog is a hard error.
