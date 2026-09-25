# Paired recovery-treatment protocol

## Question and scope

Does treatment choice change task completion from an otherwise matched state?
This is offline treatment-complementarity collection, not online mismatch detection.
An oracle retrospectively selecting the best branch is not a deployable policy.

## Frozen design

- Two tasks, twelve development seeds (0–11), approach/grasp-attempt stages,
  four conditions and three treatments: 576 branches, 192 matched contexts.
- Generate all 24 clean reference trajectories and freeze anchors first. Do not
  select seeds by success; missing/invalid anchors stop the queue, not replacement.
- Earliest eligible approach: approaching, distance in (8,14] cm, no gripper contact.
  Grasp attempt: gripper/hand contact, or closing command within 8 cm; not secure grasp.
- Branch four steps after intervention, with at least one old action remaining.
- Shift vector [0.0625, -0.03, 0.005] m is a preset intensity, not a real-world prevalence claim.
- Translation-command noise [0.12,-0.12,0], clipped to action limits, lasts four steps.
- Central primary-camera occlusion lasts eight steps; wrist camera remains visible.
  The exact mask geometry is inherited from the experimental collector and is not
  implemented by the public contract utilities. Portable collector release is pending.
- Same exogenous schedule and identical replayed prefix across treatments.
  Check state, cameras, controller goal, gripper and RNG equality before branching.
- Continue retains the old tail. Reobserve holds a fixed end-effector target and
  gripper command for four steps, then queries Cosmos. Replan discards the tail
  and immediately queries Cosmos from fresh observations. Neither resets physics.
- Native 16-step planning continues after that single treatment; no continuous
  within-chunk detector is enabled.
- Six hours or 576 branches, whichever comes first; 15 minutes per rollout;
  stop on OOM, real errors, replay mismatch, corruption or less than 40 GiB free.

GT is permitted only in offline stage scheduling and separately stored risk labels.
It must not enter a future selector's deployed features. Seeds 0–7 are training,
8–9 validation, 10–11 development test, grouped across tasks and all variants.
Historical exposure is possible: these are not a pristine final paper test set.

## Alignment and accounting

Store actual observations (including sensor corruption), plan/executed actions,
actual executed length, prediction target time, query inputs and outcomes.
Do not compare t+16 predictions with t+4 observations. A discarded or changed tail
invalidates original-plan same-condition alignment; terminal partial blocks are
not full-horizon labels. Keep invalid/censored records with reasons.

Count newly generated prefix calls separately from branch calls. Inherited cached
prefixes are not a zero-cost deployed controller. Risk proxies such as contact
penetration are descriptive, not a validated physical-safety metric.

## Public analysis input

`python -m recovery_decision.analysis` reads a JSON list. Each row requires:

```json
{"task":0,"seed":0,"stage":"approach","condition":"object_shift",
 "arm":"continue","success":true,"environment_steps":250,
 "actual_new_calls":6,"wam_seconds":5.6}
```

Every context must have exactly one row for each of continue/reobserve4/replan.
Do not treat this example as a measured outcome. The tool reports treatment
rescues and harms versus continue. Step/call/inference-time deltas are reported
only when both branches succeed; post-branch remaining steps are not recovery time.
The tool checks triplets, not full video/NPZ integrity or replay equality: those
remain a separate mandatory collection audit. A full experiment audit must also
check the expected matrix against its immutable manifest.

## Completed pilot and limitations

The 144-branch pilot used seeds 25–27; do not merge it with the new collection
as if it were an independent test of a tuned rule. Successes were 45/48 continue,
46/48 reobserve, 47/48 replan. Object-shift successes were 9/12,12/12,11/12.
Reobserve rescued three failures but harmed two otherwise successful episodes;
immediate replan rescued two. All rescue events were in task0.

Among jointly successful pairs, reobserve used 169 more total steps over 43 pairs;
replan used seven fewer over 45 pairs. Branch calls increased by 32 and 35,
respectively. These do not support a general calling-cost reduction claim.

Before training, audit repeatability and normal-condition harm. Later compare a
learned selector against each fixed treatment and an observation-only selector.
Keep the independent delay experiment separate. No outcome-dependent intervention
changes, automatic training, or claimed online benefit are part of this release.
