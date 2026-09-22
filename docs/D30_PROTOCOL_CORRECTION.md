# D30 protocol correction: native 16-step alignment

## Why the protocol changed

The released `Cosmos-Policy-LIBERO-Predict2-2B` checkpoint is trained with an
action chunk size of 16. The official evaluator requires the inference
`chunk_size` to equal that training value and recommends executing the complete
chunk. The checkpoint emits one predicted future associated with that native
action horizon; it does not expose a calibrated intermediate `t+4` future.

Earlier development runners changed `chunk_size` to 4. That did not merely
execute a shorter prefix: it changed how the action latent was decoded. It also
compared the resulting prediction with a four-step real observation. Those
runs remain useful for debugging storage, attribution routing, and safety-gate
behaviour, but their task-success numbers are not used as paper results.

A second audit found that the development runners stopped on LIBERO
`done=True` but recorded success only from `info["success"]`. The official
evaluator treats `done=True` as success. D30 and the corrected earlier runners
now use `done or info.get("success", False)`.

## Evidence

Using the frozen checkpoint and the official native horizon on LIBERO task 2:

| Initial-state seed | Success | Environment steps | WAM calls |
|---:|:---:|---:|---:|
| 40 | yes | 272 | 17 |
| 41 | yes | 223 | 14 |

The private experiment archive retains primary/wrist MP4 files, the exact trained chunk size, the
executed prefix, per-query value and latency, and the final success flag.

## Frozen D30 protocol

1. Cosmos remains frozen and decodes its native 16-action chunk.
2. All 16 planned actions are executed before residual construction.
3. Predicted future images are compared only with real observations at `t+16`.
4. The v1 attributor keeps its four-row action-residual ABI by using the first
   four plan-execution deltas; this is logged explicitly and is not described
   as a four-step controller.
5. An isolated model-unknown signal causes a four-step guarded hold and
   re-observation; a second consecutive unknown decision causes safe stop.
   Immediate safe stop is reserved for either simultaneous visual and
   action-execution evidence or model-unknown evidence supported by
   action-execution evidence.
6. Native-16 thresholds are selected only from task0/task1 validation seeds.
   Task2/task3 seeds 40--49 are reserved for qualification and evaluation.
7. No D27/D28 development result is used to select a D30 threshold.

## D30 gate outcome

- Native clean baseline: at least 5/10 successes on each of task2 and task3.
- Validation-only threshold manifest is frozen and hashed.
- Five-condition smoke completes without crashes or OOM.
- Known interventions use the expected recovery in at least 90% of smoke runs.
- Unknown is safely stopped in at least 18/20 evaluation runs.
- Later false safety stops on known interventions are at most 20%.
- Complete method clean success is no worse than one episode out of ten below
  the corresponding clean baseline.

The frozen V6 configuration was selected from 935 task0/task1 validation rows
with `control_horizon=16` and two temporal confirmations. The held-out full run
completed 100/100 episodes with the following results:

- known-intervention recovery: 59/60 (98.3%);
- unknown safe stop: 20/20;
- later false safety stop on known conditions: 2/60 (3.3%);
- clean false safety stop: 0/20;
- task2 clean success: 10/10, equal to the A-clean baseline;
- task3 clean success: 9/10, equal to the A-clean baseline.

All required decision JSON files and both camera videos were non-empty. There
was no OOM, dead loop, or runtime-fatal traceback. EGL destructor warnings
after completed episodes were treated as cleanup warnings, not run failures.

D30 therefore passes its frozen gate. D31 has not started and requires a
separately approved, same-seed/same-budget comparison protocol.
