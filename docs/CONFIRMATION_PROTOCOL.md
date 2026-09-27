# Sealed one-shot confirmation protocol

The full development gate, silver-label schema, feature contract, thresholds and hard rules must pass and be frozen before confirmation is **evaluated**. A preregistered set may be collected early with `sealed=true`, but its predictions and outcomes must remain unopened. Confirmation data must not be used for training, model selection, threshold selection, early stopping, feature design, rule changes or schema changes.

## Current preregistered sealed set

- Tasks: LIBERO task 16, task 45 and task 73. None may be inspected during development.
- Initial states: state indices 0–9 for each task.
- Intervention moments: the same two frozen block-aligned moments used by the paired collector.
- Per-group arms: `clean_a`, `clean_b` (QC only), `visual_occlusion`, `object_shift`, `execution_contact_deviation`, `unknown_cross_view`, `unknown_action_record`, and `unknown_low_magnitude`.
- Total: `3 tasks × 10 states × 2 moments × 8 arms = 480` raw branches; 420 model-evaluation samples and 60 `clean_b` QC branches.
- Three exclusive collection lanes own one task each: lane A/task16, lane B/task45 and lane C/task73.
- Pilot: state 0 for all tasks (48 raw branches); all six strict shard audits must pass before states 1–9 start.

Collection must not compute model metrics. Any access to labels, predictions or aggregate outcomes for scientific development permanently retires this set from confirmation status.

## Integrity gates

Every group must pass the existing paired-counterfactual audit:

- exact shared runtime-snapshot start and shared planned action block;
- `clean_a` versus `clean_b` simulation, dual-camera and proprioception gates;
- unique group and arm keys;
- complete files and hashes;
- task/state groups confined to confirmation;
- QC branches excluded from model metrics;
- no simulator state, intervention label or outcome label in deployment features.

An infrastructure failure stops the run and may be repaired only if no confirmation prediction or outcome was exposed. A scientific failure after legitimate unsealing is final for that frozen model and must not trigger tuning on tasks 16, 45 or 73.

## Frozen engineering gates

- five-class Macro-F1 at least 0.60;
- unknown recall at least 0.60;
- recall at least 0.50 for each retained learned factor;
- action-record hard-rule accuracy exactly 1.00;
- confirmation remains isolated from all fitting and calibration.

`cause_unresolved / evidence_insufficient` is reported as an abstention diagnostic. It is not a learned physical-cause head, and the low-magnitude intervention metadata must not be presented as a deployable prediction.

Passing this protocol permits D19–D21 closed-loop candidate-selection validation. It does not by itself establish task-success or safety improvements.
