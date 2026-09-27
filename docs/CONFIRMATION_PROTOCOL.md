# D18-v5 one-shot confirmation protocol

The D18-v5 development model, silver-label schema, feature contract, thresholds and hard rules must be frozen before this protocol starts. Confirmation data must not be used for training, model selection, threshold selection, early stopping, feature design, rule changes or schema changes.

## Proposed untouched set

- Tasks: LIBERO task 8 and task 9. Neither task may be inspected during development.
- Initial states: state indices 0–9 for each task.
- Intervention moments: the same two frozen block-aligned moments used by the paired collector.
- Per-group arms: `clean_a`, `clean_b` (QC only), `visual_occlusion`, `object_shift`, `execution_contact_deviation`, `unknown_cross_view`, `unknown_action_record`, and `unknown_low_magnitude`.
- Total: `2 tasks × 10 states × 2 moments × 8 arms = 320` raw branches; 280 model-evaluation samples and 40 `clean_b` QC branches.

Before touching task 8 or task 9, the unchanged collector and auditor may run a dry-run only on already exposed development tasks. A dry-run may test file integrity, snapshot restoration, arm scheduling and metric computation, but it may not change the frozen model contract.

## Integrity gates

Every group must pass the existing paired-counterfactual audit:

- exact shared runtime-snapshot start and shared planned action block;
- `clean_a` versus `clean_b` simulation, dual-camera and proprioception gates;
- unique group and arm keys;
- complete files and hashes;
- task/state groups confined to confirmation;
- QC branches excluded from model metrics;
- no simulator state, intervention label or outcome label in deployment features.

An infrastructure failure stops the run and may be repaired only if no confirmation prediction or outcome was exposed. A scientific failure is final for D18-v5 and must not trigger tuning on task 8 or task 9.

## Frozen engineering gates

- five-class Macro-F1 at least 0.60;
- unknown recall at least 0.60;
- recall at least 0.50 for each retained learned factor;
- action-record hard-rule accuracy exactly 1.00;
- confirmation remains isolated from all fitting and calibration.

`cause_unresolved / evidence_insufficient` is reported as an abstention diagnostic. It is not a learned physical-cause head, and the low-magnitude intervention metadata must not be presented as a deployable prediction.

Passing this protocol permits D19–D21 closed-loop candidate-selection validation. It does not by itself establish task-success or safety improvements.
