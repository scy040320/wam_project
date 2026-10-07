# Command-provenance, predicate-scoped utility trial

This experiment adds a bounded terminal-choice utility correction after the
complete immutable V8 policy's original arbitration. It does not replace that
policy with the earlier stand-alone ranker. Zero gain is an exact identity.

## Input contract

- Freeze the attributor, V8 backbone, gates, routing, candidate actions and
  original 144-train/48-consumed-validation split. No new policy queries or
  actions are executed.
- Read aligned, finite requested/applied commands from the previous full
  action block; preserve source IDs, reliability and decision-time alignment.
  Command discrepancy is not evidence that physical contact failed.
- Carry historical missing facts, separating false from unknown. Derive
  contact and subject–anchor masks independently. Fresh observation truth
  remains authoritative; candidate forecasts never certify current recovery.
- Candidate eligibility remains governed by the frozen hard gate. The new
  score channel only compares admitted, source-comparable alternatives.
  Pairwise correction cap is 0.10; the minimum original switching margin is
  0.01. Thresholds are not reduced to produce a switch.
- Actual outcomes supervise terminal choice utility and successful costs.
  They do not enter deployment features or admission masks. No physical
  recovery probability head is trained without qualified recovery labels.

## Fit and controls

One fit uses 1,200 fixed optimization steps, learning rate 0.01 and L2 0.1.
Gain is selected using training data only; no hyperparameter is selected on
validation. Full/masked/shuffled/no-dependency/without-command modes affect
only the added channel, retaining the original V8 route as the common anchor.
Shuffled donors remain within the same split. They test the new channel, not
the necessity of the entire frozen policy's original attribution mechanism.

The source audit permits one corrective pair, 13 protective pairs and 61
successful-cost pairs. All 49 zero-coverage pools remain in the denominator.
The selected gain is zero: base preservation passes, new utility gain fails.
No true closed-loop execution is claimed from these offline pool selections.

## Artifacts and reproduction boundary

The deployment implementation is `wam_reranking/command_predicate_utility.py`.
`scripts/train_command_predicate_utility.py` and its launcher preserve the
actual single-fit engineering configuration; the source audit is
`scripts/audit_command_predicate_readiness.py`.

These runners reference immutable cloud experiment artifacts, including the
complete V8 overlay, frozen model files and audited candidate recordings.
Those artifacts are required separately; a fresh repository clone alone is
not a runnable reproduction of this preserved trial. Local implementation
tests do not require the cloud data. Large raw trajectories and private host
configuration are not shipped as public training data.

The compact result JSON records actual counts, claim limits and SHA256 pins.
Do not promote this model as a new beneficial policy, rerun scientific
failures, or overwrite previous artifacts to improve the reported result.
