# Frozen attribution and joint selection development result

The 2026-10-04 result uses frozen attribution V8 with checkpoint SHA256
`68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec`.
Raw images, checkpoints and simulator assets are not redistributed in this
repository. The membership and result manifests are archived separately.

The public mechanism includes supervised-factor masking, source-scoped paired
loss membership, temporal instance ordinal locking, candidate predicate gates,
attribution-conditioned utility and the value-preserving safe residual policy.
Non-promoted diagnostic repair variants are not production interfaces.

## Reproduction inputs

Supply the immutable candidate outcomes, a bundle with paired attribution and
candidate visual evidence, and a passing exact hard-gate reconstruction audit.
The bundle's deployment features must exclude executed success and cost targets.
The six-task binding and stage contract is in `prepare_joint_bundle.py`.
Visual feature extraction additionally requires the upstream experiment
environment and its frozen Torch/Transformers checkpoints.

With an audited bundle and the same dependency versions:

```bash
python scripts/train_safe_residual_ranker.py \
  --bundle BUNDLE_DIR \
  --contract scripts/train_d21_candidate_utility.py \
  --hard-gate-audit HARD_GATE_AUDIT.json \
  --output-root TRAINING_DIR

python scripts/audit_selector_replay.py \
  --source-root CANDIDATE_DATA_DIR --bundle BUNDLE_DIR \
  --prepare-script scripts/prepare_joint_bundle.py \
  --model TRAINING_DIR/candidate_utility_model.json \
  --training-report TRAINING_DIR/training_report.json \
  --output REPLAY_AUDIT.json
```

The replay script deliberately expects the frozen 192-pool protocol. It checks
policy decisions and observed/unobserved fallback accounting against the report;
it does not establish independent generalization. Fitting/calibration uses train
only. The validation split was consumed in earlier development.

Current results: train learned/value/oracle = 102/101/107 out of144;
validation = 34/33/36 out of48, each with one gain and no success harm.
The complete attribution gate remains failed (task16 2/5), Main20 qualification
remains failed, and Gate3 is not complete. No new experiment is implied by this
repository update.
