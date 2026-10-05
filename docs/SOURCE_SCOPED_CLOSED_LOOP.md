# Frozen source-scoped full policy

The source-scoped policy V6 is frozen separately from the earlier joint
selector. It does not modify Cosmos or its checkpoints. Public implementation
files use functional names; immutable schema/run identities retain their
historical names for traceability.

## Deployment path

1. Extract previous-block predicted/actual and command evidence with the
   frozen attributor. Do not supply simulator state or intervention labels.
2. `prepare_source_scoped_decisions` masks unsupported physical interpretations,
   caps belief changes by their individual factor confidences, and retains
   directly observed FALSE prerequisites. It does not equate coarse unknown
   confidence with world-state certainty.
3. Parse each candidate's current/predicted subject–anchor and contact evidence.
   Articulated effects do not require carrying the drawer. Endpoint progress
   regression is a soft proxy, not a physical safety certificate.
4. Apply predicate gates. Unknown-only rejection can preserve the value and
   candidate-only anchors; explicit FALSE/contact contradictions are not bypassed.
5. `select_evidence_residual` uses the candidate backbone, except that unreliable
   observation or unresolved-but-consistent physical factors route to value.
   The frozen cause-residual gain is **0.0**; this version does not demonstrate
   a learned attribution score correction.

Candidate-only uses the same frozen backbone and raw candidate effects,
without attribution probabilities, belief gates, or cause interactions.
Official value-only uses just the largest value. All methods use K=4,
internal best-of-N=1 and 16-step native action chunks. The backbone is trained
on the original 144 development pools only; validation has 48 pools.

Model records are `configs/source_scoped_candidate_backbone.json` and
`configs/source_scoped_cause_residual.json`. Upstream image feature extraction
and robotic execution require the separately preserved simulator/data bundle.
The previous selector is not silently replaced by this policy.

## Execution pairing and accounting

The 64 scenes are LIBERO-90 tasks 0/9/46/57, four fixed states and four
conditions each. They share trusted complete runtime root snapshots and
initial query realizations. Every arm executes subsequent block boundaries
using its own observations. Four arms were re-executed in V6: value-only,
candidate-only, V4 full reference and V6 full.

`SharedQueryCache` keys only on observation hash, seed, task language and frozen
model signature. Payloads contain actions, predictions and value, never GT or
outcomes. Equal requests receive exactly the same checked arrays. Cache hits
are charged all logical WAM queries to each arm. Report physical generation
and cache hits separately; replay query seconds are not end-to-end policy latency.

Every arm has 400 total policy steps, including the shared 48-step prefix;
post-fork budget is 352 steps and at most 92 logical WAM calls. Prefix cost
is three calls. Fallback, when requested, is really executed and charged.

The result record explicitly reports quality passing but repair acceptance
failing: 38/64 full successes, not historical 40, and one success harm relative
to candidate-only. The observed union of baseline successes is 39; no claim
is made that all 64 candidate pools have an attainable success upper bound of 39.
These are consumed development regression results, not independent Gate3.

## Preservation and tests

The original run, data, failed versions and attributor checkpoint remain
unchanged. A small source/config dependency archive plus results is retained
locally and in cloud preservation storage. Its SHA256 is
`eda80ca5f7167ad1235f11a24029aa64ee4081887f9106c7155c76383d55cf4f`.
Large raw data and checkpoints are not distributed through GitHub.

```bash
python -m unittest discover -s tests -p test_evidence_residual.py -v
python -m unittest discover -s tests -p test_source_scoped_policy.py -v
```

Tests cover gate preservation, confidence scoping, exact request reuse,
cache identity separation and payload tampering. They are engineering checks,
not proof of zero deployment harm or module necessity.
