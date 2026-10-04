# Data protocols

## D18 attribution data

The frozen V8 model combines source-scoped, audited legacy records, the
4,800-branch expansion, and qualified Main20 attribution records. After
immutable exclusions, exact overlap replacement and conflicting-supervision
checks, the formal membership is 5,200 train / 1,732 validation, not the sum
of raw branch counts. `clean_b` and unpromoted confirmation records are not
supervised samples. Earlier schemas and failed reports remain archived.

Identity is source + benchmark suite + task + state + moment + arm, never a
bare `sample_id` or `task_id`. A migration applies only to its source, and
replacement requires an audited exact group match. Group splits are immutable
unless an explicitly audited migration is recorded.

The frozen observability contract routes pixel-equivalent physical changes to
`evidence_insufficient` and masks unsupported physical-factor supervision.
Raw records remain immutable; exclusions and schema migrations are stored as
separate audit manifests.

## Candidate-ranking data

The current completed development comparison is frozen in
`configs/joint_development_192_protocol.json`: LIBERO-90 tasks
0/9/20/44/46/57, states0–7, moment0, four conditions, `K=4`.
It contains 144 train pools (states0–5), 48 validation pools (states6–7)
and 768 candidate outcomes. All 49 zero-coverage pools remain included.
This is previously consumed development data, not independent confirmation.

`configs/candidate_collection_protocol.json` preserves a historical collection
proposal whose pilot did not qualify. It must not be presented as a completed
896-pool dataset or restarted automatically. Main20 nested K16 stage
qualification also remains failed; formal collection is not open.

Each candidate record contains deployment-time attribution and belief inputs,
parsed candidate effects, official value, and post-execution success, steps,
WAM calls, latency, and risk proxies. The post-execution fields are targets and
metrics only.

Outcome-based qualification is a development procedure and must be reported
as such. Failed pools and tasks remain recorded. An independent experiment
requires a separately preregistered eligibility procedure and task set.
