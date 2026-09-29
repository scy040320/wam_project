# Candidate-Pool Screening V3 (Frozen Cached-Language Revision)

V3 preserves the two-stage qualification logic of V2 but uses only exact
LIBERO-90 instructions already encoded by the same frozen T5-11B encoder.
The revision was made before new rollout outcomes were observed because the
cloud container could not load a 45GB monolithic T5 checkpoint within its
62GiB cgroup. Both failed T5 attempts remain archived as infrastructure
failures with zero encoded prompts and zero sampled candidates.

## S0

- 16 tasks fixed in `configs/candidate_pool_screening_v3_cached.json`;
- state45, clean condition, K=4;
- task46 is a previously exposed mixed-outcome sentinel and is excluded from
  the count of newly qualified tasks;
- 0/K success is generator-limited;
- 1..K-1 success is selection-discriminative;
- K/K success with at least 3 executed-step spread or one WAM-call spread is
  cost-discriminative;
- remaining K/K success is trivial.

No value, effect, attribution or outcome score is used to change this
classification after launch.

## S1

Only non-sentinel S0 tasks in the selection- or cost-discriminative strata
may enter S1. S1 uses state46, K=8, and clean/object-shift/execution-contact
conditions. It is a development validation, not the final independent test.

All candidate pools, including zero-success and all-success pools, remain in
the report. Selector accuracy is reported conditional on at least one
successful candidate, while overall success and harm are reported across all
scenarios.
