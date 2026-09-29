# Candidate Pool Screening V2 (preregistered 2026-09-29)

## Purpose

The previous fixed pool was bimodal: 16/24 scenarios contained no successful
candidate and 7/24 contained eight successful candidates.  This protocol
selects task templates on one calibration state and evaluates them on a
different state.  It is a pool-design protocol, not a method-result filter.

## Phase S0: clean competence screen

- Use the 16 tasks frozen in `configs/candidate_pool_screening_v2.json`.
- Use calibration state 45 only, clean condition only, one fixed block boundary.
- Generate K=4 candidates with fixed seeds and execute every candidate from the
  same runtime snapshot.
- Do not train, calibrate weights, or change candidate-effect rules with S0.

Each task is assigned exactly one stratum:

1. `generator_limited`: zero successful candidates;
2. `selection_discriminative`: 1--3 successful candidates;
3. `cost_discriminative`: all four candidates succeed, but successful outcomes
   differ by at least 3 executed steps or 1 continuation WAM call;
4. `trivial`: all four succeed without the above cost diversity.

Tasks in strata 2 or 3 qualify for Phase S1. All 16 tasks and all four strata
must be reported. No replacement task may be chosen after viewing S0.

## Phase S1: held-state K=8 pool

- Use only the S0-qualified task templates, but switch to held state 46.
- Evaluate clean, object-shift, and execution/contact-deviation conditions.
- Generate K=8 candidates per scenario with fixed seeds.
- Report all held-state scenarios, including those that become generator
  limited or trivial.

Primary metrics:

- candidate-pool coverage;
- success-count histogram from 0 through 8;
- selector success conditional on pool coverage;
- selector success on the 1--7-success discriminative subset;
- cost-optimal selection rate among scenarios with at least two successful
  candidates and a step/call difference;
- total task success, strict harm, executed steps, and WAM calls.

## Separation from formal confirmation

S0 and S1 are development evidence. They may validate the parser and choose a
stable pool design, but they are not the final independent confirmation. A
later confirmation must freeze parser, weights, thresholds, tasks, and states
before any rollout outcome is inspected.

