# CF-WAM: Belief-Constrained Candidate Reranking

CF-WAM is a candidate-selection and action-refinement framework for World Action Models (WAMs). It augments Cosmos Policy Best-of-N inference with hierarchical mismatch attribution, an explicit task belief, dependency-aware feasibility checks, and auditable candidate reranking.

The research question is:

> Can structured belief constraints select safer and more effective action blocks than value-only Best-of-N, under the same candidate and execution budget?

## Method

```mermaid
flowchart TD
    I[Completed action-block evidence<br/>imagined terminal observation<br/>real terminal observation<br/>planned and executed actions<br/>proprioception and task language]
    I --> A[Hierarchical multi-label attributor]
    A --> F[Consistency factors<br/>observation · world state<br/>execution/contact · task stage<br/>cause resolved]
    F --> B[Object-centric belief update<br/>true · false · unknown<br/>confidence and provenance]
    B --> D[Dependency DAG propagation]

    O[Current RGB · wrist RGB<br/>proprioception · instruction]
    O --> C[Cosmos Policy Best-of-N<br/>generate K action-block candidates<br/>predicted futures and values]
    C --> E[Auditable candidate-effect parser<br/>required facts · proposed effects · stage]

    D --> G[Hard prerequisite gate]
    E --> G
    G --> R[Belief-aware reranking<br/>value + compatibility + progress<br/>- dependency risk - uncertainty]
    R --> X[Execute selected action block]
    R --> Q[Reobserve · requery · safe reject]
```

The official Cosmos rule is `argmax(value)`. CF-WAM first removes candidates that contradict high-confidence task facts, then ranks only the feasible set. A high predicted value cannot override a violated grasp, pose, reachability, or placement prerequisite.

## Current implementation

| Component | Status |
|---|---|
| Cosmos multi-query and value-selection interface | Verified |
| Block-aligned development data | 520 auxiliary samples plus paired formal train/validation data; roles and groups remain isolated |
| Deployment feature contract | Frozen observations, imagined terminal frames, actions and proprioception; simulator GT excluded |
| Hierarchical attribution interface | D18-v9 development contract uses target-conditioned local residuals, auditable command evidence and cross-task contrastive regularization under silver schema v7 |
| Evidence routing | Action-record reliability and explicit requested-versus-applied command deviation are auditable hard routes; unsupported causes abstain as `cause_unresolved / evidence_insufficient` |
| Development gate | **Blocked**: v9 validation Macro-F1 0.8347, normal recall 0.7625 and unknown recall 0.7308 pass, but the learned execution/contact factor recall is 0 |
| Diagnostic tasks | Task 23/58 are consumed development diagnostics, never untouched confirmation; task58 object-shift recall remains 0 |
| Engineering gate | Blocked. A separately preregistered task16/45/73 set may be collected sealed, but cannot be evaluated until the development gate passes and the model contract is frozen |
| Three-valued object-centric belief | Implemented; calibrated `true/false/unknown` factor states are preserved |
| Dependency DAG and minimal invalidation | Implemented |
| Rule-based candidate-effect parser | Implemented |
| Hard gate, soft score and explicit fallback | Implemented and tested; score weights intentionally blocked from deployment calibration |
| Closed-loop Cosmos candidate reranking | Next integration milestone |
| Belief-conditioned action refiner | Interface only; training data not collected yet |

The repository contains mechanism code, public contracts and tests, not checkpoints, datasets, or a final benchmark claim. D18-v9 uses deterministic protocol-derived silver labels. Its development gate currently fails because one learned factor has zero recall, so D19–D21 and closed-loop task evaluation remain blocked. A sealed confirmation collection is not evidence until it is legitimately unsealed and evaluated after development freezes.

## Environment

The target experimental stack is:

- [Cosmos Policy](https://github.com/nvlabs/cosmos-policy) as the WAM and action-block generator;
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) as the simulation benchmark;
- native `16 × 7` Cosmos action blocks;
- Cosmos Best-of-N candidates controlled by `num_queries_best_of_n`;
- frozen DINOv2-small visual features for the first attribution model;
- Python 3.10+.

This repository intentionally does not redistribute Cosmos Policy, LIBERO, DINOv2, model checkpoints, datasets, simulator assets, or machine-specific launch scripts. Install those components from their upstream projects for full experiments. The standalone mechanism and tests require only NumPy.

## Installation and tests

```bash
git clone https://github.com/scy040320/wam_project.git
cd wam_project
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Package layout

```text
wam_reranking/
  contracts.py          attribution, belief, candidate and refiner contracts
  belief.py             belief update and dependency propagation
  candidate_effects.py  interpretable 16×7 action-block parser
  reranker.py           hard feasibility gate, soft score and fallback
  refiner.py             bounded action-residual application interface
  paired_audit.py        paired-clean physical gates and cache-only diagnostics
  evidence_routing.py    hard action/command evidence routes and unresolved-cause abstention
  target_localization.py target-conditioned residual pooling from deployable RGB/text evidence
  contrastive.py         cross-task factor contrastive objective
configs/
  task_bindings.json
  score_weights.template.json
  d18_v5_frozen_contract.json
  d18_v9_sealed_confirmation_protocol.json
docs/
  METHOD.md
  EVALUATION_PROTOCOL.md
  CONFIRMATION_PROTOCOL.md
tests/
  test_belief_reranking.py
```

## Evaluation contract

All methods must use the same Cosmos checkpoint, candidate count `K`, query budget, 16-step horizon, initial states and intervention schedule.

Required comparisons include:

- Cosmos `K=1` direct execution;
- Cosmos Best-of-N with value-only selection;
- binary mismatch rejection/replanning;
- coarse attribution without dependency propagation;
- hierarchical attribution without dependency propagation;
- full belief-constrained reranking.

Primary metrics are task success, clean-scene degradation, false rejection and unsafe-action proxies. Successful-run action steps, WAM calls and latency are reported as secondary efficiency metrics. Thresholds and score weights must be frozen on a development split before held-out evaluation.

## Research roadmap

1. Train and validate hierarchical multi-label attribution.
2. Demonstrate that belief constraints change candidate selection and improve closed-loop task performance.
3. Collect matched `(original candidate, execution outcome, better candidate or corrected action)` records.
4. Train an external belief-conditioned action refiner that predicts a bounded `16 × 7` action residual.
5. Consider a LoRA or cross-attention adapter in the Cosmos action decoder only if candidate-set coverage is the verified bottleneck.

```text
explain mismatch
    → update belief
    → constrain candidates
    → refine actions
    → condition the generator only when necessary
```

## Scope

- Attribution currently uses complete action-block evidence; within-block prediction is not claimed.
- The candidate-effect parser is intentionally rule based and auditable in the first version.
- Unresolved or conflicting evidence maps to `unknown` and a conservative fallback.
- Score weights remain blocked from deployment until calibrated on the development split.
- Calibration confidence does not turn silver supervision into human-validated causal truth.
- D18-v9's development contract does not yet pass: the learned execution/contact factor recall is zero, despite passing coarse-class, normal, unknown, visual, object-state and cross-view thresholds.
- Task58 remains a development diagnostic with zero object-shift recall. It must be repaired using development data only.
- The task16/task45/task73 confirmation set is preregistered and may be pre-collected with `sealed=true`; it cannot be opened, evaluated or used for development until the full development gate passes and the model/schema/thresholds are frozen.
- Action-record reliability is not learned: the acquisition-contract availability bit applies a hard `unknown` rule and is masked from the neural input.
- Existing features did not meet the preregistered feasibility thresholds for a 5 mm weak displacement. Those records map to `cause_unresolved / evidence_insufficient`; the method does not claim to identify their hidden physical cause.
- Paired counterfactual data uses `clean_a` versus QC-only `clean_b` after identical snapshot restoration. Cached-query versus forced re-observation differences are diagnostic and never replace the unchanged clean-pair hard gates.
- No project license has been selected yet; a license will be added before formal open-source release.
