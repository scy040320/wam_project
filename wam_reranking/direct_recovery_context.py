"""Outcome-free bridge from audited before-context to direct recovery scores.

This is preparation, not a trained deployment. A missing predicate is credited
to a cause only when the *current* attribution-owned fact has a matching,
legal invalidation path in this block. Initial unknowns, candidate forecasts,
and refreshed observations cannot become attribution-created recovery needs.
No teacher, executed endpoint or candidate-success argument exists here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import CandidateEffect, ConsistencyFactor, TriValue
from .direct_recovery import (
    CandidateRecoveryInput, CandidateRequirement, RecoveryNeed,
    extract_raw_candidate_features,
)
from .recovery_contract import (
    ROUTES, ROUTE_ROOTS, VISUAL_EVIDENCE_KEYS, CandidateRecoveryTrace,
    RecoveryContext, command_requirements, trace_candidate_recovery,
)

BRIDGE_SCHEMA = "predicted_before_context_to_direct_recovery_input_v1"
VISIBILITY = frozenset(("target_visible", "target_pose_current", "receptacle_visible"))
REASONS = {
    "visual_occlusion": frozenset((ConsistencyFactor.OBSERVATION_RELIABLE.value,)),
    "object_shift": frozenset((ConsistencyFactor.WORLD_STATE_CONSISTENT.value,)),
    "execution_contact_deviation": frozenset((ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value,)),
    "unknown": frozenset((ConsistencyFactor.CAUSE_RESOLVED.value,
                           ConsistencyFactor.OBSERVATION_RELIABLE.value)),
}


@dataclass(frozen=True)
class DirectRecoveryBridge:
    inputs: tuple[CandidateRecoveryInput, ...]
    traces: tuple[CandidateRecoveryTrace, ...]
    need_provenance: tuple[dict, ...]
    schema: str = BRIDGE_SCHEMA


def _valid_path(context, cause, predicate, change):
    path = tuple(change.propagation_path)
    if not path or path[-1] != predicate:
        return False
    if cause == "unknown":
        # Epistemic evidence need, never a guessed physical descendant.
        return predicate in VISIBILITY and change.direct and path == (predicate,)
    roots = ROUTE_ROOTS[cause]
    if path[0] not in roots:
        return False
    if cause == "visual_occlusion":
        return predicate in VISIBILITY and change.direct and path == (predicate,)
    if change.direct:
        return len(path) == 1 and predicate in roots
    if len(path) < 2 or len(set(path)) != len(path):
        return False
    edges = set(context.graph.edges)
    return all(edge in edges for edge in zip(path, path[1:]))


def build_direct_recovery_inputs(context: RecoveryContext,
                                 effects: Sequence[CandidateEffect],
                                 actions: Sequence[np.ndarray],
                                 backbone_scores: Sequence[float]) -> DirectRecoveryBridge:
    """Preserve common candidate B; derive cause needs from before-state only.

    Candidate use times come exclusively from its planned commands and
    declared required facts. Route mass inherits the existing recovery trace.
    This first bridge admits only facts that were true immediately before this
    block's invalidation. Repeated UNKNOWN->UNKNOWN updates cannot establish
    attribution-created missingness; earlier-block needs require a separate
    preserved before-provenance contract, not inference from a rewritten source.
    Need weight = route mass * invalidation confidence. Candidate requirement
    confidence remains separate and is multiplied ONCE by the existing scorer.
    This fixed formula is not calibrated against candidate execution results.
    Matching a need does not admit a head, override a hard gate, restore belief
    or establish an entry prerequisite using a future prediction.
    """
    if not isinstance(context, RecoveryContext) or not context.attribution.source_block_id:
        raise ValueError("A referenced, predicted before-context is required")
    if not effects or len(effects) != len(actions) or len(effects) != len(backbone_scores):
        raise ValueError("Complete aligned candidate effects/plans/base scores required")
    if any(not isinstance(e, CandidateEffect) for e in effects):
        raise ValueError("Typed preexecution effects required")
    if len({e.candidate_id for e in effects}) != len(effects):
        raise ValueError("Duplicate candidate identity")
    inputs, traces, audit = [], [], []
    source_id = context.attribution.source_block_id
    for effect, plan, score in zip(effects, actions, backbone_scores, strict=True):
        trace = trace_candidate_recovery(context, effect, plan)
        uses, _ = command_requirements(effect, plan)
        requirements = tuple(CandidateRequirement(u.predicate, u.at_step, u.confidence) for u in uses)
        features = dict(zip(trace.feature_names, trace.features, strict=True))
        # Only the existing allowlisted predicted-visual proxies can enter B.
        visual = {k: effect.evidence[k] for k in VISUAL_EVIDENCE_KEYS if k in effect.evidence}
        raw = extract_raw_candidate_features(trace.features, original_visual=visual)
        needs = []
        for use in uses:
            fact = context.belief.facts[use.predicate]
            if (fact.source != "attribution" or fact.value is TriValue.TRUE
                    or fact.updated_at_block != context.block_index
                    or source_id not in fact.evidence_ids):
                continue
            for cause in ROUTES:
                if cause == "normal" or (cause in ("unknown", "visual_occlusion")
                                         and use.predicate not in VISIBILITY):
                    continue
                # The route-weighted need bit is zero for unscoped/disabled causes.
                route_mass = float(features[f"{cause}.{use.predicate}.need"])
                if not np.isfinite(route_mass) or not 0 <= route_mass <= 1:
                    raise ValueError("Invalid source-scoped route mass")
                if route_mass == 0 or use.confidence == 0:
                    continue
                eligible = [c for c in context.changes
                    if c.predicate == use.predicate and c.reason in REASONS[cause]
                    and c.evidence_id == source_id and c.new_value is not TriValue.TRUE
                    and c.old_value is TriValue.TRUE and c.old_confidence > 0
                    and _valid_path(context, cause, use.predicate, c)]
                if not eligible:
                    continue
                # Deterministic maximum confidence; tied paths use lexical order.
                eligible.sort(key=lambda c: (-c.new_confidence, tuple(c.propagation_path)))
                change = eligible[0]
                if not np.isfinite(change.new_confidence) or not 0 <= change.new_confidence <= 1:
                    raise ValueError("Invalid before invalidation confidence")
                weight = route_mass * change.new_confidence
                if weight == 0:
                    continue
                path = tuple(change.propagation_path)
                needs.append(RecoveryNeed(cause, use.predicate, weight, use.at_step,
                    propagated=not change.direct,
                    evidence_ids=(source_id, f"before:block:{context.block_index}",
                                  "dependency:" + cause + ":" + "->".join(path))))
                audit.append(dict(candidate_id=effect.candidate_id, cause=cause,
                    predicate=use.predicate, required_at_step=use.at_step,
                    route_mass=route_mass, invalidation_confidence=change.new_confidence,
                    requirement_confidence=use.confidence, weight=weight,
                    scorer_effective_need_weight=weight * use.confidence,
                    source_block_id=source_id, propagation_path=list(path),
                    raw_candidate_fingerprint=raw.fingerprint(),
                    source="predicted_attribution_and_preexecution_belief"))
        inputs.append(CandidateRecoveryInput(effect.candidate_id, score, raw, tuple(needs), requirements))
        traces.append(trace)
    return DirectRecoveryBridge(tuple(inputs), tuple(traces), tuple(audit))
