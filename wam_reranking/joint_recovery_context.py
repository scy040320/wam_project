"""Outcome-free bridge for the separate whole-run recovery contract.

The caller freezes desired boundary goals before querying candidates. Goals
are admitted here only for source-owned deficits in the current before state;
candidate predictions and offline labels cannot manufacture that deficit.
This module does not fit, certify a teacher label or unlock a hard gate.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from .contracts import CandidateEffect, CoarseCause, Stage, TriValue
from .direct_recovery_context import REASONS, VISIBILITY, _valid_path
from .joint_recovery_contract import OBSERVER, PREDICATES
from .recovery_contract import RecoveryContext, ROUTES, command_requirements, trace_candidate_recovery

SCHEMA = "source_owned_whole_recovery_context_bridge_v1"


def _source_needs(context, effect, plan, objectives):
    trace = trace_candidate_recovery(context, effect, plan)
    features = dict(zip(trace.feature_names, trace.features, strict=True))
    output = []
    for objective in objectives:
        pred = objective["predicate"]
        fact = context.belief.facts[pred]
        source = context.attribution.source_block_id
        if (fact.source != "attribution" or fact.value is TriValue.TRUE
            or fact.updated_at_block != context.block_index or source not in fact.evidence_ids):
            continue
        for cause in ROUTES:
            if cause == "normal" or (cause in ("unknown", "visual_occlusion") and pred not in VISIBILITY):
                continue
            mass = float(features[f"{cause}.{pred}.need"])
            if not np.isfinite(mass) or not 0 <= mass <= 1:
                raise ValueError("Invalid before route mass")
            changes = [c for c in context.changes if c.predicate == pred
                and c.reason in REASONS[cause] and c.evidence_id == source
                and c.old_value is TriValue.TRUE and c.old_confidence > 0
                and c.new_value is not TriValue.TRUE and _valid_path(context, cause, pred, c)]
            if not changes or mass == 0:
                continue
            changes.sort(key=lambda c: (-c.new_confidence, tuple(c.propagation_path)))
            change = changes[0]
            weight = mass * change.new_confidence
            if weight <= 0:
                continue
            output.append(dict(cause=cause, predicate=pred, deadline=objective["deadline"],
                kind=objective["kind"], weight=weight, propagated=not change.direct,
                path=list(change.propagation_path), origin="current_attribution_invalidation",
                old_value=change.old_value.value, old_confidence=change.old_confidence,
                new_value=change.new_value.value, source_block_id=source))
    return output


def build_whole_recovery_tables(context: RecoveryContext,
    effects: Sequence[CandidateEffect], actions: Sequence[np.ndarray], *,
    frozen_boundary_goals: Sequence[str] = (), information_acquisition=False,
    before_observer_available=None):
    """Construct requirements, goals and needs with no execution-result input.

    Required use0..15 is inherited from the original planned action. The
    independent boundary16 goal cannot change any such deadline. Physical
    boundary goals require a real TRUE-to-invalid change, not initial unknown.
    Observer availability is a distinct epistemic goal and never a fabricated
    physical state. Goal tables are not learned labels or permissions to act.
    """
    if not isinstance(context, RecoveryContext) or not context.attribution.source_block_id:
        raise ValueError("Referenced before recovery context required")
    if not effects or len(effects) != len(actions) or len({e.candidate_id for e in effects}) != len(effects):
        raise ValueError("Nonempty aligned unique candidates required")
    goals_requested = tuple(frozen_boundary_goals)
    if len(set(goals_requested)) != len(goals_requested) or any(p not in PREDICATES for p in goals_requested):
        raise ValueError("Unique frozen semantic boundary goals required")
    if type(information_acquisition) is not bool or (before_observer_available is not None and type(before_observer_available) is not bool):
        raise ValueError("Literal before-only information-acquisition contract required")
    if context.attribution.projected_cause in (CoarseCause.UNKNOWN, CoarseCause.VISUAL_OCCLUSION):
        if any(p not in VISIBILITY for p in goals_requested):
            raise ValueError("Unknown/occlusion cannot prescribe a guessed physical recovery")
    result = []
    for effect, plan in zip(effects, actions, strict=True):
        uses, _ = command_requirements(effect, plan)
        requirements = [dict(predicate=u.predicate, deadline=u.at_step,
            confidence=u.confidence, kind="required_use") for u in uses]
        req_needs = _source_needs(context, effect, plan, requirements)
        declared = [dict(predicate=p, deadline=16, confidence=1., kind="next_boundary_goal")
                    for p in goals_requested]
        goal_needs = _source_needs(context, effect, plan, declared)
        # This is not outcome-based selection: only before provenance decides
        # whether a frozen requested goal is genuinely attribution-created.
        keys = {n["predicate"] for n in goal_needs}
        goals = [g for g in declared if g["predicate"] in keys]
        if (information_acquisition and before_observer_available is False
            and context.attribution.confidence > 0
            and context.attribution.projected_cause in (CoarseCause.UNKNOWN, CoarseCause.VISUAL_OCCLUSION)):
            cause = context.attribution.projected_cause.value
            goals.append(dict(predicate=OBSERVER, deadline=16, confidence=1., kind="epistemic_goal"))
            goal_needs.append(dict(cause=cause, predicate=OBSERVER, deadline=16, kind="epistemic_goal",
                weight=float(context.attribution.confidence), propagated=False, path=[OBSERVER],
                origin="before_epistemic_insufficiency", old_value="unknown", old_confidence=0.,
                new_value="unknown", source_block_id=context.attribution.source_block_id))
        purpose = ("information_acquisition" if effect.stage is Stage.OBSERVE
                   else "task_with_observation_recovery") if information_acquisition else "task_action"
        result.append(dict(candidate_id=effect.candidate_id, stage=effect.stage.value,
            purpose=purpose,
            requirements=requirements, goals=goals, needs=req_needs + goal_needs,
            schema=SCHEMA, hard_gate_unchanged=True,
            terminal_goal_is_not_entry_or_earlier_use_evidence=True))
    return tuple(result)
