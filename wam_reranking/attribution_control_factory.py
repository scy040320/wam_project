"""Outcome-free current-block attribution/belief controls, not need-weight ablations.

No models, fitting, Y, simulator state or future probability inputs. This
reruns the complete current verified block's attribution channel and propagation
from a common prior, not all attribution history of a reconstructed rollout.
It explicitly reports retained prior history and changed gate outcomes. Actual
dataset provenance, supervised admission and fair execution are separate gates.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

from .belief import DEFAULT_GRAPH, DependencyGraph
from .contracts import (AttributionOutput, BeliefChange, BeliefFact, BeliefState,
    CandidateEffect, CoarseCause, ConsistencyFactor, PREDICATES, Stage, TriValue)
from .evidence_residual import candidate_backbone_features, typed_gate_effect
from .joint_recovery_context import build_whole_recovery_tables
from .joint_recovery_contract import RESIDUAL_CAP
from .recovery_contract import (CurrentFactEvidence, RecoveryContext, ROUTES,
    prepare_recovery_context, trace_candidate_recovery)

SCHEMA = "complete_before_attribution_controls_v1"
MODES = ("full", "no_dag", "masked", "shuffled", "effect_only")
IDENTITY = ("dataset", "suite", "task", "state", "observed_block_id")
FORBIDDEN = frozenset(("success", "final_success", "terminal_success", "labels",
    "teacher", "actual_after", "private_physics", "simulator_gt", "outcomes"))


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return value


def _digest(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _no_outcome_keys(value):
    if isinstance(value, Mapping):
        if any(str(k).lower() in FORBIDDEN for k in value):
            raise ValueError("outcomes/private future fields are not control inputs")
        for child in value.values():
            _no_outcome_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _no_outcome_keys(child)


@dataclass(frozen=True)
class ControlPool:
    identity: Mapping
    pool_id: str
    split: str
    prior: BeliefState
    attribution: AttributionOutput
    observations: Sequence[CurrentFactEvidence]
    effects: Sequence[CandidateEffect]
    plans: Sequence[np.ndarray]
    official_values: Sequence[float]
    backbone_scores: Sequence[float]
    budget: Mapping
    block_index: int
    relation: str = "in"
    graph: DependencyGraph = DEFAULT_GRAPH
    frozen_boundary_goals: Sequence[str] = ()
    information_acquisition: bool = False
    before_observer_available: bool | None = None
    selection_wrapper_id: str = "shared_actual_before_gates_and_fixed_fallback_v1"
    fallback: str = "safe_reject"


def neutral_attribution(attribution: AttributionOutput) -> AttributionOutput:
    """Explicit experimental carrier, NOT a real NORMAL/UNKNOWN prediction.

    Old update_belief treats UNKNOWN even at confidence0 as state changes.
    Therefore this object MUST use prepare_masked_context, not ordinary prepare.
    Sensor quality/source remain real; learned class/factor channels are erased.
    """
    return AttributionOutput(
        factor_probs={f.value: .5 for f in ConsistencyFactor},
        class_probs={c.value: 1./len(CoarseCause) for c in CoarseCause},
        projected_cause=CoarseCause.UNKNOWN, confidence=0., entropy=math.log(len(CoarseCause)),
        evidence_quality=attribution.evidence_quality, source_block_id=attribution.source_block_id,
        factor_states={f.value: TriValue.UNKNOWN for f in ConsistencyFactor},
        factor_confidences={f.value: 0. for f in ConsistencyFactor})


def prepare_masked_context(*, prior, attribution, observations, block_index, graph):
    """Remove learned updates, retain actual quality/record evidence and facts.

    Mirrors original assumption sanitation and observation/record quality
    filtering. No fake consistent_block, no graph propagation of masked UNKNOWN.
    It does not add new quality-based invalidation of old facts to this arm.
    Masking removes only learned classification/factor channels, not the D18
    perception/quality module or real measured record and observation inputs.
    Hard record evidence must arrive explicitly as CurrentFactEvidence; the
    factory does not guess whether a learned execution factor is a hard record.
    """
    neutral = neutral_attribution(attribution)
    belief = deepcopy(prior)
    for name, fact in list(belief.facts.items()):
        if fact.source == "candidate_prediction" or "initial_observation" in fact.evidence_ids:
            belief.facts[name] = BeliefFact(TriValue.UNKNOWN, 0., "observation", block_index,
                                          ("unverified_assumption_masked",))
    changes = []
    q = attribution.evidence_quality
    if len({o.predicate for o in observations}) != len(observations):
        raise ValueError("Conflicting current predicate sources require explicit adjudication")
    for item in observations:
        if not isinstance(item, CurrentFactEvidence):
            raise ValueError("Actual CurrentFactEvidence required")
        if item.source == "measured_execution":
            if item.predicate != "execution_consistent":
                raise ValueError("Command equality cannot certify grasp, lift or placement")
            if not q.execution_reliable:
                continue
        else:
            usable = {"primary":q.primary_reliable, "wrist":q.wrist_reliable,
                      "fused":q.primary_reliable and q.wrist_reliable, "record":False}[item.view]
            if q.cross_view_conflict or not usable:
                continue
        old = belief.facts[item.predicate]
        belief.facts[item.predicate] = BeliefFact(item.value,item.confidence,"observation",block_index,(item.evidence_id,))
        changes.append(BeliefChange(item.predicate,old.value,old.confidence,item.value,item.confidence,
            item.source,True,(item.predicate,),item.evidence_id))
    belief.history.extend(changes)
    return RecoveryContext(neutral,belief,graph,tuple(changes),block_index)


def _validate(pool):
    if not isinstance(pool, ControlPool) or set(pool.identity) != set(IDENTITY):
        raise ValueError("Complete exact pool source identity required")
    if any(type(pool.identity[name]) is not int or pool.identity[name]<0 for name in ("task","state")):
        raise ValueError("Identity task/state require literal nonnegative integers, not bool or aliases")
    if any(type(pool.identity[name]) is not str or not pool.identity[name].strip()
           for name in ("dataset","suite","observed_block_id")):
        raise ValueError("Identity namespace/block require nonempty strings")
    if pool.split not in ("train", "val") or not pool.pool_id or not pool.attribution.source_block_id:
        raise ValueError("Explicit train/val, pool and real before-source block required")
    if not isinstance(pool.prior,BeliefState) or not isinstance(pool.attribution,AttributionOutput):
        raise ValueError("Typed prior and before attribution required")
    if set(pool.graph.nodes) != set(PREDICATES):
        raise ValueError("Same fixed predicate nodes required")
    if type(pool.block_index) is not int or pool.block_index < 0:
        raise ValueError("Valid actual-before block index required")
    k=len(pool.effects)
    if not k or any(len(v)!=k for v in (pool.plans,pool.official_values,pool.backbone_scores)):
        raise ValueError("Aligned complete candidate pool required")
    if len({e.candidate_id for e in pool.effects}) != k:
        raise ValueError("Duplicate candidate IDs")
    for effect,plan in zip(pool.effects,pool.plans,strict=True):
        _no_outcome_keys(effect.evidence)
        _no_outcome_keys(effect.hard_violations)
        a=np.asarray(plan,dtype=float)
        if a.shape!=(16,7) or not np.isfinite(a).all():
            raise ValueError("Complete finite native 16x7 plan required")
    if not np.isfinite(pool.official_values).all():
        raise ValueError("Finite official values required; native range is preserved")
    if not np.isfinite(pool.backbone_scores).all():
        raise ValueError("Finite precomputed frozen backbone scores required")
    if not isinstance(pool.budget,Mapping) or not pool.budget or not pool.selection_wrapper_id:
        raise ValueError("Explicit common budget and selection wrapper required")
    _no_outcome_keys(pool.budget)
    if pool.fallback not in ("safe_reject","requery","reobserve"):
        raise ValueError("Common explicit fallback required")
    if any(not isinstance(o,CurrentFactEvidence) for o in pool.observations):
        raise ValueError("Only actual-before CurrentFactEvidence accepted")


def _pool_key(pool):
    return (pool.split,str(pool.identity["dataset"]),str(pool.identity["suite"]),
        str(pool.identity["task"]),str(pool.identity["state"]),
        str(pool.identity["observed_block_id"]),pool.pool_id)


def _shuffle_map(pools, seed):
    if type(seed) is not int or seed!=0:
        raise ValueError("Predeclared shuffle seed0 required")
    grouped=defaultdict(list)
    for p in pools:grouped[p.split].append(p)
    output={}
    rng=random.Random(seed)
    for split in ("train","val"):
        members=sorted(grouped[split],key=_pool_key)
        def group(p):return (str(p.identity["suite"]),str(p.identity["task"]),str(p.identity["state"]))
        edges=[]
        for receiver in members:
            possible=[i for i,p in enumerate(members) if group(p)!=group(receiver)]
            rng.shuffle(possible);edges.append(possible)
        # Outcome-free deterministic bipartite derangement. Every donor is
        # used exactly once, or the ENTIRE split shuffle is unavailable.
        owners={}
        def match(receiver,seen):
            for donor in edges[receiver]:
                if donor in seen:continue
                seen.add(donor)
                if donor not in owners or match(owners[donor],seen):
                    owners[donor]=receiver
                    return True
            return False
        order=list(range(len(members)));rng.shuffle(order)
        passed=all(match(i,set()) for i in order)
        inverse={receiver:donor for donor,receiver in owners.items()} if passed else {}
        for i,receiver in enumerate(members):
            output[_pool_key(receiver)]=members[inverse[i]] if passed else None
    return output


def _substitute(receiver,donor):
    original=donor.attribution
    # The recipient's ACTUAL quality cannot be replaced by donor quality.
    cause=CoarseCause.UNKNOWN if receiver.attribution.evidence_quality.cross_view_conflict else original.projected_cause
    # A quality-forced final route cannot retain confidence of a different
    # donor class. Preserve the corresponding existing direct probability;
    # this is label/confidence correspondence, not calibration or modification
    # of the donor's probability values. A different class slot can naturally
    # be numerically higher or lower; that change is explicitly reported.
    confidence=float(original.class_probs[cause.value])
    if not np.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError("Shuffled projected class requires its original direct confidence")
    return replace(original,evidence_quality=receiver.attribution.evidence_quality,
        source_block_id=receiver.attribution.source_block_id,projected_cause=cause,confidence=confidence)


def _changed_facts(prior,after):
    before=prior.snapshot();now=after.snapshot()
    return dict(task_stage_before=before["task_stage"],task_stage_after=now["task_stage"],
        facts=[dict(predicate=name,before=before["facts"][name],after=now["facts"][name])
               for name in PREDICATES if before["facts"][name]!=now["facts"][name]])


def _gate(context,effect,plan,mode):
    trace=trace_candidate_recovery(context,effect,plan)
    if mode in ("masked","effect_only"):
        # UNKNOWN placeholder must not leak unknown=1/uniform route features.
        vector=trace.features.copy()
        for i,name in enumerate(trace.feature_names):
            if any(name.startswith(route+".") for route in ROUTES):
                vector[i]=0.
        trace=replace(trace,features=vector)
    critical={Stage.GRASP:{"target_pose_current","target_reachable"},
              Stage.LIFT:{"grasped"},Stage.TRANSPORT:{"grasped","lifted"},
              Stage.PLACE:{"lifted"}}.get(effect.stage,set())
    reasons=list(trace.hard_reasons)
    reasons.extend(name+"=unknown_requires_current_evidence" for name in trace.unresolved_requirements
                   if name in critical and trace.facts[name]["value"]=="unknown")
    return trace,dict(candidate_id=effect.candidate_id,accepted=not reasons,
        entry_route="verified" if not reasons else "safe_fallback",rejection_reasons=reasons,
        no_future_prediction_updates_before_or_unlocks_gate=True)


def build_complete_attribution_controls(pools:Sequence[ControlPool],*,seed=0):
    """Build all five complete controls without reading or producing Y.

    Shared requirements and desired objectives come from plans/original full
    BEFORE context, never from control outcomes. Controls may change derived
    needs/beliefs/gates; all such differences are retained, not called same gates.
    score_pool_mode is full for rerun controls: do NOT apply a second old
    need-only no-DAG or masked ablation and call that a full attribution removal.
    """
    pools=tuple(pools)
    if not pools:
        raise ValueError("Nonempty source-scoped before pools required")
    for pool in pools:_validate(pool)
    if len({_pool_key(p) for p in pools})!=len(pools):
        raise ValueError("Duplicate complete source pool identity")
    donors=_shuffle_map(pools,seed)
    output=[]
    for pool in pools:
        effects=tuple(typed_gate_effect(e,pool.relation) for e in pool.effects)
        full=prepare_recovery_context(prior=pool.prior,attribution=pool.attribution,
            observations=pool.observations,block_index=pool.block_index,graph=pool.graph)
        common=build_whole_recovery_tables(full,effects,pool.plans,
            frozen_boundary_goals=pool.frozen_boundary_goals,information_acquisition=pool.information_acquisition,
            before_observer_available=pool.before_observer_available)
        raw=dict(identity=pool.identity,pool_id=pool.pool_id,split=pool.split,
            actual_before=dict(prior=pool.prior.snapshot(),evidence_quality=asdict(pool.attribution.evidence_quality),
                               observations=[asdict(o) for o in pool.observations]),
            candidates=[dict(effect=asdict(e),plan=np.asarray(p),official_value=float(v),backbone_score=float(s))
                for e,p,v,s in zip(pool.effects,pool.plans,pool.official_values,pool.backbone_scores,strict=True)],
            budget=pool.budget,residual_cap=RESIDUAL_CAP,wrapper=pool.selection_wrapper_id,fallback=pool.fallback,
            before_scope=dict(block_index=pool.block_index,relation=pool.relation,
                frozen_boundary_goals=pool.frozen_boundary_goals,information_acquisition=pool.information_acquisition,
                before_observer_available=pool.before_observer_available,graph_nodes=sorted(pool.graph.nodes),
                graph_edges=pool.graph.edges,original_prediction_sha256=_digest(asdict(pool.attribution))))
        common_sha=_digest(raw)
        variants={}
        for mode in MODES:
            donor=donors[_pool_key(pool)] if mode=="shuffled" else None
            mask_adapter=mode in ("masked","effect_only") or (mode=="shuffled" and donor is None)
            graph=DependencyGraph(()) if mode=="no_dag" else pool.graph
            if mask_adapter:
                ctx=prepare_masked_context(prior=pool.prior,attribution=pool.attribution,
                    observations=pool.observations,block_index=pool.block_index,graph=graph)
            elif mode=="full":
                ctx=full
            else:
                attr=_substitute(pool,donor) if mode=="shuffled" else pool.attribution
                ctx=prepare_recovery_context(prior=pool.prior,attribution=attr,
                    observations=pool.observations,block_index=pool.block_index,graph=graph)
            # The old builder rejects physical goal requests under UNKNOWN.
            # No such goal is fabricated: its shared comparison objective stays
            # fixed, but a controlled context only derives legal own needs.
            requested=tuple(pool.frozen_boundary_goals)
            if ctx.attribution.projected_cause in (CoarseCause.UNKNOWN,CoarseCause.VISUAL_OCCLUSION):
                requested=tuple(p for p in requested if p in ("target_visible","target_pose_current","receptacle_visible"))
            own=build_whole_recovery_tables(ctx,effects,pool.plans,frozen_boundary_goals=requested,
                information_acquisition=pool.information_acquisition,before_observer_available=pool.before_observer_available)
            rows,gates,traces=[],[],[]
            for effect,plan,base,actual in zip(effects,pool.plans,common,own,strict=True):
                trace,gate=_gate(ctx,effect,plan,"masked" if mask_adapter else mode)
                allowed={(o["predicate"],o["deadline"],o["kind"]) for o in base["requirements"]+base["goals"]}
                needs=[deepcopy(n) for n in actual["needs"] if (n["predicate"],n["deadline"],n["kind"]) in allowed]
                if mask_adapter:
                    for need in needs:need["weight"]=0.
                index=next(i for i,e in enumerate(pool.effects) if e.candidate_id==effect.candidate_id)
                rows.append(dict(identity=dict(pool.identity,candidate_id=effect.candidate_id),pool_id=pool.pool_id,
                    split=pool.split,candidate_id=effect.candidate_id,stage=base["stage"],purpose=base["purpose"],
                    requirements=deepcopy(base["requirements"]),goals=deepcopy(base["goals"]),needs=needs,
                    own_derived_goals=deepcopy(actual["goals"]),backbone_score=float(pool.backbone_scores[index]),
                    entry_route=gate["entry_route"],common_before_and_candidate_x_sha256=common_sha))
                gates.append(gate);traces.append(trace)
            metadata=dict(mode=mode,experimental_intervention=mode in ("masked","shuffled","effect_only"),
                real_attribution_prediction_claimed=mode in ("full","no_dag"),
                masked_adapter_used=mask_adapter,placeholder_is_not_normal_or_real_unknown_prediction=mask_adapter,
                learned_channel_updates_disabled=mask_adapter,route_conditioned_features_disabled=mask_adapter,
                actual_evidence_quality_and_record_inputs_retained=True,
                masked_channels="learned_classification_and_factor_channels_only" if mask_adapter else None,
                D18_perception_quality_removed=False,
                extra_quality_prior_invalidation_added=False,
                explicit_record_hard_evidence_count=sum(o.source=="measured_execution" for o in pool.observations),
                masked_adapter_reason="old UNKNOWN/conf0 still changes belief; explicit update mask required" if mask_adapter else None,
                shuffled_seed=seed if mode=="shuffled" else None,
                donor_identity=None if donor is None else dict(donor.identity,pool_id=donor.pool_id,split=donor.split),
                donor_prediction_sha256=None if donor is None else _digest(asdict(donor.attribution)),
                recipient_source_block_preserved=True,
                shuffled_missing_donor="no_complete_split_local_cross_task_state_bijection_masked" if mode=="shuffled" and donor is None else None,
                shuffled_distribution_preserving_pool_bijection=mode=="shuffled" and donor is not None,
                recipient_quality_forced_unknown_projection=bool(donor is not None and pool.attribution.evidence_quality.cross_view_conflict),
                shuffled_selected_confidence_from_original_direct_projected_class=donor is not None,
                shuffled_confidence_calibrated=False,shuffled_probability_values_modified=False,
                shuffled_original_selected_confidence=None if donor is None else donor.attribution.confidence,
                shuffled_controlled_selected_confidence=None if donor is None else ctx.attribution.confidence,
                shuffled_selected_confidence_increased_by_class_projection=bool(donor is not None and ctx.attribution.confidence>donor.attribution.confidence),
                no_dag_same_nodes=set(graph.nodes)==set(pool.graph.nodes),graph_edges=list(graph.edges),
                intervention_scope="current_verified_block_update_from_common_prior",
                not_a_history_reconstructed_closed_loop_ablation=True,
                prior_attribution_facts_retained=[name for name,fact in pool.prior.facts.items() if fact.source=="attribution"],
                prior_propagation_history_present=any(not c.direct for c in pool.prior.history),
                score_pool_mode="effect_only" if mode=="effect_only" else "full",
                prediction_must_not_write_before_belief=True,no_Y_read=True,
                actual_source_content_binding_and_supervision_gate_still_required=True)
            variants[mode]=dict(context=ctx,rows=tuple(rows),traces=tuple(traces),gates=tuple(gates),
                before_changes=_changed_facts(pool.prior,ctx.belief),metadata=metadata,
                common_before_and_candidate_x_sha256=common_sha,budget=deepcopy(pool.budget),
                residual_cap=RESIDUAL_CAP,selection_wrapper_id=pool.selection_wrapper_id,
                fallback=pool.fallback,official_values=tuple(float(v) for v in pool.official_values),
                backbone_scores=tuple(float(v) for v in pool.backbone_scores))
        reference=variants["full"]["gates"]
        for variant in variants.values():
            variant["gate_differences_vs_full"]=[dict(candidate_id=a["candidate_id"],full=deepcopy(a),control=deepcopy(b))
                for a,b in zip(reference,variant["gates"],strict=True) if a!=b]
            variant["gate_outcomes_equal_full"]=not variant["gate_differences_vs_full"]
            variant["same_gate_formula_not_same_gate_outcomes_claim"]=True
        output.append(dict(identity=dict(pool.identity),pool_id=pool.pool_id,split=pool.split,controls=variants))
    return dict(schema=SCHEMA,pools=tuple(output),seed=seed,outcomes_read=False,models_loaded=False,
        fitting_started=False,training_ready=False,actual_dataset_fairness_pass=False,
        shuffled_complete_prediction_bijection_available=all(donor is not None for donor in donors.values()),
        interpretation="Complete current-block before attribution reruns from a common prior; not reconstructed history; distinct from old need-weight-only score controls.")


def control_factory_record(result):
    """JSON-ready diagnostic export; NOT a training bundle or source auditor."""
    output={key:deepcopy(value) for key,value in result.items() if key!="pools"}
    output["pools"]=[]
    for pool in result["pools"]:
        item={key:deepcopy(value) for key,value in pool.items() if key!="controls"}
        item["controls"]={}
        for mode,variant in pool["controls"].items():
            record={key:deepcopy(value) for key,value in variant.items() if key not in ("context","traces")}
            ctx=variant["context"]
            record["before_context"]=dict(attribution=asdict(ctx.attribution),belief=ctx.belief.snapshot(),
                changes=[asdict(c) for c in ctx.changes],block_index=ctx.block_index,
                graph_nodes=sorted(ctx.graph.nodes),graph_edges=ctx.graph.edges)
            record["traces"]=[asdict(t) for t in variant["traces"]]
            item["controls"][mode]=record
        output["pools"].append(item)
    return _plain(output)
