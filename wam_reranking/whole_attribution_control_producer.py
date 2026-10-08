"""Byte-bound current-block upstream controls; no inference, fitting or Y.

This is distinct from the old need-weight controls. Actual recorded inputs are
read, then attribution -> belief -> dependency graph -> needs/gates is rerun
from the same prior. It does not reconstruct historical closed-loop arms.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from . import attribution_control_factory as factory
from . import candidate_effects, joint_recovery_context, native_before_adapter
from . import observed_recovery, recovery_contract
from . import belief, policy, reranker, candidate_utility, direct_recovery_context
from . import joint_recovery_contract, recovery_input_sources
from .belief import DEFAULT_GRAPH
from .contracts import (BeliefChange, BeliefFact, BeliefState, CandidateVisualEvidence,
                        PREDICATES, Stage, TriValue)
from .evidence_residual import typed_gate_effect, candidate_backbone_features
from .candidate_utility import CandidateUtilityModel
from .joint_recovery_contract import (ArtifactReader, OBSERVER, RESIDUAL_CAP,
    PREDICATES as RECOVERY_PREDICATES, deployment_payload, digest, file_sha, validate_x)
from .recovery_contract import CurrentFactEvidence, VISUAL_EVIDENCE_KEYS

SOURCE_SCHEMA = "whole_current_block_control_source_spec_v1"
OUTPUT_SCHEMA = "byte_bound_current_block_attribution_controls_v1"
SCOPE = "current_block_shared_prior"
MODES = factory.MODES
POOL_KEYS = ("dataset", "suite", "task", "state", "observed_block_id")
SPEC_KEYS = ("schema", "identity", "pool_id", "split", "block_index",
    "attribution_source", "attribution_source_block_id", "prior_source",
    "current_observations_source", "candidate_effect_sources", "frozen_goal_source",
    "runtime_input_source", "ranking_source", "membership_source")


def _keys(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(name + " exact fields required")


def _plain_ref(ref):
    return {k: ref[k] for k in ("path", "sha256")}


def _same_ref(a, b):
    return _plain_ref(a) == _plain_ref(b)


def _ref_table(refs):
    # Paths matter, and repeated equal-content files remain distinct artifacts.
    return sorted(digest(_plain_ref(ref)) for ref in refs)


def _pool_key(value):
    ident = value["identity"]
    _keys(ident, POOL_KEYS, "five-field source pool identity")
    if any(type(ident[k]) is not int or ident[k] < 0 for k in ("task", "state")):
        raise ValueError("literal source task/state required")
    if any(type(ident[k]) is not str or not ident[k] for k in ("dataset", "suite", "observed_block_id")):
        raise ValueError("literal dataset/suite/block required")
    if value["split"] not in ("train", "val") or type(value["pool_id"]) is not str or not value["pool_id"]:
        raise ValueError("source pool and development split required")
    return tuple(ident[k] for k in POOL_KEYS) + (value["pool_id"], value["split"])


def _row_pool_key(row):
    return _pool_key(dict(identity={k: row["identity"][k] for k in POOL_KEYS},
                          pool_id=row["pool_id"], split=row["split"]))


def _scope(record, spec, schema):
    if (record.get("schema") != schema or record.get("identity") != spec["identity"]
        or record.get("pool_id") != spec["pool_id"] or record.get("split") != spec["split"]):
        raise ValueError("recorded source identity/split/pool mismatch: " + schema)


def _array(ref, reader, shape):
    path = reader.path(ref)
    if path.suffix.lower() != ".npy":
        raise ValueError("real immutable NPY source required")
    value = np.load(path, allow_pickle=False)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError("recorded source shape/finite values mismatch")
    return value


def _code_sources():
    modules = dict(producer=__file__, factory=factory.__file__,
        context=joint_recovery_context.__file__, parser=candidate_effects.__file__,
        raw_contract=recovery_contract.__file__, attribution_adapter=native_before_adapter.__file__,
        observer=observed_recovery.__file__, whole_contract=joint_recovery_contract.__file__,
        input_content_reader=recovery_input_sources.__file__, belief=belief.__file__,
        dependency_policy=policy.__file__, prerequisite_gate=reranker.__file__,
        backbone_features=typed_gate_effect.__code__.co_filename,
        backbone_scoring=candidate_utility.__file__, direct_need_provenance=direct_recovery_context.__file__)
    return {k: dict(path=str(Path(p).resolve()), sha256=file_sha(p)) for k, p in modules.items()}


def _witness(ref, reader, spec, *, predicate, value, confidence, evidence_id, before_block):
    """Accept an independently recorded deployable-before witness, never GT.

    This verifies the recorded witness and its source contents/lineage. The
    independently frozen predicate observer contract remains a separate gate;
    this module cannot turn a role or code hash into physical observability.
    """
    w = reader.json(ref)
    _scope(w, spec, "deployable_before_predicate_witness_v1")
    required = ("schema", "identity", "pool_id", "split", "predicate", "value",
        "confidence", "evidence_id", "block_index", "relative_step", "role",
        "input_sources", "observer_code_source", "observer_model_source",
        "predicate_contract_source", "private_teacher_used")
    _keys(w, required, "actual-before predicate witness")
    if (w["predicate"] != predicate or w["value"] != value or w["confidence"] != confidence
        or w["evidence_id"] != evidence_id or type(w["block_index"]) is not int
        or w["block_index"] != before_block or before_block > spec["block_index"]
        or type(w["relative_step"]) is not int or w["relative_step"] > 0
        or w["role"] != "deployable_before_observation" or w["private_teacher_used"] is not False):
        raise ValueError("prior/current witness is not actual before evidence")
    if not isinstance(w["input_sources"], list) or not w["input_sources"]:
        raise ValueError("actual recorded predicate witness inputs required")
    for source in w["input_sources"]:
        reader.path(source)
    for name in ("observer_code_source", "observer_model_source", "predicate_contract_source"):
        reader.path(w[name])
    semantic = reader.json(w["predicate_contract_source"])
    if (semantic.get("schema") != "observed_recovery_predicate_contract_v1"
        or semantic.get("predicate") != predicate or semantic.get("missing_is_masked_not_negative") is not True
        or semantic.get("entry_verification_safe") is not True):
        raise ValueError("independent deployable predicate definition missing")
    return w


def _prior_and_observations(spec, rows, reader):
    p = reader.json(spec["prior_source"])
    _scope(p, spec, "recorded_shared_before_prior_v1")
    _keys(p, ("schema", "identity", "pool_id", "split", "policy", "prior", "fact_witnesses"), "recorded prior")
    state = p["prior"]
    _keys(state, ("task_id", "target_object", "receptacle", "task_stage", "facts", "history"), "prior state")
    binding = rows[0]["x"]["task_context"]
    if state["task_id"] != spec["identity"]["task"] or state["target_object"] != binding["target"] or state["receptacle"] != binding["anchor"]:
        raise ValueError("prior task/entity binding mismatch")
    _keys(state["facts"], PREDICATES, "all prior fact slots")
    facts = {}
    for name, f in state["facts"].items():
        _keys(f, ("value", "confidence", "source", "updated_at_block", "evidence_ids"), "prior fact")
        facts[name] = BeliefFact(TriValue(f["value"]), f["confidence"], f["source"], f["updated_at_block"], tuple(f["evidence_ids"]))
        if type(f["updated_at_block"]) is not int or f["updated_at_block"] >= spec["block_index"]:
            raise ValueError("prior must precede this current-block update")
    history = []
    for c in state["history"]:
        history.append(BeliefChange(c["predicate"], TriValue(c["old_value"]), c["old_confidence"],
            TriValue(c["new_value"]), c["new_confidence"], c["reason"], c["direct"], tuple(c["propagation_path"]), c["evidence_id"]))
    if p["policy"] == "explicit_uninformative_no_verified_history":
        if history or p["fact_witnesses"] or any(f.value is not TriValue.UNKNOWN or f.confidence != 0. or f.evidence_ids for f in facts.values()):
            raise ValueError("uninformative prior cannot fabricate TRUE or historical evidence")
    elif p["policy"] == "verified_deployable_history":
        if not isinstance(p["fact_witnesses"], Mapping):
            raise ValueError("prior witness table required")
        for name, f in facts.items():
            # Candidate forecasts are never sufficient prior observations.
            if f.source != "observation" or f.value is TriValue.UNKNOWN:
                if f.value is not TriValue.UNKNOWN:
                    raise ValueError("first producer admits no unaudited historical attribution/forecast prior")
                continue
            if len(f.evidence_ids) != 1 or name not in p["fact_witnesses"]:
                raise ValueError("nonunknown prior needs an independent before witness")
            _witness(p["fact_witnesses"][name], reader, spec, predicate=name, value=f.value.value,
                confidence=f.confidence, evidence_id=f.evidence_ids[0], before_block=f.updated_at_block)
    else:
        raise ValueError("explicit recorded prior policy required")
    prior = BeliefState(state["task_id"], state["target_object"], state["receptacle"], Stage(state["task_stage"]), facts, history)
    record = reader.json(spec["current_observations_source"])
    _scope(record, spec, "recorded_current_before_observations_v1")
    _keys(record, ("schema", "identity", "pool_id", "split", "block_index", "observations"), "before observations")
    if record["block_index"] != spec["block_index"]:
        raise ValueError("current observation block mismatch")
    observations = []
    for item in record["observations"]:
        _keys(item, ("predicate", "value", "confidence", "evidence_id", "source", "view", "witness_source"), "current observation")
        _witness(item["witness_source"], reader, spec, predicate=item["predicate"], value=item["value"],
            confidence=item["confidence"], evidence_id=item["evidence_id"], before_block=spec["block_index"])
        observations.append(CurrentFactEvidence(item["predicate"], TriValue(item["value"]), item["confidence"],
            item["evidence_id"], source=item["source"], view=item["view"]))
    return prior, tuple(observations)


def _membership(spec, rows, reader):
    record = reader.json(spec["membership_source"])
    if record.get("schema") != "frozen_before_pool_membership_v1" or not isinstance(record.get("pools"), list):
        raise ValueError("actual frozen source-group membership manifest required")
    matches = [p for p in record["pools"] if _pool_key(p) == _pool_key(spec)]
    if len(matches) != 1 or matches[0].get("candidate_ids") != [r["identity"]["candidate_id"] for r in rows]:
        raise ValueError("complete candidate membership or frozen split mismatch")
    groups = {}
    for pool in record["pools"]:
        _pool_key(pool)
        i = pool["identity"]; group = (i["suite"], i["task"], i["state"])
        if group in groups and groups[group] != pool["split"]:
            raise ValueError("membership physical task/state train/val leakage")
        groups[group] = pool["split"]


def _runtime(spec, rows, reader, attribution):
    """Read actual captured order receipts, not self-written passed flags."""
    record = reader.json(spec["runtime_input_source"])
    _scope(record, spec, "recorded_preexecution_control_lineage_v1")
    _keys(record, ("schema", "identity", "pool_id", "split", "block_index", "attribution_source_block_id",
        "attribution_source", "shared_before_sources", "before_lineage", "before_quality_evaluation",
        "code_sources", "event_log_source", "candidate_sources"), "runtime before/query lineage")
    if (record["block_index"] != spec["block_index"] or record["attribution_source_block_id"] != spec["attribution_source_block_id"]
        or not _same_ref(record["attribution_source"], spec["attribution_source"])):
        raise ValueError("actual attribution block/provenance mismatch")
    before = rows[0]["x"]["actual_before"]
    expected = dict(primary=_plain_ref(before["primary_rgb"]), wrist=_plain_ref(before["wrist_rgb"]), proprio=_plain_ref(before["proprio_source"]))
    if record["shared_before_sources"] != expected:
        raise ValueError("runtime actual before RGB/proprio mismatch")
    expected_lineage = {k: _plain_ref(v) for k, v in rows[0]["before_lineage"].items()}
    if record["before_lineage"] != expected_lineage:
        raise ValueError("runtime shared snapshot/query parent mismatch")
    for ref in record["shared_before_sources"].values(): reader.path(ref)
    for ref in record["before_lineage"].values(): reader.path(ref)
    quality = record["before_quality_evaluation"]
    _keys(quality, ("identity", "relative_step", "actual_before_sources", "quality", "subject_maps_source",
        "observer_code_source", "observer_model_source"), "actual before quality evaluation")
    if quality["identity"] != spec["identity"] or quality["relative_step"] != 0 or quality["actual_before_sources"] != expected or quality["quality"] != before["quality"]:
        raise ValueError("actual before observer source/time/content mismatch")
    for key in ("observer_code_source", "observer_model_source"): reader.path(quality[key])
    if quality["observer_code_source"]["sha256"] != file_sha(observed_recovery.__file__):
        raise ValueError("before observer map endpoint definition differs")
    maps_path = reader.path(quality["subject_maps_source"])
    maps = np.load(maps_path, allow_pickle=False)
    if maps.ndim != 3 or maps.shape[0] != 2 or not np.isfinite(maps).all():
        raise ValueError("actual before maps must be finite own dual-view maps")
    q = before["quality"]
    if {k: q[k] for k in ("primary_reliable", "wrist_reliable", "execution_reliable", "cross_view_conflict")} != asdict(attribution.evidence_quality):
        raise ValueError("before quality and frozen attribution quality disagree")
    available = not q["cross_view_conflict"] and any(q[v+"_reliable"] and observed_recovery.map_endpoint(maps[i])["usable"] for i, v in enumerate(("primary", "wrist")))
    if q["observer_evidence_available"] is not available:
        raise ValueError("before observer availability differs from actual frozen map evaluation")
    _keys(record["code_sources"], ("input_producer", "attribution_code", "attribution_model", "candidate_model", "effect_parser"), "runtime producing code/model")
    for ref in record["code_sources"].values(): reader.path(ref)
    if record["code_sources"]["effect_parser"]["sha256"] != file_sha(candidate_effects.__file__):
        raise ValueError("own predicted effect parser differs from frozen producer")
    own = {item["candidate_id"]: item for item in record["candidate_sources"]}
    if len(record["candidate_sources"]) != len(rows) or len(own) != len(rows) or set(own) != {r["identity"]["candidate_id"] for r in rows}:
        raise ValueError("runtime incomplete candidate sources")
    for row in rows:
        cid = row["identity"]["candidate_id"]; item = own[cid]; x = row["x"]
        _keys(item, ("candidate_id", "planned_actions_source", "predicted_primary_source", "predicted_wrist_source", "effect_source"), "runtime own candidate")
        expected_own = (x["planned_actions_source"], x["candidate_predicted"]["primary_endpoint_rgb"], x["candidate_predicted"]["wrist_endpoint_rgb"])
        if any(not _same_ref(item[k], ref) for k, ref in zip(("planned_actions_source", "predicted_primary_source", "predicted_wrist_source"), expected_own)):
            raise ValueError("runtime own plan/prediction substituted")
        for k in item:
            if k != "candidate_id": reader.path(item[k])
    log = reader.json(record["event_log_source"])
    _scope(log, spec, "recorded_preexecution_generation_order_v1")
    _keys(log, ("schema", "identity", "pool_id", "split", "producer_code_source", "events"), "recorded event log")
    if not _same_ref(log["producer_code_source"], record["code_sources"]["input_producer"]):
        raise ValueError("actual runtime event producer differs")
    events = log["events"]
    if not isinstance(events, list) or not events:
        raise ValueError("actual ordered runtime events unavailable")
    seen, generated, execution_at = {}, {}, []
    for e in events:
        _keys(e, ("sequence", "kind", "candidate_id", "artifact_sources"), "recorded runtime event")
        if type(e["sequence"]) is not int or e["sequence"] < 0 or e["sequence"] in seen:
            raise ValueError("literal unique event order required")
        seen[e["sequence"]] = e
        if not isinstance(e["artifact_sources"], list): raise ValueError("event artifact table required")
        for ref in e["artifact_sources"]: reader.path(ref)
        if e["kind"] == "candidate_generated":
            cid = e["candidate_id"]
            if cid not in own or cid in generated:
                raise ValueError("candidate generation event duplicated or wrong membership")
            if _ref_table(e["artifact_sources"]) != _ref_table(own[cid][k] for k in own[cid] if k != "candidate_id"):
                raise ValueError("generation event does not bind exact own X outputs")
            generated[cid] = e["sequence"]
        elif e["kind"] == "candidate_execution_started":
            if e["candidate_id"] not in own or e["artifact_sources"]:
                raise ValueError("execution metadata cannot import outcomes into controls")
            execution_at.append(e["sequence"])
        elif e["kind"] not in ("actual_before_captured", "before_attribution_produced", "before_quality_produced", "boundary_goal_declared") or e["candidate_id"] is not None:
            raise ValueError("unknown runtime event or future outcome field")
    if list(seen) != sorted(seen) or set(generated) != set(own):
        raise ValueError("actual before query chronology or membership incomplete")
    expected_events = dict(actual_before_captured=list(expected.values()), before_attribution_produced=[spec["attribution_source"]],
        before_quality_produced=[quality["subject_maps_source"], before["quality_source"]], boundary_goal_declared=[spec["frozen_goal_source"]])
    for kind, refs in expected_events.items():
        items = [e for e in events if e["kind"] == kind]
        if len(items) != 1 or _ref_table(items[0]["artifact_sources"]) != _ref_table(refs) or items[0]["sequence"] >= min(generated.values()):
            raise ValueError("real before/goal artifact event missing or after candidate query")
    times = {k: next(e["sequence"] for e in events if e["kind"] == k) for k in expected_events}
    if not (times["actual_before_captured"] < times["before_attribution_produced"]
        and times["actual_before_captured"] < times["before_quality_produced"]
        and max(times["before_attribution_produced"], times["before_quality_produced"]) < times["boundary_goal_declared"]):
        raise ValueError("before attribution/quality/goal producing order inconsistent")
    if execution_at and max(generated.values()) >= min(execution_at):
        raise ValueError("all candidate X must be saved before first execution")
    return record, own


def _pool(spec, rows, reader):
    _keys(spec, SPEC_KEYS, "complete pipeline source spec")
    if spec["schema"] != SOURCE_SCHEMA or type(spec["block_index"]) is not int or spec["block_index"] < 0:
        raise ValueError("new actual current-block source schema required")
    _pool_key(spec)
    if not rows or any(_row_pool_key(r) != _pool_key(spec) for r in rows):
        raise ValueError("complete source rows mismatch")
    if len({r["identity"]["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate actual candidate identity")
    common = digest(dict(actual_before=rows[0]["x"]["actual_before"], before_lineage=rows[0]["before_lineage"], task_context=rows[0]["x"]["task_context"]))
    for row in rows:
        validate_x(row["x"], reader)
        _keys(row["before_lineage"], ("snapshot_source", "query_source"), "shared non-input before lineage")
        for name, ref in row["before_lineage"].items():
            _keys(ref, ("path", "sha256", "metadata_only", "private_teacher"), "non-input before lineage ref")
            if ref["metadata_only"] is not True or ref["private_teacher"] is not (name == "snapshot_source"):
                raise ValueError("snapshot/query are strictly typed non-input provenance")
            reader.path(ref)
        if digest(dict(actual_before=row["x"]["actual_before"], before_lineage=row["before_lineage"], task_context=row["x"]["task_context"])) != common:
            raise ValueError("source candidates do not share actual before/context/query")
    _membership(spec, rows, reader)
    attr, attr_audit = native_before_adapter.convert_frozen_attribution(reader.json(spec["attribution_source"]), spec["attribution_source_block_id"])
    prior, observations = _prior_and_observations(spec, rows, reader)
    runtime, own = _runtime(spec, rows, reader, attr)
    effect_refs = spec["candidate_effect_sources"]
    if not isinstance(effect_refs, list) or sorted(item["candidate_id"] for item in effect_refs) != [r["identity"]["candidate_id"] for r in rows]:
        raise ValueError("full own candidate effect membership required")
    effects, backbone_effects, plans, values = [], [], [], []
    for row, effect_ref in zip(rows, sorted(effect_refs, key=lambda i:i["candidate_id"]), strict=True):
        _keys(effect_ref, ("candidate_id", "source"), "candidate effect source")
        cid = row["identity"]["candidate_id"]
        if not _same_ref(effect_ref["source"], own[cid]["effect_source"]): raise ValueError("effect/runtime source mismatch")
        record = reader.json(effect_ref["source"])
        _keys(record, ("schema", "identity", "pool_id", "split", "stage", "candidate_visual", "value", "planned_action_sha256", "predicted_primary_sha256", "predicted_wrist_sha256", "query_source_sha256"), "own preexecution candidate effect")
        if record["schema"] != "recorded_own_preexecution_candidate_effect_v1" or record["identity"] != row["identity"] or record["pool_id"] != spec["pool_id"] or record["split"] != spec["split"]:
            raise ValueError("effect source identity/role mismatch")
        x = row["x"]; pred = x["candidate_predicted"]
        for name, ref in (("planned_action_sha256", x["planned_actions_source"]), ("predicted_primary_sha256", pred["primary_endpoint_rgb"]), ("predicted_wrist_sha256", pred["wrist_endpoint_rgb"]), ("query_source_sha256", row["before_lineage"]["query_source"])):
            if record[name] != ref["sha256"]: raise ValueError("effect source own plan/pred/query binding mismatch")
        plan = np.asarray(x["planned_actions"], dtype=float)
        visual_evidence = CandidateVisualEvidence(**record["candidate_visual"])
        effect = typed_gate_effect(candidate_effects.parse_candidate_effect(cid, plan, visual_evidence=visual_evidence, close_when_negative=False), x["task_context"]["relation"])
        # Do not reinterpret the immutable inherited candidate backbone. Its
        # original parser default is a separately named legacy contract; only
        # the new recovery requirements use LIBERO's positive-close semantics.
        backbone_effects.append(candidate_effects.parse_candidate_effect(cid, plan, visual_evidence=visual_evidence))
        if effect.stage.value != record["stage"] or effect.stage.value != row["stage"]:
            raise ValueError("actual primary candidate stage changed")
        visual = {key: dict(value=float(np.clip(effect.evidence[key], -1, 1)) if key in effect.evidence else None, available=key in effect.evidence) for key in VISUAL_EVIDENCE_KEYS}
        if pred["visual"] != visual: raise ValueError("ordinary predicted visual information differs from own parser")
        effects.append(effect); plans.append(plan); values.append(record["value"])
    goal = reader.json(spec["frozen_goal_source"])
    _scope(goal, spec, "prequery_frozen_recovery_goal_policy_v1")
    _keys(goal, ("schema", "identity", "pool_id", "split", "frozen_boundary_goals", "information_acquisition"), "frozen goal policy")
    ranking = reader.json(spec["ranking_source"])
    _scope(ranking, spec, "recorded_frozen_candidate_backbone_scores_v1")
    _keys(ranking, ("schema", "identity", "pool_id", "split", "role", "model_source", "producer_code_source", "feature_contract", "parser_contract", "x_payload_sha256", "scores", "reference_candidate_id", "reference_policy", "budget", "selection_wrapper_id", "fallback"), "actual frozen ranking reference")
    if ranking["role"] not in ("preexecution_frozen_learned_backbone", "deterministic_before_only_frozen_backbone_replay") or ranking["x_payload_sha256"] != digest([r["x"] for r in rows]):
        raise ValueError("official value/renamed zeros cannot replace actual frozen learned backbone")
    for name in ("model_source", "producer_code_source"): reader.path(ranking[name])
    scores = ranking["scores"]
    if len(scores) != len(rows) or [i["candidate_id"] for i in scores] != [r["identity"]["candidate_id"] for r in rows] or any(set(i) != {"candidate_id", "features", "score"} for i in scores):
        raise ValueError("actual frozen backbone scores incomplete")
    if ranking["feature_contract"] != "candidate_only_zero_tail27_v1":
        raise ValueError("attribution-routed backbone is not the common candidate-only control")
    if ranking["parser_contract"] != "frozen_legacy_default_close_negative_v1":
        raise ValueError("immutable backbone parser contract must remain separately preserved")
    model = CandidateUtilityModel.from_record(reader.json(ranking["model_source"]))
    for item, effect, value in zip(scores, backbone_effects, values, strict=True):
        features = candidate_backbone_features(effect, value)
        if (not np.array_equal(features, np.asarray(item["features"], dtype=float))
            or np.any(features[-6:] != 0.) or model.score(features) != item["score"]):
            raise ValueError("common candidate-only backbone features/score do not reproduce")
    if any(i["score"] != r["backbone_score"] for i, r in zip(scores, rows, strict=True)):
        raise ValueError("frozen backbone output/content mismatch")
    if ranking["reference_policy"] != "frozen_backbone_argmax_lowest_id_tie" or ranking["reference_candidate_id"] != max(zip([r["identity"]["candidate_id"] for r in rows], [i["score"] for i in scores]), key=lambda p:(p[1], -p[0]))[0]:
        raise ValueError("non-GT frozen reference candidate required")
    if ranking["budget"].get("K") != len(rows): raise ValueError("common budget K differs from real membership")
    pool = factory.ControlPool(spec["identity"], spec["pool_id"], spec["split"], prior, attr, observations, tuple(effects), tuple(plans), tuple(values), tuple(i["score"] for i in scores), ranking["budget"], spec["block_index"], relation=rows[0]["x"]["task_context"]["relation"], graph=DEFAULT_GRAPH, frozen_boundary_goals=tuple(goal["frozen_boundary_goals"]), information_acquisition=goal["information_acquisition"], before_observer_available=rows[0]["x"]["actual_before"]["quality"]["observer_evidence_available"], selection_wrapper_id=ranking["selection_wrapper_id"], fallback=ranking["fallback"])
    return pool, dict(attribution_semantics=attr_audit, raw_actual_x_sha256=digest([r["x"] for r in rows]), shared_before_context_and_lineage_sha256=common, runtime_lineage_sha256=digest(runtime), reference_candidate_id=ranking["reference_candidate_id"],
        backbone_source_role=ranking["role"],
        backbone_computed_offline_after_collection=ranking["role"] == "deterministic_before_only_frozen_backbone_replay",
        backbone_replay_reads_only_own_before_prediction_plan_and_value=True)


def produce_complete_pipeline_controls(pool_specs, candidates, reader: ArtifactReader):
    """Build and audit current-block controls from actual source artifacts.

    Returns fail-closed precise gaps. No Y, heavy model, query or action is
    read/run. The small frozen candidate-backbone JSON is read and its scores
    are deterministically replayed on ordinary own-candidate inputs.
    rows may carry Y for an outer training join, but deployment_payload drops it
    before this producer uses anything. The producer never reads those fields.
    """
    # An isolated registry makes the receipt deterministic and independent of
    # unrelated references cached by a calling whole-run auditor.
    outer_reader = reader
    reader = ArtifactReader(outer_reader.base_path)
    failures, controls = [], {m: [] for m in MODES}
    if not isinstance(pool_specs, list) or not pool_specs or not isinstance(candidates, list) or not candidates:
        raise ValueError("complete actual pool specs and candidate rows required")
    raw_rows = deployment_payload(candidates)
    grouped, source_groups, names = defaultdict(list), {}, {}
    for row in raw_rows:
        key = _row_pool_key(row); grouped[key].append(row)
        ident = row["identity"]; physical = (ident["suite"], ident["task"], ident["state"])
        if physical in source_groups and source_groups[physical] != row["split"]: failures.append("task_state_split_leakage")
        source_groups[physical] = row["split"]
        alias = (row["pool_id"], row["split"])
        if alias in names and names[alias] != key: failures.append("source_pool_name_alias")
        names[alias] = key
    pools, audits, keys = [], [], []
    for spec in pool_specs:
        try:
            key = _pool_key(spec)
            if key in keys: raise ValueError("duplicate source spec")
            keys.append(key)
            rows = sorted(grouped[key], key=lambda r:r["identity"]["candidate_id"])
            pool, audit = _pool(spec, rows, reader); pools.append(pool)
            audits.append(dict(identity=spec["identity"],pool_id=spec["pool_id"],split=spec["split"],**audit))
        except (ValueError,KeyError,TypeError,OSError) as error:
            failures.append("pool_source:" + str(error))
    if set(keys) != set(grouped): failures.append("source_spec_candidate_pool_membership_mismatch")
    factory_record = None; active = set(); active_by_control = {m:set() for m in MODES}
    if not failures:
        factory_record = factory.control_factory_record(factory.build_complete_attribution_controls(pools, seed=0))
        if not factory_record["shuffled_complete_prediction_bijection_available"]:
            failures.append("complete_split_local_cross_group_prediction_shuffle_unavailable")
        by_key = {_pool_key(p): p for p in factory_record["pools"]}
        for spec in pool_specs:
            key = _pool_key(spec); item = by_key[key]
            originals = {r["identity"]["candidate_id"]: r for r in grouped[key]}
            for mode, variant in item["controls"].items():
                belief = variant["before_context"]["belief"]["facts"]
                verifier = {p: dict(value=belief[p]["value"], deployable_observer_verified=belief[p]["source"]=="observation" and bool(belief[p]["evidence_ids"]) and belief[p]["confidence"]>0) for p in RECOVERY_PREDICATES}
                for generated in variant["rows"]:
                    row = deepcopy(originals[generated["candidate_id"]])
                    for field in ("requirements", "goals", "needs", "purpose", "entry_route"):
                        row[field] = deepcopy(generated[field])
                    row["before_verification"] = deepcopy(verifier)
                    controls[mode].append(row)
                    objectives = {(o["predicate"],o["deadline"],o["kind"]):o for o in row["requirements"]+row["goals"]}
                    for need in row["needs"]:
                        k = (need["predicate"], need["deadline"], need["kind"])
                        if need["cause"] != "normal" and need["deadline"] > 0 and need["weight"] * objectives[k]["confidence"] > 0:
                            active.add(k)
                            active_by_control[mode].add(k)
    # ArtifactReader caches references; independently rehash ALL sources after
    # generation so read-during-change cannot receive an admission certificate.
    code = _code_sources()
    for ref in code.values(): reader.path(ref)
    sources = [dict(path=p,sha256=h) for p,h in sorted(reader.verified)]
    for source in sources:
        if file_sha(source["path"]) != source["sha256"]: failures.append("source_changed_during_control_production:"+source["path"])
    outer_reader.verified.update(reader.verified)
    return dict(schema=OUTPUT_SCHEMA, scope=SCOPE, passed=not failures,
        complete_pipeline_controls_ready=not failures, training_ready=False, training_started=False,
        not_history_reconstructed_closed_loop=True, empirical_benefit_claimed=False,
        pool_specs=deepcopy(pool_specs), full_input_rows=raw_rows,
        controls=controls, control_payload_sha256={m:digest(deployment_payload(r)) for m,r in controls.items()},
        factory_record=factory_record, pool_source_audits=audits, source_artifacts=sources,
        producer_code_sources=code, source_artifacts_sha256=digest(sources),
        shared_active_heads=[list(k) for k in sorted(active)], active_scope="union_nonzero_future_needs_across_all_five_controls",
        active_heads_by_control={m:[list(k) for k in sorted(heads)] for m,heads in active_by_control.items()},
        downstream_score_pool_modes={m:"effect_only" if m=="effect_only" else "full" for m in MODES},
        same_admitted_heads_required_for_all_controls=True, failures=failures,
        outcomes_read=False, frozen_backbone_scores_recomputed=True,
        frozen_candidate_backbone_evaluated=bool(audits), heavy_models_loaded=False,
        attribution_or_cosmos_models_loaded=False, queries_started=0, actions_executed=0, fits_started=0)


def audit_complete_pipeline_controls(artifact, candidates, reader):
    """Rerun actual sources, not audit booleans, and compare exact five arms."""
    if not isinstance(artifact, Mapping) or artifact.get("schema") != OUTPUT_SCHEMA or artifact.get("scope") != SCOPE:
        raise ValueError("new actual current-block complete control artifact required")
    code = _code_sources()
    if artifact.get("producer_code_sources") != code:
        raise ValueError("actual loaded producer/context/parser code differs from frozen artifact")
    for ref in artifact["producer_code_sources"].values(): reader.path(ref)
    expected = produce_complete_pipeline_controls(artifact.get("pool_specs"), artifact.get("full_input_rows"), reader)
    if not expected["passed"]: raise ValueError("actual source-control audit failed:" + ";".join(expected["failures"]))
    if digest(artifact) != digest(expected):
        raise ValueError("complete source/control payload differs from independent recomputation")
    if digest(deployment_payload(candidates)) != digest(deployment_payload(expected["controls"]["full"])):
        raise ValueError("training full candidate payload differs from source-derived full pipeline")
    return dict(passed=True, scope=SCOPE, shared_active_heads=expected["shared_active_heads"],
        active_heads_by_control=expected["active_heads_by_control"],
        downstream_score_pool_modes=expected["downstream_score_pool_modes"],
        control_payload_sha256=expected["control_payload_sha256"],
        full_context_control_payload_sha256=digest(expected["factory_record"]),
        source_artifacts_sha256=expected["source_artifacts_sha256"],
        producer_code_sources=code, no_y_used=True,
        complete_pipeline_controls_ready=True, training_ready=False,
        not_history_reconstructed_closed_loop=True, empirical_benefit_claimed=False)
