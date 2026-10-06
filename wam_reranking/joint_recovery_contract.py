"""Independent whole-run recovery contract; no fitting and no frozen changes.

Entry prerequisites, future prerequisite use, and next-boundary recovery goals
are different objects. A goal prediction never certifies an earlier hard gate.
The gate fails the entire new run if any active target or input lacks evidence;
it does not silently fit a partially admitted or zero-head model.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from typing import Mapping

SCHEMA = "whole_run_cause_recovery_contract_v1"
INPUT_SCHEMA = "actual_before_and_predicted_candidate_input_v1"
BUNDLE_SCHEMA = "whole_run_cause_recovery_training_bundle_v1"
AT_USE_LABEL_SCHEMA = "certified_predicate_valid_at_use_time_v1"
GOAL_LABEL_SCHEMA = "certified_next_boundary_recovery_goal_v1"
RESIDUAL_CAP = .1
PREDICATES = ("target_visible", "target_pose_current", "target_reachable",
              "grasped", "lifted", "receptacle_visible", "place_ready")
OBSERVER = "observer_evidence_available"
TARGETS = PREDICATES + (OBSERVER,)
CAUSES = ("normal", "visual_occlusion", "object_shift",
          "execution_contact_deviation", "unknown")
VISUAL = frozenset(("target_visible", "target_pose_current", "receptacle_visible", OBSERVER))
EDGES = frozenset((("target_visible", "target_pose_current"),
    ("target_pose_current", "target_reachable"), ("target_reachable", "grasped"),
    ("execution_consistent", "grasped"), ("grasped", "lifted"),
    ("lifted", "place_ready"), ("receptacle_visible", "place_ready")))
ROOTS = {"object_shift": {"target_pose_current"},
         "execution_contact_deviation": {"execution_consistent"}}
IDENTITY_KEYS = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id")
CONTRACT_TESTS = ("normal_no_caused_need", "occlusion_epistemic_only",
    "unknown_no_physical_guess", "true_dependency_paths", "initial_unknown_no_new_deficit",
    "entry_not_backfilled", "goal_does_not_unlock_gate", "masked_same_candidate_information",
    "shuffled_same_candidate_information", "no_dag_only_propagated_credit",
    "effect_only_equal_information", "same_wrapper_tie_fallback_budget", "bounded_centered_score")
INPUT_AUDIT_FLAGS = ("all_field_sources_verified", "actual_before_boundary_verified",
    "candidate_predictions_generated_before_execution", "quality_observer_deployable",
    "no_actual_future_or_private_teacher_in_x", "old_weighted604_not_used_as_raw_input",
    "inherited_membership_preserved", "all_labels_y_only", "control_raw_inputs_equal")

HEAD_SEMANTICS = {
 "target_visible": "Relevant target has usable certified actual visual evidence at the specified time; absence of a certificate is unknown, not physical absence.",
 "target_pose_current": "Correct target identity and freshly usable target-relative localization at the specified time; visibility or joint motion alone is insufficient.",
 "target_reachable": "Task-specific observable and audited motion/reachability predicate at the specified time; no workspace or collision threshold is invented by this module.",
 "grasped": "Narrow currently-held/carried predicate, not force closure. Sustained independently certified target/EEF common motion may be sufficient; negatives require independent actual loss/separation.",
 "lifted": "Target is still carried and observably detached from its support with an audited reference at this time; an earlier rise event alone is insufficient.",
 "receptacle_visible": "Correct relevant receptacle has usable certified actual visual evidence at the specified time, not merely another scene object.",
 "place_ready": "Task-specific correct target-anchor relation, retained holding and readiness before release; 2D overlap, release or final success alone does not establish it.",
 OBSERVER: "Explicit information-acquisition outcome: previously insufficient observer evidence is usable at the next boundary. Not physical visibility=false, semantic-conflict resolution, or pose/holding truth."
}


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(name + " must be a finite literal number")
    return float(value)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _strict_keys(record, keys, name):
    if not isinstance(record, Mapping) or set(record) != set(keys):
        raise ValueError(name + " exact fields required")


def _identity(row):
    item = row.get("identity")
    _strict_keys(item, IDENTITY_KEYS, "source identity")
    if any(type(item[k]) is not int or item[k] < 0 for k in ("task", "state", "candidate_id")):
        raise ValueError("literal nonnegative task/state/candidate ids required")
    if any(not isinstance(item[k], str) or not item[k] for k in ("dataset", "suite", "observed_block_id")):
        raise ValueError("dataset/suite/block provenance required")
    return tuple(item[k] for k in IDENTITY_KEYS)


def file_sha(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class ArtifactReader:
    """Verify declared bytes; producer audit is additionally mandatory.

    SHA and a role string alone do not prove absence of upstream leakage. This
    class never upgrades those declarations into a source-producer audit.
    """
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.verified = {}

    def path(self, ref):
        if not isinstance(ref, Mapping) or not _sha(ref.get("sha256")) or not isinstance(ref.get("path"), str):
            raise ValueError("hash-bound artifact reference required")
        path = Path(ref["path"])
        if not path.is_absolute():
            path = self.base_path / path
        path = path.resolve()
        key = (str(path), ref["sha256"])
        if key not in self.verified:
            if not path.is_file() or file_sha(path) != ref["sha256"]:
                raise ValueError("artifact missing or changed: " + str(path))
            self.verified[key] = True
        return path

    def json(self, ref):
        return json.loads(self.path(ref).read_text(encoding="utf-8"))


def _source(ref, role, reader):
    _strict_keys(ref, ("path", "sha256", "role", "available_at_boundary",
                      "relative_step", "private_teacher"), "input source")
    if (ref["role"] != role or ref["available_at_boundary"] is not True or
        type(ref["relative_step"]) is not int or ref["relative_step"] != 0 or
        ref["private_teacher"] is not False):
        raise ValueError("only verified actual-before / preexecution-prediction X is allowed")
    reader.path(ref)


def validate_x(x, reader):
    """New explicit B, never a renamed attribution-weighted old 604 vector."""
    _strict_keys(x, ("schema", "actual_before", "candidate_predicted", "planned_actions",
                    "planned_actions_source", "task_context"), "candidate X")
    if x["schema"] != INPUT_SCHEMA:
        raise ValueError("new explicit before/predicted input schema required")
    before = x["actual_before"]
    _strict_keys(before, ("primary_rgb", "wrist_rgb", "proprio", "proprio_source",
                         "quality", "quality_source"), "actual-before X")
    for name in ("primary_rgb", "wrist_rgb", "proprio_source"):
        _source(before[name], "actual_before", reader)
    if not isinstance(before["proprio"], list) or len(before["proprio"]) != 9:
        raise ValueError("actual-before 9D proprio required")
    for value in before["proprio"]:
        _number(value, "before proprio")
    _strict_keys(before["quality"], ("primary_reliable", "wrist_reliable", "execution_reliable",
        "cross_view_conflict", "observer_evidence_available"), "before-only observer quality")
    if any(type(before["quality"][k]) is not bool for k in before["quality"] if k != "observer_evidence_available"):
        raise ValueError("literal before-quality booleans required")
    if before["quality"]["observer_evidence_available"] is not None and type(before["quality"]["observer_evidence_available"]) is not bool:
        raise ValueError("before observer availability is literal bool or unknown")
    _source(before["quality_source"], "deployable_before_observer", reader)
    predicted = x["candidate_predicted"]
    _strict_keys(predicted, ("primary_endpoint_rgb", "wrist_endpoint_rgb", "visual",
        "visual_source", "predicted_for_step", "intermediate_frames_available"), "predicted candidate X")
    for name in ("primary_endpoint_rgb", "wrist_endpoint_rgb", "visual_source"):
        _source(predicted[name], "candidate_prediction", reader)
    if predicted["predicted_for_step"] != 16 or type(predicted["predicted_for_step"]) is not int:
        raise ValueError("endpoint prediction must be explicitly for16")
    if type(predicted["intermediate_frames_available"]) is not bool:
        raise ValueError("intermediate prediction availability must be explicit, not invented")
    from .recovery_contract import VISUAL_EVIDENCE_KEYS
    _strict_keys(predicted["visual"], VISUAL_EVIDENCE_KEYS, "predicted visual value/mask slots")
    for name, item in predicted["visual"].items():
        _strict_keys(item, ("value", "available"), "visual field " + name)
        if type(item["available"]) is not bool:
            raise ValueError("literal visual availability required")
        if item["available"]:
            if not -1 <= _number(item["value"], name) <= 1:
                raise ValueError("existing clipped predicted visual range exceeded")
        elif item["value"] is not None:
            raise ValueError("unavailable predicted field must remain None, not zero evidence")
    actions = x["planned_actions"]
    if not isinstance(actions, list) or len(actions) != 16 or any(not isinstance(a, list) or len(a) != 7 for a in actions):
        raise ValueError("complete planned16x7 commands required")
    for action in actions:
        for value in action:
            _number(value, "planned command")
    _source(x["planned_actions_source"], "planned_command", reader)
    context = x["task_context"]
    _strict_keys(context, ("language", "target", "anchor", "relation", "entity_binding_source"), "before task binding")
    if any(not isinstance(context[k], str) or not context[k] for k in ("language", "target", "anchor", "relation")):
        raise ValueError("language and target/anchor binding required")
    _source(context["entity_binding_source"], "frozen_task_binding", reader)
    from .recovery_input_sources import validate_input_source_contents
    validate_input_source_contents(x, reader)
    return digest(x)


def expected_requirements(stage, actions, relation):
    """Inherit original use indices, not future goal indices or label times."""
    table = {
        "approach": {"target_visible": .5, "target_pose_current": .6},
        "grasp": {"target_pose_current": .8, "target_reachable": .8},
        "lift": {"grasped": .85}, "transport": {"grasped": .8, "lifted": .75},
        "place": {"lifted": .75, "receptacle_visible": .6, "place_ready": .7},
        "observe": {}, "uncertain": {}
    }
    if stage not in table:
        raise ValueError("typed preexecution candidate stage required")
    if not isinstance(relation, str) or not relation:
        raise ValueError("frozen task relation binding required, never inferred from Y")
    requirements = table[stage]
    if relation.lower() in ("open", "closed", "articulated"):
        # Exact frozen typed_gate_effect contract: do not require carrying the
        # entire drawer. Keep stage-specific original command use indices.
        requirements = {"target_visible": .5, "target_pose_current": .6}
    close = [i for i, a in enumerate(actions) if a[6] > .25]
    up = [i for i, a in enumerate(actions) if a[2] > 0]
    release = [i for i, a in enumerate(actions) if a[6] < -.25 and any(j <= i for j in close)]
    when = close[0] if stage == "grasp" and close else up[0] if stage == "lift" and up else 0
    return [{"predicate": p, "deadline": release[0] if p == "place_ready" and release else when,
             "confidence": confidence, "kind": "required_use"} for p, confidence in requirements.items()]


def _objective(obj):
    _strict_keys(obj, ("predicate", "deadline", "confidence", "kind"), "objective")
    p, t, kind = obj["predicate"], obj["deadline"], obj["kind"]
    if p not in TARGETS or type(t) is not int or not 0 <= t <= 16 or not 0 <= _number(obj["confidence"], "objective confidence") <= 1:
        raise ValueError("invalid target/deadline/confidence")
    if kind == "required_use":
        if p not in PREDICATES or t >= 16:
            raise ValueError("required use retains original step0..15")
    elif kind == "next_boundary_goal":
        if p not in PREDICATES or t != 16:
            raise ValueError("predicate recovery goal is boundary16, not an earlier proof")
    elif kind == "epistemic_goal":
        if p != OBSERVER or t != 16:
            raise ValueError("separate observer availability goal at16 required")
    else:
        raise ValueError("unrecognized objective type")
    return p, t, kind


def _need(need, row, objectives):
    _strict_keys(need, ("cause", "predicate", "deadline", "kind", "weight", "propagated",
        "path", "origin", "old_value", "old_confidence", "new_value", "source_block_id"), "cause need")
    key = (need["predicate"], need["deadline"], need["kind"])
    if key not in objectives or need["cause"] not in CAUSES or type(need["propagated"]) is not bool:
        raise ValueError("need cannot invent candidate objectives or causes")
    w = _number(need["weight"], "cause need weight")
    if not 0 <= w <= 1 or not isinstance(need["source_block_id"], str) or not need["source_block_id"]:
        raise ValueError("finite referenced before need required")
    cause, predicate, path = need["cause"], need["predicate"], need["path"]
    if cause == "normal":
        if w != 0:
            raise ValueError("normal cannot invent caused recovery needs")
        return key
    if cause in ("unknown", "visual_occlusion") and predicate not in VISUAL:
        raise ValueError("unknown/occlusion cannot guess physical missing state")
    if not isinstance(path, list) or not path or path[-1] != predicate or len(set(path)) != len(path):
        raise ValueError("literal acyclic source-owned need path required")
    if predicate == OBSERVER:
        purpose_valid = (row["stage"] == "observe" and row.get("purpose") == "information_acquisition") or (
            row.get("purpose") == "task_with_observation_recovery")
        if (cause not in ("unknown", "visual_occlusion") or need["origin"] != "before_epistemic_insufficiency"
            or need["propagated"] or path != [OBSERVER] or not purpose_valid
            or row["x"]["actual_before"]["quality"]["observer_evidence_available"] is not False):
            raise ValueError("observer goal needs an explicit primary/secondary information goal and insufficient before evidence")
    else:
        if (need["origin"] != "current_attribution_invalidation" or need["old_value"] != "true"
            or _number(need["old_confidence"], "old confidence") <= 0 or need["new_value"] not in ("false", "unknown")):
            raise ValueError("initial UNKNOWN/FALSE cannot become new attribution-created deficit")
        if cause in ("unknown", "visual_occlusion"):
            if path != [predicate] or need["propagated"]:
                raise ValueError("epistemic scope is direct, not physical propagation")
        else:
            if path[0] not in ROOTS[cause] or any(e not in EDGES for e in zip(path, path[1:])):
                raise ValueError("actual route root and every legal DAG edge required")
            if need["propagated"] != (len(path) > 1):
                raise ValueError("propagation flag cannot fabricate/remove dependency provenance")
    return key


def validate_candidate(row, reader):
    _identity(row)
    if row.get("split") not in ("train", "val") or not isinstance(row.get("pool_id"), str) or not row["pool_id"]:
        raise ValueError("explicit development split/pool required")
    if row.get("role") not in ("native_terminal_rank", "native_paired_recovery_auxiliary",
        "historical_selected_auxiliary", "constructed_recovery_auxiliary"):
        raise ValueError("source roles cannot be promoted silently")
    fingerprint = validate_x(row.get("x"), reader)
    lineage = row.get("before_lineage")
    _strict_keys(lineage, ("snapshot_source", "query_source"), "before non-input lineage")
    for name, ref in lineage.items():
        _strict_keys(ref, ("path", "sha256", "metadata_only", "private_teacher"), "before lineage source")
        if (ref["metadata_only"] is not True or type(ref["private_teacher"]) is not bool or
            (name == "snapshot_source" and ref["private_teacher"] is not True)):
            raise ValueError("private runtime/query provenance is metadata-only, never deployment X")
        reader.path(ref)
    if row.get("requirements") != expected_requirements(row.get("stage"), row["x"]["planned_actions"],
        row["x"]["task_context"]["relation"]):
        raise ValueError("original planned requirement/use-time table was changed")
    goals = row.get("goals")
    if not isinstance(goals, list):
        raise ValueError("independent goal table required even when empty")
    objectives = {}
    for item in row["requirements"] + goals:
        key = _objective(item)
        if key in objectives:
            raise ValueError("duplicate recovery objective")
        if item in goals and key[2] == "required_use":
            raise ValueError("goal cannot masquerade as an earlier prerequisite")
        objectives[key] = item
    needs = row.get("needs")
    if not isinstance(needs, list):
        raise ValueError("typed before-derived needs required")
    seen = set()
    for need in needs:
        key = _need(need, row, objectives)
        nk = (need["cause"], key)
        if nk in seen:
            raise ValueError("duplicate cause/objective recovery credit")
        seen.add(nk)
    if any(k[2] != "required_use" and not any(_need(n, row, objectives) == k and n["weight"] > 0 for n in needs) for k in objectives):
        raise ValueError("recovery goals must be source-owned before needs, not Y-selected progress goals")
    before = row.get("before_verification")
    _strict_keys(before, PREDICATES, "all-seven entry verifier slots")
    for item in before.values():
        _strict_keys(item, ("value", "deployable_observer_verified"), "entry predicate verifier")
        if item["value"] not in ("true", "false", "unknown") or type(item["deployable_observer_verified"]) is not bool:
            raise ValueError("literal before predicate value and observation provenance required")
    unsafe_entry = [obj["predicate"] for obj in row["requirements"] if obj["deadline"] == 0 and
        (before[obj["predicate"]]["value"] != "true" or not before[obj["predicate"]]["deployable_observer_verified"])]
    if unsafe_entry and row.get("entry_route") != "safe_fallback":
        raise ValueError("uncertified entry facts require safe fallback; a goal cannot unlock them")
    if row.get("entry_route") not in ("verified", "safe_fallback"):
        raise ValueError("explicit entry verifier/fallback route required")
    if row["role"] != "native_terminal_rank" and any(k in row for k in ("terminal_success", "terminal_outcome")):
        raise ValueError("auxiliary/probe cannot acquire invented native terminal outcomes")
    if row["role"] == "native_terminal_rank" and type(row.get("terminal_success")) is not bool:
        raise ValueError("native rank supervision retains literal terminal success Y")
    _number(row.get("backbone_score"), "frozen backbone score")
    return fingerprint, objectives


def deployment_payload(rows):
    """Canonical outcome-free producer-audit binding, not a GT-free proof.

    The independent producer must audit the actual referenced contents and
    the actual before attribution/belief changes of these exact rows. An old
    audit of a different feature table cannot certify this new payload.
    """
    keys = ("identity", "pool_id", "split", "role", "stage", "purpose", "x", "before_lineage",
            "requirements", "goals", "needs", "before_verification", "entry_route", "backbone_score")
    return [{k: row.get(k) for k in keys} for row in rows]


def _pool_identity(row):
    ident = row["identity"]
    return (ident["dataset"], ident["suite"], ident["task"], ident["state"],
            ident["observed_block_id"], row["pool_id"], row["split"])


def validate_observer_quality_certificate(cert, label, reader, definition):
    """Auditable observer-quality Y, not a physical-visibility certificate."""
    import numpy as np
    from .observed_recovery import map_endpoint
    from . import observed_recovery
    from .recovery_input_sources import _rgb
    record = reader.json(cert.get("after_quality_source"))
    if (record.get("schema") != "actual_after_frozen_deployment_observer_quality_v1" or
        record.get("role") != "offline_supervision_only" or record.get("identity") != label["identity"] or
        type(record.get("step")) is not int or record["step"] != 16):
        raise ValueError("observer quality must bind actual own-candidate after step16 as Y only")
    quality = record.get("quality")
    names = ("primary_reliable", "wrist_reliable", "execution_reliable", "cross_view_conflict", "observer_evidence_available")
    _strict_keys(quality, names, "after frozen observer quality")
    if any(type(quality[k]) is not bool for k in names):
        raise ValueError("observer quality requires literal measured booleans")
    for name in ("observer_code_source", "observer_model_source"):
        ref = record.get(name)
        reader.path(ref)
        if (not isinstance(definition, Mapping) or definition.get("frozen_" + name.removesuffix("_source") + "_sha256") != ref["sha256"]):
            raise ValueError("after observer code/model differs from independently frozen semantic contract")
    if record["observer_code_source"]["sha256"] != file_sha(observed_recovery.__file__):
        raise ValueError("after observer does not use the frozen map_endpoint implementation")
    refs = record.get("actual_after_rgb_sources")
    _strict_keys(refs, ("primary", "wrist"), "actual after observer RGB sources")
    for view in refs:
        _rgb(refs[view], reader, "actual after observer " + view)
    path = reader.path(record.get("subject_maps_source"))
    if path.suffix.lower() != ".npy":
        raise ValueError("actual after observer maps require immutable NPY")
    maps = np.load(path, allow_pickle=False)
    if maps.ndim != 3 or maps.shape[0] != 2 or min(maps.shape[1:]) < 1 or not np.isfinite(maps).all():
        raise ValueError("actual after observer requires two finite own-image relevance maps")
    usable = [map_endpoint(m)["usable"] for m in maps]
    available = not quality["cross_view_conflict"] and any(
        quality[view + "_reliable"] and usable[index] for index, view in enumerate(("primary", "wrist")))
    if quality["observer_evidence_available"] is not available or label["value"] is not available:
        raise ValueError("observer-quality label differs from actual frozen observer evaluation")
    binding = dict(identity=record["identity"], step=16, actual_after_rgb_sources=refs,
        subject_maps_source=record["subject_maps_source"], observer_code_source=record["observer_code_source"],
        observer_model_source=record["observer_model_source"])
    if record.get("observer_input_lineage_sha256") != digest(binding):
        raise ValueError("actual after observer input/model lineage binding missing")
    if label["value"] is True and cert.get("actual_rgb_observable") is not True:
        raise ValueError("positive observer quality still requires independent actual RGB identity evidence")
    return True


def control_rows(rows, mode):
    if mode not in ("full", "masked", "no_dag", "effect_only"):
        raise ValueError("shuffled needs require independently audited before-source substitution")
    result = deepcopy(rows)
    if mode in ("masked", "no_dag"):
        for row in result:
            for need in row["needs"]:
                if mode == "masked" or need["propagated"]:
                    need["weight"] = 0.
    return result


def shuffled_recovery_need_weights(rows, *, seed=0):
    """Split-local group derangement of weights, not an attribution rerun.

    No label/outcome is read. The recipient's cause, source and DAG path stay
    unchanged. Only a weight at the exact same predicate/use-time/kind is
    substituted. A missing objective or a lone group becomes explicit zero,
    never a fabricated physical need. This does not remove attribution's
    earlier effect on the common frozen belief/gates.
    """
    if type(seed) is not int or seed != 0:
        raise ValueError("predeclared shuffle seed0 required")
    result, grouped = deepcopy(rows), defaultdict(list)
    def group(row):
        ident = row["identity"]
        return (ident["dataset"], ident["suite"], ident["task"], ident["state"])
    for row in rows:
        _identity(row)
        if row.get("split") not in ("train", "val"):
            raise ValueError("shuffle cannot use confirmation or unlabeled split")
        grouped[(row["split"], group(row))].append(row)
    mapping = {}
    rng = random.Random(seed)
    for split in ("train", "val"):
        groups = sorted(g for s, g in grouped if s == split)
        rng.shuffle(groups)
        for index, receiver in enumerate(groups):
            mapping[(split, receiver)] = groups[(index + 1) % len(groups)] if len(groups) > 1 else None
    substitutions = []
    for original, row in zip(rows, result, strict=True):
        receiver_group = group(original)
        donor_group = mapping[(row["split"], receiver_group)]
        donors = [] if donor_group is None else sorted(grouped[(row["split"], donor_group)], key=_identity)
        for ni, need in enumerate(row["needs"]):
            objective = (need["predicate"], need["deadline"], need["kind"])
            matching = [donor for donor in donors if any((n["predicate"], n["deadline"], n["kind"]) == objective
                        and n["cause"] != "normal" for n in donor["needs"])]
            donor = matching[0] if matching else None
            donor_budget = 0. if donor is None else min(1., sum(n["weight"] for n in donor["needs"]
                if (n["predicate"], n["deadline"], n["kind"]) == objective and n["cause"] != "normal"))
            receiver_budget = sum(n["weight"] for n in original["needs"]
                if (n["predicate"], n["deadline"], n["kind"]) == objective and n["cause"] != "normal")
            # Replace one objective-level total, not the same donor total in
            # every cause slot. Keep the recipient's original cause shares.
            share = original["needs"][ni]["weight"] / receiver_budget if receiver_budget > 0 else 0.
            weight = donor_budget * share
            # A normal recipient remains a no-credit route, even if a donor
            # happened to have a physical recovery requirement.
            if need["cause"] == "normal":
                weight = 0.
            need["weight"] = weight
            substitutions.append(dict(seed=seed, split=row["split"], receiver_identity=deepcopy(row["identity"]),
                receiver_group=list(receiver_group), donor_group=None if donor_group is None else list(donor_group),
                objective=list(objective), need_index=ni, original_weight=original["needs"][ni]["weight"],
                substituted_weight=weight, donor_objective_total=donor_budget,
                receiver_original_objective_total=receiver_budget, receiver_cause_share=share,
                donor_identity=None if donor is None else deepcopy(donor["identity"]),
                donor_payload_sha256=None if donor is None else digest(deployment_payload([donor])[0]),
                reason="single_group_no_derangement" if donor_group is None else
                    "matched_before_need_weight" if donor is not None else "no_same_time_objective_in_donor",
                label_or_terminal_outcome_read=False, cause_path_source_unchanged=True))
    audit = dict(schema="split_local_shuffled_recovery_need_weights_v1", seed=seed,
        donors_split_local=True, donors_cross_group=True,
        does_not_claim_complete_attribution_removal_or_rerun=True,
        mapping=[dict(split=split, receiver_group=list(g), donor_group=None if d is None else list(d))
            for (split, g), d in sorted(mapping.items())],
        substitution_table_sha256=digest(substitutions))
    return result, substitutions, audit


def assert_equal_control_information(reference, *controls):
    def key(rows):
        result = {}
        for row in rows:
            ident = _identity(row)
            if ident in result:
                raise ValueError("control identities duplicated")
            # Need channel alone may change; gates, goals and raw B may not.
            common = {k: row.get(k) for k in ("pool_id", "split", "role", "stage", "purpose", "x", "before_lineage",
                "requirements", "goals", "before_verification", "entry_route", "backbone_score")}
            result[ident] = digest(common)
        return result
    expected = key(reference)
    if any(key(control) != expected for control in controls):
        raise ValueError("control changed ordinary input/objectives/gates/backbone/identity")
    return True


def score_pool(rows, probabilities, *, mode, reference_candidate_id, admitted_keys=None):
    """Synthetic/direct wiring only; caller must pass an entirely admitted fit.

    No fitting is provided here. Probabilities are model outputs from common B,
    never actual labels. Production must pass the SAME whole-run admitted_keys
    to every mode; a missing key is inactive for all modes, including effect-
    only. Default inference is only a synthetic wiring convenience, NOT data
    admission. This function cannot write belief or unlock an entry gate.
    """
    if mode not in ("full", "masked", "shuffled", "no_dag", "effect_only") or not rows:
        raise ValueError("explicit common pool and control required")
    ids = [r["identity"]["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids) or reference_candidate_id not in ids:
        raise ValueError("unique candidates and non-GT fixed reference required")
    explicit_admission_scope = admitted_keys is not None
    if admitted_keys is None:
        admitted_keys = {(n["predicate"], n["deadline"], n["kind"]) for r in rows for n in r["needs"]
            if n["weight"] > 0 and n["cause"] != "normal" and n["deadline"] > 0 and
            any(_objective(o) == (n["predicate"], n["deadline"], n["kind"]) and o["confidence"] > 0
                for o in r["requirements"] + r["goals"])}
    else:
        admitted_keys = {tuple(k) for k in admitted_keys}
    if any(len(k) != 3 or k[0] not in TARGETS or type(k[1]) is not int or not 1 <= k[1] <= 16 or
           k[2] not in ("required_use", "next_boundary_goal", "epistemic_goal") for k in admitted_keys):
        raise ValueError("explicit valid admitted future head keys required")
    output = {}
    for row in rows:
        cid = row["identity"]["candidate_id"]
        terms = []
        all_objectives = row["requirements"] + row["goals"]
        for obj in all_objectives:
            key = _objective(obj)
            if key[1] == 0:
                terms.append(dict(objective=list(key), credit=0., inactive_reason="entry_only_actual_before_not_future_prediction"))
                continue
            if key not in admitted_keys:
                terms.append(dict(objective=list(key), credit=0., inactive_reason="head_not_empirically_admitted_in_shared_scope"))
                continue
            p = _number(probabilities[cid][key], "candidate model probability")
            ref = _number(probabilities[reference_candidate_id][key], "reference model probability")
            if not 0 <= p <= 1 or not 0 <= ref <= 1:
                raise ValueError("model probabilities must be in0..1")
            if mode == "effect_only":
                weight = obj["confidence"]
            else:
                weight = 0.
                for need in row["needs"]:
                    if (need["predicate"], need["deadline"], need["kind"]) != key or need["cause"] == "normal":
                        continue
                    if mode == "masked" or (mode == "no_dag" and need["propagated"]):
                        continue
                    weight += need["weight"] * obj["confidence"]
                weight = min(weight, obj["confidence"])
            terms.append(dict(objective=list(key), weight=weight,
                predicted_probability=p, reference_predicted_probability=ref,
                credit=weight * (p-ref), prediction_never_updates_entry_belief=True))
        denominator = max(1., sum(o["confidence"] for o in all_objectives))
        delta = RESIDUAL_CAP * sum(t["credit"] for t in terms) / denominator
        if abs(delta) > RESIDUAL_CAP + 1e-12:
            raise RuntimeError("single total recovery cap exceeded")
        output[cid] = dict(backbone_score=row["backbone_score"], recovery_residual=delta,
            total_score=row["backbone_score"]+delta, terms=terms,
            entry_route=row["entry_route"], hard_gate_unchanged=True,
            candidate_execution_allowed=row["entry_route"] == "verified",
            goals_never_override_safe_fallback=True,
            explicit_shared_admission_scope=explicit_admission_scope)
    return output


def audit_whole_training(bundle, *, base_path):
    """Fail the whole run on any gap. No partial-head waiver or auto-fit."""
    failures, reader = [], ArtifactReader(base_path)
    report = dict(schema=SCHEMA, passed=False, training_ready=False, training_started=False,
        partial_training_allowed=False, source_hashes_verified=0, active_heads=[], coverage={}, failures=failures)
    if not isinstance(bundle, Mapping) or bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("input_schema") != INPUT_SCHEMA:
        failures.append("new_bundle_and_before_predicted_input_schema_required")
        return report
    def check_artifact(ref, schema, flags):
        try:
            data = reader.json(ref)
            if data.get("schema") != schema or data.get("passed") is not True or any(data.get(k) is not True for k in flags):
                raise ValueError("independent audit flags/schema incomplete")
            return data
        except Exception as error:
            failures.append("artifact_audit:" + str(error))
            return None
    producer = check_artifact(bundle.get("input_producer_audit"), "whole_recovery_input_producer_audit_v1",
        INPUT_AUDIT_FLAGS + ("source_content_binding_verified", "before_attribution_belief_changes_bound"))
    control_audit = check_artifact(bundle.get("control_contract_audit"), "whole_recovery_control_contract_audit_v1", CONTRACT_TESTS)
    if bundle.get("preserved_inherited_rank_rows") != 768 or bundle.get("preserved_inherited_auxiliary_rows") != 957:
        failures.append("768_rank_and957_auxiliary_inheritance_not_preserved")
    if (not _sha(bundle.get("source_input_sha256_before")) or
        bundle.get("source_input_sha256_before") != bundle.get("source_input_sha256_after")):
        failures.append("immutable_source_input_sha_changed_or_missing")
    try:
        source_ref = bundle.get("immutable_source_input")
        source = reader.json(source_ref)
        if source_ref["sha256"] != bundle.get("source_input_sha256_before"):
            raise ValueError("source byte hash does not match before/after hashes")
        inherited = bundle.get("preserved_inherited_rows")
        _strict_keys(inherited, ("rank_rows", "auxiliary_rows"), "inherited original row lists")
        for section, length in (("rank_rows", 768), ("auxiliary_rows", 957)):
            if (not isinstance(source.get(section), list) or len(source[section]) != length or
                not isinstance(inherited[section], list) or len(inherited[section]) != length or
                [digest(r) for r in inherited[section]] != [digest(r) for r in source[section]]):
                raise ValueError("original row identity/order/content not preserved: " + section)
    except Exception as error:
        failures.append("immutable_source_membership:" + str(error))
    contracts = bundle.get("head_contracts", {})
    if not isinstance(contracts, Mapping):
        contracts = {}
    if set(contracts) != set(TARGETS):
        failures.append("all_seven_predicate_and_separate_observer_semantic_contracts_required")
    definitions = {}
    for target in TARGETS:
        definition = check_artifact(contracts.get(target), "observed_recovery_predicate_contract_v1",
            ("observable_definition_passed", "positive_definition_audited", "negative_definition_independent",
             "missing_is_masked_not_negative", "prefix_causal", "old_numeric_thresholds_unchanged",
             "entry_verification_safe"))
        if definition is not None and definition.get("predicate") != target:
            failures.append("semantic_contract_target_mismatch:" + target)
        definitions[target] = definition
    rows, lookup, active, groups, shared_before = bundle.get("candidates", []), {}, set(), {}, {}
    if not isinstance(rows, list) or not rows:
        failures.append("complete_explicit_candidate_rows_required")
        return report
    try:
        if (producer is None or producer.get("deployment_payload_sha256") != digest(deployment_payload(rows)) or
            not _sha(producer.get("need_factory_source_sha256")) or
            not _sha(producer.get("input_builder_source_sha256"))):
            raise ValueError("new exact X and before-needs producer audit binding incomplete")
    except Exception as error:
        failures.append("producer_payload_binding:" + str(error))
    try:
        controls = bundle.get("controls")
        _strict_keys(controls, ("full", "masked", "shuffled", "no_dag", "effect_only"), "all actual control row tables")
        if any(not isinstance(controls[mode], list) for mode in controls):
            raise ValueError("explicit row list in every control required")
        assert_equal_control_information(rows, *controls.values())
        for mode in ("full", "masked", "no_dag", "effect_only"):
            if digest(deployment_payload(controls[mode])) != digest(deployment_payload(control_rows(rows, mode))):
                raise ValueError("unexpected need-channel change in control: " + mode)
        control_hashes = {mode: digest(deployment_payload(control)) for mode, control in controls.items()}
        if control_audit is None or control_audit.get("control_payload_sha256") != control_hashes:
            raise ValueError("actual five control tables not bound to control audit")
        expected_shuffled, expected_substitutions, expected_shuffle_audit = shuffled_recovery_need_weights(rows)
        if (digest(deployment_payload(controls["shuffled"])) != digest(deployment_payload(expected_shuffled)) or
            bundle.get("shuffled_substitution_table") != expected_substitutions):
            raise ValueError("actual split-local weight shuffle or donor table disagrees with deterministic factory")
        shuffle = control_audit.get("shuffled_before_source_audit")
        if (not isinstance(shuffle, Mapping) or shuffle.get("passed") is not True or
            shuffle.get("seed") != 0 or shuffle.get("donors_split_local") is not True or
            shuffle.get("donors_cross_group") is not True or
            shuffle.get("semantic_conflicts_masked_not_normalized_into_physical_needs") is not True or
            not _sha(shuffle.get("factory_source_sha256")) or
            shuffle.get("substitution_table_sha256") != expected_shuffle_audit["substitution_table_sha256"]):
            raise ValueError("actual before-only fixed-seed shuffled donor lineage missing")
    except Exception as error:
        failures.append("control_payload_binding:" + str(error))
    pool_bindings = {}
    for i, row in enumerate(rows):
        try:
            ident = _identity(row)
            if ident in lookup:
                raise ValueError("duplicate six-field source identity")
            _, objectives = validate_candidate(row, reader)
            lookup[ident] = row
            group = (row["identity"]["suite"], row["identity"]["task"], row["identity"]["state"])
            if group in groups and groups[group] != row["split"]:
                raise ValueError("task/state train/val leakage")
            groups[group] = row["split"]
            before_key = _pool_identity(row)
            name_key = (row["pool_id"], row["split"])
            if name_key in pool_bindings and pool_bindings[name_key] != before_key:
                raise ValueError("pool string aliases different dataset/suite/task/state/block sources")
            pool_bindings[name_key] = before_key
            before_digest = digest(dict(actual_before=row["x"]["actual_before"], before_lineage=row["before_lineage"]))
            if before_key in shared_before and shared_before[before_key] != before_digest:
                raise ValueError("candidate pool does not share actual before state")
            shared_before[before_key] = before_digest
            active.update(_need(n, row, objectives) for n in row["needs"]
                if n["weight"] > 0 and n["cause"] != "normal" and n["deadline"] > 0
                and objectives[(n["predicate"], n["deadline"], n["kind"])]["confidence"] > 0)
        except Exception as error:
            failures.append("candidate:" + str(i) + ":" + str(error))
    if not active:
        failures.append("no_genuine_active_future_recovery_task_zero_head_fit_forbidden")
    report["active_heads"] = [list(k) for k in sorted(active)]
    counts = defaultdict(lambda: {"train_positive": 0, "train_negative": 0,
        "val_positive": 0, "val_negative": 0, "train_same_pool_time_pairs": 0, "val_same_pool_time_pairs": 0})
    peers, seen_labels = defaultdict(dict), set()
    labels = bundle.get("labels")
    if not isinstance(labels, list):
        failures.append("explicit_label_list_required_even_when_masked")
        labels = []
    for i, label in enumerate(labels):
        try:
            ident = _identity(label)
            row = lookup[ident]
            if label.get("split") != row["split"] or label.get("pool_id") != row["pool_id"]:
                raise ValueError("label identity/split/pool join mismatch")
            key = (label.get("predicate"), label.get("deadline"), label.get("kind"))
            label_key = (ident, key)
            if label_key in seen_labels:
                raise ValueError("duplicate six-field identity/target/time/kind Y")
            seen_labels.add(label_key)
            if key not in active:
                continue
            if type(label.get("mask")) is not bool:
                raise ValueError("literal observation mask required")
            if label["mask"] is False:
                continue
            if type(label.get("value")) is not bool or label.get("actual_evidence_role") != "offline_supervision_only":
                raise ValueError("certified literal Y only required")
            expected = AT_USE_LABEL_SCHEMA if key[2] == "required_use" else GOAL_LABEL_SCHEMA
            flags = ("identity_bound", "time_bound", "repeat_noise_verified", "negative_not_inferred_from_missing")
            flags += (("actual_rgb_measurement_verified", "no_physical_absence_claim") if key[0] == OBSERVER else ("actual_rgb_observable",))
            cert = check_artifact(label.get("certificate"), expected, flags)
            if cert is None:
                continue
            if (cert.get("predicate") != key[0] or cert.get("deadline") != key[1]
                or cert.get("kind") != key[2] or cert.get("value") is not label["value"]
                or cert.get("identity") != row["identity"]):
                raise ValueError("certificate target/time/value/source mismatch")
            if key[0] == OBSERVER:
                validate_observer_quality_certificate(cert, label, reader, definitions[OBSERVER])
            name = row["split"] + ("_positive" if label["value"] else "_negative")
            counts[key][name] += 1
            if row["role"] in ("native_paired_recovery_auxiliary", "constructed_recovery_auxiliary"):
                peers[(key, _pool_identity(row), row["split"])][row["identity"]["candidate_id"]] = label["value"]
        except Exception as error:
            failures.append("label:" + str(i) + ":" + str(error))
    for (key, pool, split), candidate_values in peers.items():
        if set(candidate_values.values()) == {True, False}:
            counts[key][split + "_same_pool_time_pairs"] += 1
    for key in sorted(active):
        coverage = counts[key]
        report["coverage"]["|".join(map(str, key))] = coverage
        if any(coverage[k] == 0 for k in coverage):
            failures.append("whole_run_head_supervision_gap:" + "|".join(map(str, key)))
    report["source_hashes_verified"] = len(reader.verified)
    report["passed"] = report["training_ready"] = not failures
    report["input_bundle_sha256"] = digest(bundle)
    report["all_future_use_and_goal_heads_required_together"] = True
    report["inactive_semantic_contract_is_not_empirical_head_admission"] = True
    report["entry0_is_before_verification_not_future_head_training"] = True
    report["raw_source_semantics_inherited_from_exact_bound_independent_producer_audit"] = True
    report["artifact_hash_or_role_string_alone_certifies_gt_free_x"] = False
    report["current_control_scope"] = "direct_recovery_need_score_channel_only"
    report["input_supervision_and_direct_channel_contracts_passed"] = not failures
    report["complete_pipeline_controls_ready"] = False
    failures.append("complete_attribution_and_belief_control_producer_not_available")
    report["passed"] = report["training_ready"] = False
    report["gate3_or_main_experiment_ready_claimed"] = False
    return report
