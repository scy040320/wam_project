"""Direct, bounded recovery-score wiring; no fitting or rollout in this module.

The frozen V8/V9 residuals are untouched.  Candidate information B is common
to every control, while predicted attribution/before-belief needs N are a
separate channel. Only an admitted, predicate-specific at-use-time head
may contribute N * P(predicate valid when it is used) to the score.
Predictions never establish a current belief fact or override a hard gate.

Admission metadata is a frozen *reference* to an independently checked label
manifest, not a substitute for opening and verifying that manifest offline.
The synthetic tests of this module prove wiring, never real trained benefit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .contracts import PREDICATES
from .recovery_contract import (
    COMMAND_FEATURE_NAMES, FEATURE_NAMES, FEATURE_SCHEMA, ROUTES,
    VISUAL_EVIDENCE_KEYS,
)

RAW_SCHEMA = "preexecution_candidate_command_visual_v1"
SCORE_SCHEMA = "direct_certified_recovery_score_v1"
LABEL_SCHEMA = "certified_predicate_valid_at_use_time_v1"
RESIDUAL_CAP = .1
RAW_FEATURE_NAMES = (
    tuple("command." + name for name in COMMAND_FEATURE_NAMES)
    + tuple("visual." + key + "." + part for key in VISUAL_EVIDENCE_KEYS
            for part in ("value", "available"))
)
NEED_SOURCE = "predicted_attribution_and_preexecution_belief"


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _vector(value, width, name):
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"invalid finite {name} vector")
    result.setflags(write=False)
    return result


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RawCandidateFeatures:
    """Only planned commands and candidate *predicted* visual proxies.

    ``visual_reconstruction_complete=False`` means route-weighted historical
    features cannot identify whether a zero visual field was present.  It is
    not a negative observation and cannot enable a recovery correction.
    """

    values: np.ndarray
    visual_reconstruction_complete: bool
    feature_names: tuple[str, ...] = RAW_FEATURE_NAMES
    schema: str = RAW_SCHEMA

    def __post_init__(self):
        if self.schema != RAW_SCHEMA or tuple(self.feature_names) != RAW_FEATURE_NAMES:
            raise ValueError("raw candidate schema/names mismatch")
        if type(self.visual_reconstruction_complete) is not bool:
            raise ValueError("raw candidate reconstruction status must be explicit")
        values = _vector(self.values, len(RAW_FEATURE_NAMES), "raw candidate")
        for i in range(len(COMMAND_FEATURE_NAMES), len(values), 2):
            if not -1 <= values[i] <= 1 or values[i + 1] not in (0., 1.):
                raise ValueError("visual proxy must retain clipped value and literal availability")
            if values[i + 1] == 0. and values[i] != 0.:
                raise ValueError("unavailable visual proxy cannot contain an invented value")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "feature_names", RAW_FEATURE_NAMES)

    def fingerprint(self):
        return _digest(dict(schema=self.schema, names=list(self.feature_names),
                            values=self.values.tolist(),
                            visual_reconstruction_complete=self.visual_reconstruction_complete))


def extract_raw_candidate_features(features, *, feature_names=FEATURE_NAMES,
                                   feature_schema=FEATURE_SCHEMA,
                                   original_visual: Mapping[str, float] | None = None):
    """Undo route multiplication without discarding ordinary candidate info.

    All nonzero ``route.visual.value / route.visual.available`` ratios must
    agree; contradictory routes are rejected rather than averaged.  Values
    are already clipped by the historical producer, so the new B explicitly
    inherits that clipping.  If every route is zero, an original, hash-audited
    pre-execution candidate-visual record is needed to disambiguate absence.
    ``original_visual`` cannot contain outcomes, teacher atoms or extra keys.
    """
    if feature_schema != FEATURE_SCHEMA or tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("historical recovery feature schema/names mismatch")
    features = _vector(features, len(FEATURE_NAMES), "historical recovery")
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    if original_visual is not None:
        if not isinstance(original_visual, Mapping) or set(original_visual) - set(VISUAL_EVIDENCE_KEYS):
            raise ValueError("only pre-execution visual proxy keys are allowed")
        original_visual = {key: _finite(value, key) for key, value in original_visual.items()}
    raw = [float(features[index["command." + name]]) for name in COMMAND_FEATURE_NAMES]
    reconstructed = {}
    any_available = False
    for key in VISUAL_EVIDENCE_KEYS:
        ratios = []
        for route in ROUTES:
            value = float(features[index[f"{route}.visual.{key}.value"]])
            availability = float(features[index[f"{route}.visual.{key}.available"]])
            if not 0. <= availability <= 1. or not -1. <= value <= 1.:
                raise ValueError("invalid route-weighted visual proxy")
            if availability == 0.:
                if value != 0.:
                    raise ValueError("route value exists without route availability")
                continue
            ratio = value / availability
            if not -1. - 1e-8 <= ratio <= 1. + 1e-8:
                raise ValueError("route visual ratio exceeds the producer's clipping contract")
            ratios.append(float(np.clip(ratio, -1., 1.)))
        if ratios and not np.allclose(ratios, ratios[0], rtol=0., atol=1e-8):
            raise ValueError("contradictory raw visual values across attribution routes")
        if original_visual is not None:
            present = key in original_visual
            value = float(np.clip(original_visual.get(key, 0.), -1., 1.))
            if ratios and (not present or abs(ratios[0] - value) > 1e-8):
                raise ValueError("original candidate visual does not match weighted source")
            reconstructed[key] = (value if present else 0., float(present))
        else:
            reconstructed[key] = (ratios[0] if ratios else 0., float(bool(ratios)))
        any_available |= bool(ratios)
    for key in VISUAL_EVIDENCE_KEYS:
        raw.extend(reconstructed[key])
    # An all-zero routed record cannot tell true absence from zero route mass.
    complete = original_visual is not None or any_available
    return RawCandidateFeatures(np.asarray(raw), complete)


@dataclass(frozen=True)
class RecoveryNeed:
    """Deployable cause-specific need, never simulator/teacher missing state.

    ``weight`` already includes predicted cause probability and the degree of
    a still-invalid before-belief fact.  A generic initially-unknown fact must
    not be relabeled as an attribution-created missing prerequisite.
    """

    cause: str
    predicate: str
    weight: float
    requirement_time: int
    propagated: bool = False
    source: str = NEED_SOURCE
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.cause not in ROUTES or self.predicate not in PREDICATES:
            raise ValueError("unknown cause/predicate need")
        if self.source != NEED_SOURCE:
            raise ValueError("teacher, GT or future execution cannot define deployment needs")
        if type(self.requirement_time) is not int or not 0 <= self.requirement_time <= 16:
            raise ValueError("requirement time must be the action-prefix index in [0,16]")
        weight = _finite(self.weight, "need weight")
        if not 0 <= weight <= 1 or type(self.propagated) is not bool:
            raise ValueError("invalid need weight/propagation status")
        if not self.evidence_ids or any(not isinstance(v, str) or not v for v in self.evidence_ids):
            raise ValueError("before-belief/attribution provenance references are required")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def effective_weight(self):
        # Normal has no attribution-created recovery need.  Unknown does not
        # get a guessed physical cause; unreliable vision does not negate grip.
        if self.cause == "normal":
            return 0.
        visibility = {"target_visible", "target_pose_current", "receptacle_visible"}
        if self.cause in {"unknown", "visual_occlusion"} and self.predicate not in visibility:
            return 0.
        return self.weight


@dataclass(frozen=True)
class HeadAdmission:
    """Frozen train-only provenance and observable prefix-label coverage.

    Only predicate-specific validity AT use time counts. An earlier grasp
    event followed by a drop is not a true grasp prerequisite at use time.
    In particular a finger contact is
    not ``grasped``, joint movement is not ``target_pose_current``, and release
    is not ``placed``.  Their seven physical atoms may be diagnostic labels,
    but converting them requires a separately audited predicate contract.
    Counts refer to certified positives/negatives at EACH use time, not raw
    physics or absence of a certified positive.  Metadata cannot verify the
    source files by itself: the preparation gate must bind the checked hashes.
    """

    predicate: str
    target_name: str
    label_manifest_sha256: str
    observation_contract_sha256: str
    training_input_sha256: str
    positive_by_time: tuple[int, ...]
    negative_by_time: tuple[int, ...]
    paired_difference_by_time: tuple[int, ...]
    input_roles: tuple[str, ...] = ("train",)
    label_schema: str = LABEL_SCHEMA
    label_integrity_passed: bool = False
    prefix_causal: bool = False
    masked_is_not_negative: bool = False
    private_teacher_excluded_from_x: bool = False

    def __post_init__(self):
        if self.predicate not in PREDICATES or self.target_name != f"{self.predicate}.valid_at_use_time":
            raise ValueError("raw contact/motion/release atoms cannot silently become predicate recovery")
        if self.label_schema != LABEL_SCHEMA or tuple(self.input_roles) != ("train",):
            raise ValueError("predicate prefix-label schema and train-only fitting are required")
        object.__setattr__(self, "input_roles", tuple(self.input_roles))
        for name in ("positive_by_time", "negative_by_time", "paired_difference_by_time"):
            counts = tuple(getattr(self, name))
            if len(counts) != 17 or any(type(v) is not int or v < 0 for v in counts):
                raise ValueError("certified label counts must cover prefix times 0..16")
            object.__setattr__(self, name, counts)
        for name in ("label_integrity_passed", "prefix_causal", "masked_is_not_negative",
                     "private_teacher_excluded_from_x"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("head admission audit flags must be explicit booleans")

    def inactive_reason(self, deadline):
        if not all(_is_sha(getattr(self, name)) for name in
                   ("label_manifest_sha256", "observation_contract_sha256", "training_input_sha256")):
            return "unverified_or_unbound_label_provenance"
        if not all((self.label_integrity_passed, self.prefix_causal,
                    self.masked_is_not_negative, self.private_teacher_excluded_from_x)):
            return "supervision_admission_not_passed"
        if deadline == 0:
            return "future_prediction_cannot_establish_entry_fact"
        if not self.positive_by_time[deadline] or not self.negative_by_time[deadline]:
            return "no_certified_positive_and_negative_at_use_time"
        if not self.paired_difference_by_time[deadline]:
            return "no_observable_same_pool_recovery_difference_at_use_time"
        return None


@dataclass(frozen=True)
class LinearRecoveryHead:
    """A serialized learned at-use-time head; this module never fits it.

    Predicate validity can FALL after dropping/releasing, so time probability
    is not forced monotonic. An endpoint-only head cannot claim availability
    before16. A first-event-only certificate cannot admit an at-use-time head.
    All support tests use COMMON raw candidate B, never manipulated causes.
    """

    admission: HeadAdmission
    weights: np.ndarray
    bias: float
    time_weight: float
    support_min: np.ndarray
    support_max: np.ndarray
    prediction_kind: str = "learned_predicate_probability_at_use"

    def __post_init__(self):
        if not isinstance(self.admission, HeadAdmission):
            raise ValueError("a predicate-specific supervision admission is required")
        for name in ("weights", "support_min", "support_max"):
            object.__setattr__(self, name, _vector(getattr(self, name), len(RAW_FEATURE_NAMES), name))
        if np.any(self.support_min > self.support_max):
            raise ValueError("invalid train-only raw candidate support")
        bias, tw = _finite(self.bias, "bias"), _finite(self.time_weight, "time_weight")
        if self.prediction_kind not in ("learned_predicate_probability_at_use", "predicted_endpoint_only"):
            raise ValueError("invalid causal predicate-at-use prediction contract")
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "time_weight", tw)

    def predict(self, raw: RawCandidateFeatures, deadline: int):
        if not isinstance(raw, RawCandidateFeatures) or type(deadline) is not int or not 0 <= deadline <= 16:
            raise ValueError("prediction accepts pre-execution raw candidate and an explicit use time")
        reason = self.admission.inactive_reason(deadline)
        if reason:
            return None, reason
        if not raw.visual_reconstruction_complete:
            return None, "raw_candidate_visual_not_reconstructable"
        if self.prediction_kind == "predicted_endpoint_only" and deadline < 16:
            return None, "endpoint_cannot_certify_earlier_prerequisite"
        if np.any(raw.values < self.support_min - 1e-8) or np.any(raw.values > self.support_max + 1e-8):
            return None, "outside_train_raw_candidate_support"
        logit = float(raw.values @ self.weights) + self.bias + self.time_weight * deadline / 16.
        return float(1. / (1. + math.exp(-float(np.clip(logit, -60., 60.))))), None


@dataclass(frozen=True)
class CandidateRequirement:
    """Cause-independent candidate requirement; identical across controls."""

    predicate: str
    requirement_time: int
    confidence: float = 1.
    source: str = "candidate_plan_or_prediction"

    def __post_init__(self):
        if self.predicate not in PREDICATES or self.source != "candidate_plan_or_prediction":
            raise ValueError("candidate requirements must be plan/prediction-derived, not GT")
        if type(self.requirement_time) is not int or not 0 <= self.requirement_time <= 16:
            raise ValueError("invalid candidate requirement use time")
        confidence = _finite(self.confidence, "requirement confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("invalid requirement confidence")
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class CandidateRecoveryInput:
    candidate_id: int
    backbone_score: float
    raw: RawCandidateFeatures
    needs: tuple[RecoveryNeed, ...]
    requirements: tuple[CandidateRequirement, ...] = ()

    def __post_init__(self):
        if type(self.candidate_id) is not int or self.candidate_id < 0:
            raise ValueError("explicit nonnegative candidate identity required")
        if not isinstance(self.raw, RawCandidateFeatures):
            raise ValueError("raw pre-execution candidate features required")
        needs = tuple(self.needs)
        if any(not isinstance(v, RecoveryNeed) for v in needs):
            raise ValueError("typed deployment needs required, not label dictionaries")
        if len({(v.cause, v.predicate) for v in needs}) != len(needs):
            raise ValueError("duplicate cause/predicate need would amplify recovery credit")
        requirements = tuple(self.requirements)
        if (any(not isinstance(r, CandidateRequirement) for r in requirements) or
            len({r.predicate for r in requirements}) != len(requirements)):
            raise ValueError("typed unique candidate requirements are required")
        requirement_times = {r.predicate: r.requirement_time for r in requirements}
        if any(requirement_times.get(n.predicate) != n.requirement_time for n in needs):
            raise ValueError("cause need cannot invent/change candidate predicate requirements or use times")
        object.__setattr__(self, "backbone_score", _finite(self.backbone_score, "frozen backbone score"))
        object.__setattr__(self, "needs", needs)
        object.__setattr__(self, "requirements", requirements)


def masked_attribution_input(candidate):
    """Change ONLY attribution-derived needs; all ordinary information stays."""
    return replace(candidate, needs=tuple(replace(n, weight=0.) for n in candidate.needs))


def no_dependency_input(candidate):
    """Keep raw proposals/requirements; remove ONLY propagated need credit."""
    return replace(candidate, needs=tuple(replace(n, weight=0.) if n.propagated else n for n in candidate.needs))


def replace_control_needs(candidate, needs):
    """For split-audited shuffled attribution; labels and B are never shuffled."""
    return replace(candidate, needs=tuple(needs))


def assert_equal_candidate_information(reference: Sequence[CandidateRecoveryInput],
                                       *controls: Sequence[CandidateRecoveryInput]):
    """Exact B/backbone/identity equality before interpreting a cause ablation."""
    def keyed(rows):
        if not rows or len({r.candidate_id for r in rows}) != len(rows):
            raise ValueError("control pools must preserve unique candidate identities")
        return {r.candidate_id: (r.raw.fingerprint(), r.backbone_score,
            _digest([asdict(requirement) for requirement in r.requirements])) for r in rows}
    expected = keyed(reference)
    for control in controls:
        if keyed(control) != expected:
            raise ValueError("cause control changed raw candidate information/backbone/identity")
    return True


@dataclass(frozen=True)
class DirectRecoveryModel:
    heads: tuple[LinearRecoveryHead, ...]
    residual_cap: float = RESIDUAL_CAP
    schema: str = SCORE_SCHEMA

    def __post_init__(self):
        heads = tuple(self.heads)
        if self.schema != SCORE_SCHEMA or self.residual_cap != RESIDUAL_CAP:
            raise ValueError("direct recovery model/cap contract mismatch")
        if any(not isinstance(h, LinearRecoveryHead) for h in heads):
            raise ValueError("direct score only accepts typed admitted recovery heads")
        if len({h.admission.predicate for h in heads}) != len(heads):
            raise ValueError("duplicate predicate recovery heads")
        object.__setattr__(self, "heads", heads)

    def score_pool(self, candidates: Sequence[CandidateRecoveryInput], *, reference_candidate_id: int):
        """Score only; the unchanged common wrapper still owns gate/tie/fallback.

        The caller must use the SAME frozen-backbone reference in every arm,
        selected without true execution outcomes.  Centering against it removes
        a shared saturated pool offset. Cause weights for one predicate are
        capped jointly at its common candidate requirement confidence. The
        denominator uses ONLY that cause-independent requirement table, so
        dropping a zero-gain DAG term cannot amplify a retained direct term.
        Positive weights/probabilities keep TOTAL correction within +/- .1.
        No independent ranking head can bypass the recovery probabilities.
        """
        candidates = tuple(candidates)
        if not candidates or any(not isinstance(c, CandidateRecoveryInput) for c in candidates):
            raise ValueError("typed nonempty pre-execution candidate pool required")
        by_id = {c.candidate_id: c for c in candidates}
        if len(by_id) != len(candidates) or reference_candidate_id not in by_id:
            raise ValueError("unique candidates and an explicit frozen non-GT reference required")
        reference = by_id[reference_candidate_id]
        heads = {h.admission.predicate: h for h in self.heads}
        output = {}
        for candidate in candidates:
            contributions = []
            for need in candidate.needs:
                requirement = next(r for r in candidate.requirements if r.predicate == need.predicate)
                weight = need.effective_weight() * requirement.confidence
                probability, reference_probability, reason = None, None, None
                if weight == 0.:
                    reason = "no_active_cause_specific_missing_prerequisite"
                elif need.predicate not in heads:
                    reason = "no_admitted_predicate_recovery_head"
                else:
                    head = heads[need.predicate]
                    probability, reason = head.predict(candidate.raw, need.requirement_time)
                    if reason is None:
                        reference_probability, reason = head.predict(reference.raw, need.requirement_time)
                    if reason is not None:
                        probability = reference_probability = None
                term = 0.
                if reason is None:
                    term = weight * (probability - reference_probability)
                contributions.append(dict(cause=need.cause, predicate=need.predicate,
                    requirement_time=need.requirement_time, propagated=need.propagated,
                    effective_need_weight=weight, predicted_recovery_probability=probability,
                    reference_predicted_recovery_probability=reference_probability,
                    weighted_probability_difference=term, inactive_reason=reason,
                    prediction_is_not_current_belief_evidence=True))
            # Multiple cause routes may need the SAME predicate, but must not
            # multiply its budget. This is not a learned/free ranking bypass.
            numerator = 0.
            for requirement in candidate.requirements:
                active = [term for term in contributions
                          if term["predicate"] == requirement.predicate
                          and term["inactive_reason"] is None]
                weight_sum = sum(term["effective_need_weight"] for term in active)
                scale = (min(1., requirement.confidence / weight_sum)
                         if weight_sum else 0.)
                for term in active:
                    term["weighted_probability_difference"] *= scale
                    term["joint_cause_budget_scale"] = scale
                    numerator += term["weighted_probability_difference"]
            denominator = max(1., sum(r.confidence for r in candidate.requirements))
            correction = self.residual_cap * numerator / denominator
            if not math.isfinite(correction) or abs(correction) > self.residual_cap + 1e-12:
                raise RuntimeError("direct recovery correction violated its single total cap")
            output[candidate.candidate_id] = dict(
                frozen_backbone_score=candidate.backbone_score,
                cause_recovery_residual=float(correction),
                total_score=candidate.backbone_score + float(correction),
                reference_candidate_id=reference_candidate_id,
                recovery_contributions=contributions,
                raw_candidate_sha256=candidate.raw.fingerprint(),
                common_requirement_normalization=denominator,
                residual_cap=self.residual_cap,
                no_hard_gate_override=True,
                auxiliary_predictions_do_not_restore_belief=True,
                observed_execution_labels_used_as_score_input=False)
        return output

    def score_effect_only_pool(self, candidates: Sequence[CandidateRecoveryInput], *, reference_candidate_id: int):
        """Fair ordinary-effect comparator using the EXACT same trained heads.

        This arm deliberately does not read ``needs``. Common candidate-plan
        requirements and all raw candidate information remain available; only
        attribution/belief weighting is absent. Thus a generic effect predictor
        improving selection cannot be mislabeled as an attribution benefit.
        Admission, support guards, reference and total +/- .1 cap are identical
        to the cause-weighted arm. This method neither trains nor gates actions.
        """
        candidates = tuple(candidates)
        if not candidates or any(not isinstance(c, CandidateRecoveryInput) for c in candidates):
            raise ValueError("typed nonempty pre-execution candidate pool required")
        by_id = {c.candidate_id: c for c in candidates}
        if len(by_id) != len(candidates) or reference_candidate_id not in by_id:
            raise ValueError("unique candidates and an explicit frozen non-GT reference required")
        reference = by_id[reference_candidate_id]
        heads = {h.admission.predicate: h for h in self.heads}
        output = {}
        for candidate in candidates:
            contributions = []
            for requirement in candidate.requirements:
                probability = reference_probability = None
                reason = "no_admitted_predicate_recovery_head"
                if requirement.predicate in heads:
                    head = heads[requirement.predicate]
                    probability, reason = head.predict(candidate.raw, requirement.requirement_time)
                    if reason is None:
                        reference_probability, reason = head.predict(reference.raw, requirement.requirement_time)
                    if reason is not None:
                        probability = reference_probability = None
                term = 0. if reason is not None else requirement.confidence * (probability - reference_probability)
                contributions.append(dict(predicate=requirement.predicate,
                    requirement_time=requirement.requirement_time,
                    predicted_recovery_probability=probability,
                    reference_predicted_recovery_probability=reference_probability,
                    weighted_probability_difference=term, inactive_reason=reason))
            denominator = max(1., sum(r.confidence for r in candidate.requirements))
            correction = self.residual_cap * sum(t["weighted_probability_difference"] for t in contributions) / denominator
            if not math.isfinite(correction) or abs(correction) > self.residual_cap + 1e-12:
                raise RuntimeError("ordinary effect comparator violated common residual cap")
            output[candidate.candidate_id] = dict(frozen_backbone_score=candidate.backbone_score,
                effect_only_residual=float(correction), total_score=candidate.backbone_score + float(correction),
                reference_candidate_id=reference_candidate_id, effect_contributions=contributions,
                raw_candidate_sha256=candidate.raw.fingerprint(), common_requirement_normalization=denominator,
                residual_cap=self.residual_cap, attribution_or_belief_needs_used=False,
                same_recovery_heads_and_candidate_information=True, no_hard_gate_override=True,
                observed_execution_labels_used_as_score_input=False)
        return output

    def to_record(self):
        heads = []
        for head in self.heads:
            record = asdict(head)
            for name in ("weights", "support_min", "support_max"):
                record[name] = getattr(head, name).tolist()
            heads.append(record)
        record = dict(schema=self.schema, raw_feature_schema=RAW_SCHEMA,
            raw_feature_names=list(RAW_FEATURE_NAMES), label_schema=LABEL_SCHEMA,
            residual_cap=self.residual_cap, heads=heads,
            no_training_in_module=True, predictions_are_not_current_belief=True,
            score_is_direct_recovery_probability_times_need=True)
        record["model_sha256"] = _digest(record)
        return record

    @classmethod
    def from_record(cls, record):
        payload = dict(record)
        digest = payload.pop("model_sha256", None)
        if digest != _digest(payload):
            raise ValueError("direct recovery serialized model hash mismatch")
        if (record.get("raw_feature_schema") != RAW_SCHEMA or
            tuple(record.get("raw_feature_names", ())) != RAW_FEATURE_NAMES or
            record.get("label_schema") != LABEL_SCHEMA):
            raise ValueError("direct recovery raw/label contract mismatch")
        heads = []
        for item in record["heads"]:
            values = dict(item)
            values["admission"] = HeadAdmission(**values["admission"])
            heads.append(LinearRecoveryHead(**values))
        return cls(tuple(heads), record["residual_cap"], record["schema"])
