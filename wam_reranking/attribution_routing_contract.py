"""Source-separated deployment attribution, without another class projection.

This additive contract does not edit a frozen classifier or any threshold.
``predicted`` is the final frozen route, even when an auxiliary unknown head
disagrees.  A routed UNKNOWN is not itself a physical invalidation certificate.
Physical factor probabilities, observation quality, and reason resolution are
represented separately.  The controlled replacements are offline ablations,
not additional measurements or evidence of a physical recovery.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from .contracts import AttributionOutput, CoarseCause, ConsistencyFactor, EvidenceQuality, TriValue
from .native_before_adapter import RAW_FACTORS, convert_frozen_attribution, digest

SCHEMA = "source_separated_frozen_attribution_routing_v1"
CONTROL_SCHEMA = "fixed_donor_source_separated_attribution_control_v1"
UNKNOWN_THRESHOLD = .64
FACTOR_THRESHOLD = .5
PHYSICAL_FACTORS = (
    ConsistencyFactor.WORLD_STATE_CONSISTENT.value,
    ConsistencyFactor.EXECUTION_CONTACT_CONSISTENT.value,
)
OBSERVATION_FACTOR = ConsistencyFactor.OBSERVATION_RELIABLE.value
RESOLUTION_FACTOR = ConsistencyFactor.CAUSE_RESOLVED.value
FORBIDDEN_FIELDS = frozenset((
    "actual_after", "outcomes", "success", "terminal_success", "labels",
    "teacher", "private_physics", "simulator_gt", "actual_observation_evidence",
))


def _probability(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} requires a literal finite probability")
    return float(value)


def _availability(value) -> tuple[bool, bool, bool]:
    if not isinstance(value, (tuple, list)) or len(value) != 3 or any(type(x) is not bool for x in value):
        raise ValueError("Actual primary/wrist/execution availability must be three literal bools")
    return tuple(value)


def _no_future(value):
    if isinstance(value, Mapping):
        if any(str(k).lower() in FORBIDDEN_FIELDS for k in value):
            raise ValueError("Future/outcome/teacher fields are not attribution inputs")
        for child in value.values():
            _no_future(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _no_future(child)


def _threshold_metadata(record):
    for name, expected in (("unknown_threshold", UNKNOWN_THRESHOLD), ("factor_threshold", FACTOR_THRESHOLD)):
        if name in record and record[name] != expected:
            raise ValueError(f"Frozen {name} metadata changed")


def attribution_snapshot(attribution) -> dict:
    """JSON-compatible exact policy-facing values, without invented metadata."""
    return dict(
        factor_probs=dict(attribution.factor_probs), class_probs=dict(attribution.class_probs),
        projected_cause=attribution.projected_cause.value, confidence=attribution.confidence,
        entropy=attribution.entropy, evidence_quality=asdict(attribution.evidence_quality),
        source_block_id=attribution.source_block_id,
        factor_states={k: v.value for k, v in attribution.factor_states.items()},
        factor_confidences=dict(attribution.factor_confidences),
    )


def decode_frozen_attribution(record, source_block_id, availability=None):
    """Return ``(attribution, audit)`` preserving the sole frozen final route.

    Optional ``normal_prob`` and threshold metadata are not fabricated when
    absent in a historical donor whitelist.  In all cases direct confidence
    must equal the probability of the selected final class.  A mismatch is an
    audit failure, not an invitation to guess a different cause or confidence.
    """
    if not isinstance(record, Mapping) or not isinstance(source_block_id, str) or not source_block_id:
        raise ValueError("Source-scoped block and raw frozen record are required")
    _no_future(record)
    _threshold_metadata(record)
    raw_hash = digest(dict(record))
    if "normal_prob" in record:
        _probability(record["normal_prob"], "auxiliary normal")
    if "normal_prob" in record and "unknown_threshold" in record:
        attribution, native_audit = convert_frozen_attribution(record, source_block_id)
        decoder = "native_before_adapter.convert_frozen_attribution"
    else:
        cause = CoarseCause(record["predicted"])
        cp, fp = record["class_probs"], record["factor_prob"]
        if not isinstance(cp, Mapping) or set(cp) != {c.value for c in CoarseCause}:
            raise ValueError("All original direct coarse probabilities are required")
        classes = {k: _probability(v, "direct class " + k) for k, v in cp.items()}
        if not math.isclose(sum(classes.values()), 1., abs_tol=1e-5):
            raise ValueError("Direct class distribution must sum to one")
        confidence = _probability(record["confidence"], "selected final route confidence")
        if not math.isclose(confidence, classes[cause.value], rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Selected-label confidence does not match direct[selected]")
        if not isinstance(fp, Mapping) or set(fp) != set(RAW_FACTORS):
            raise ValueError("All original four anomaly-head probabilities are required")
        raw = {k: _probability(v, "anomaly factor " + k) for k, v in fp.items()}
        _probability(record["unknown_prob"], "auxiliary unknown")
        conflict = raw["cross_view_conflict"] >= FACTOR_THRESHOLD
        if conflict and cause is not CoarseCause.UNKNOWN:
            raise ValueError("Frozen final route conflicts with cross-view-conflict contract")
        if cause is CoarseCause.NORMAL and any(p >= FACTOR_THRESHOLD for p in raw.values()):
            raise ValueError("Frozen final normal conflicts with its all-factor-negative contract")
        actual = _availability((record["primary_available"], record["wrist_available"],
                                record["action_record_available"]))
        observation = 1. - max(raw["visual_evidence_corrupted"], raw["cross_view_conflict"])
        quality = EvidenceQuality(actual[0] and observation >= FACTOR_THRESHOLD,
                                  actual[1] and observation >= FACTOR_THRESHOLD, actual[2], conflict)
        factors = {
            OBSERVATION_FACTOR: observation,
            PHYSICAL_FACTORS[0]: 1. - raw["object_or_environment_state_changed"],
            PHYSICAL_FACTORS[1]: 1. - raw["execution_or_contact_deviated"],
            ConsistencyFactor.TASK_STAGE_CONSISTENT.value: 1.,
            RESOLUTION_FACTOR: float(cause is not CoarseCause.UNKNOWN),
        }
        states = {k: TriValue.TRUE if p >= FACTOR_THRESHOLD else TriValue.FALSE for k, p in factors.items()}
        attribution = AttributionOutput(
            factors, classes, cause, confidence,
            -sum(p * math.log(p) for p in classes.values() if p), quality, source_block_id,
            states, {k: max(p, 1.-p) for k, p in factors.items()},
        )
        native_audit = None
        decoder = "equivalent_optional_metadata_decoder"
    if availability is not None:
        actual = _availability(availability)
        q = attribution.evidence_quality
        attribution = replace(attribution, evidence_quality=EvidenceQuality(
            q.primary_reliable and actual[0], q.wrist_reliable and actual[1],
            q.execution_reliable and actual[2], q.cross_view_conflict))
    else:
        actual = _availability((record["primary_available"], record["wrist_available"],
                                record["action_record_available"]))
    audit = dict(
        schema=SCHEMA, decoder=decoder, raw_record_payload_sha256=raw_hash,
        native_adapter_audit=native_audit, frozen_final_cause=attribution.projected_cause.value,
        selected_direct_confidence=attribution.confidence, final_route_is_sole_projection=True,
        class_probs_unchanged=True, raw_factor_probabilities_unchanged=True,
        auxiliary_unknown_probability=_probability(record["unknown_prob"], "auxiliary unknown"),
        auxiliary_normal_probability=record.get("normal_prob"),
        auxiliary_normal_probability_available="normal_prob" in record,
        auxiliary_unknown_disagrees_final_route=(record["unknown_prob"] >= UNKNOWN_THRESHOLD
            and attribution.projected_cause is not CoarseCause.UNKNOWN),
        reason_resolution=dict(final_route_resolved=attribution.projected_cause is not CoarseCause.UNKNOWN,
            semantics="noncalibrated_final_frozen_route_flag", physical_invalidation_certificate=False),
        physical_evidence={k: dict(consistency_probability=attribution.factor_probs[k],
            state=attribution.factor_states[k].value, confidence=attribution.factor_confidences[k])
            for k in PHYSICAL_FACTORS},
        visual_evidence=asdict(attribution.evidence_quality), actual_availability=list(actual),
        cause_resolved_is_calibrated_probability=False,
        cause_resolved_confidence_is_physical_confidence=False,
        unknown_implies_blanket_physical_invalidation=False,
        thresholds_unchanged=True, model_or_prediction_modified=False,
        output_payload_sha256=digest(attribution_snapshot(attribution)),
    )
    return attribution, audit


@dataclass(frozen=True)
class ControlledAttribution:
    """Duck-compatible attribution whose partial-control conflicts are explicit.

    Old ``AttributionOutput`` forbids a known route paired with cross-view
    conflict.  A cause-only/quality-only intervention can legitimately create
    that combination; silently changing its class would defeat the control.
    Such a combination therefore carries an UNKNOWN resolution state and zero
    resolution confidence, while preserving all donor/recipient probabilities.
    No physical state is changed merely to make the channels look coherent.
    """
    factor_probs: Mapping[str, float]
    class_probs: Mapping[str, float]
    projected_cause: CoarseCause
    confidence: float
    entropy: float
    evidence_quality: EvidenceQuality
    source_block_id: str
    factor_states: Mapping[str, TriValue]
    factor_confidences: Mapping[str, float]
    semantic_conflicts: tuple[str, ...] = ()

    def __post_init__(self):
        expected = {x.value for x in ConsistencyFactor}
        if any(set(getattr(self, key)) != expected for key in ("factor_probs", "factor_states", "factor_confidences")):
            raise ValueError("All factor channels are required")
        if set(self.class_probs) != {x.value for x in CoarseCause}:
            raise ValueError("All class channels are required")
        for group in (self.factor_probs, self.class_probs, self.factor_confidences):
            for key, value in group.items():
                _probability(value, key)
        if not math.isclose(sum(self.class_probs.values()), 1., abs_tol=1e-5):
            raise ValueError("Class probabilities must sum to one")
        if not math.isclose(self.confidence, self.class_probs[self.projected_cause.value],
                            rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Selected direct confidence cannot change in a control")
        if any(not isinstance(v, TriValue) for v in self.factor_states.values()):
            raise ValueError("Literal tri-valued factor states are required")
        if not self.source_block_id or not math.isfinite(self.entropy) or self.entropy < 0:
            raise ValueError("Source-scoped block and finite entropy are required")

    def factor(self, factor: ConsistencyFactor) -> float:
        return float(self.factor_probs[factor.value])

    def factor_state(self, factor: ConsistencyFactor) -> TriValue:
        return self.factor_states[factor.value]

    def factor_confidence(self, factor: ConsistencyFactor) -> float:
        return float(self.factor_confidences[factor.value])


def replace_attribution_channels(recipient, donor, *, mode, source_block_id, availability):
    """Fixed-donor cause-only, quality-only or whole-output offline intervention.

    Cause content = final route/class distribution + WORLD/EXECUTION factors.
    Quality = OBS factor + learned visual reliability/cross-view conflict.
    Task-stage carrier is always retained from the recipient.  Actual command
    record availability is never copied from the donor.  Neither donor nor
    recipient is mutated, and no class projection or probability boosting runs.
    """
    if mode not in ("cause_only", "quality_only", "all"):
        raise ValueError("Explicit cause_only/quality_only/all channel control required")
    if not isinstance(source_block_id, str) or not source_block_id:
        raise ValueError("Recipient source-scoped block is required")
    actual = _availability(availability)
    cause_source = donor if mode in ("cause_only", "all") else recipient
    quality_source = donor if mode in ("quality_only", "all") else recipient
    factors = dict(recipient.factor_probs)
    states = {f.value: recipient.factor_state(f) for f in ConsistencyFactor}
    confidences = {f.value: recipient.factor_confidence(f) for f in ConsistencyFactor}
    for k in (*PHYSICAL_FACTORS, RESOLUTION_FACTOR):
        factors[k] = cause_source.factor_probs[k]
        states[k] = cause_source.factor_state(ConsistencyFactor(k))
        confidences[k] = cause_source.factor_confidence(ConsistencyFactor(k))
    factors[OBSERVATION_FACTOR] = quality_source.factor_probs[OBSERVATION_FACTOR]
    states[OBSERVATION_FACTOR] = quality_source.factor_state(ConsistencyFactor.OBSERVATION_RELIABLE)
    confidences[OBSERVATION_FACTOR] = quality_source.factor_confidence(ConsistencyFactor.OBSERVATION_RELIABLE)
    q = quality_source.evidence_quality
    quality = EvidenceQuality(q.primary_reliable and actual[0], q.wrist_reliable and actual[1],
                              actual[2], q.cross_view_conflict)
    conflicts = []
    if quality.cross_view_conflict and cause_source.projected_cause is not CoarseCause.UNKNOWN:
        conflicts.append("known_final_route_with_cross_view_conflict")
    if cause_source.projected_cause is CoarseCause.NORMAL and states[OBSERVATION_FACTOR] is TriValue.FALSE:
        conflicts.append("normal_final_route_with_unreliable_observation")
    if conflicts:
        states[RESOLUTION_FACTOR] = TriValue.UNKNOWN
        confidences[RESOLUTION_FACTOR] = 0.
    output = ControlledAttribution(
        factors, dict(cause_source.class_probs), cause_source.projected_cause,
        cause_source.confidence, cause_source.entropy, quality, source_block_id,
        states, confidences, tuple(conflicts))
    recipient_payload = attribution_snapshot(recipient)
    donor_payload = attribution_snapshot(donor)
    audit = dict(
        schema=CONTROL_SCHEMA, mode=mode,
        recipient_source_block_id=recipient.source_block_id, donor_source_block_id=donor.source_block_id,
        recipient_attribution_payload_sha256=digest(recipient_payload),
        donor_attribution_payload_sha256=digest(donor_payload),
        output_attribution_payload_sha256=digest(attribution_snapshot(output)),
        cause_channels_source="donor" if cause_source is donor else "recipient",
        quality_channels_source="donor" if quality_source is donor else "recipient",
        stage_channel_source="recipient", actual_execution_availability_source="recipient_actual_record",
        actual_availability=list(actual), semantic_conflicts=list(conflicts),
        contradictory_resolution_marked_unknown=bool(conflicts),
        probabilities_reprojected=False, probabilities_boosted=False,
        physical_factor_confidences_unchanged=True,
        unknown_implies_blanket_physical_invalidation=False,
        same_fixed_donor_for_all_decompositions=True,
    )
    return output, audit
