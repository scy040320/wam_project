"""Actual-only current geometry and independently validated fact admission.

Geometry is an uncalibrated image-space proxy, not a physical-state witness.
The default registry admits no predicates.  A fact extractor needs an
explicitly reviewed validation receipt AND an actual-input validator; merely
presenting matching hashes or a plausible certificate is never sufficient.
No candidate action or forecast is accepted by these interfaces.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np


GEOMETRY_EXTRACTOR_VERSION = "actual_current_geometry_proxy_v1"
SOURCE_ROLE = "runtime_actual_only"
HASH_SCHEME = "sha256_canonical_npy_v1"
KINDS = MappingProxyType({
    "target_visible": frozenset({"actual_entity_visibility"}),
    "receptacle_visible": frozenset({"actual_anchor_visibility"}),
    "target_pose_current": frozenset({"actual_identity_localization"}),
    "target_reachable": frozenset({"actual_robot_target_reachability"}),
    "grasped": frozenset({"actual_sustained_target_eef_motion", "actual_holding_interface"}),
    "lifted": frozenset({"actual_target_support_separation"}),
    "place_ready": frozenset({"actual_target_anchor_readiness"}),
    "placed": frozenset({"actual_supported_placement"}),
})


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA256")


def array_sha256(array: np.ndarray) -> str:
    """Hash deterministic NPY bytes, including shape and dtype, not a PNG."""
    array = np.asarray(array)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError("Actual arrays must be finite and contain no Python objects")
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class SourceSnapshot:
    pool_id: str
    entity_key: str
    anchor_key: str
    block_index: int
    policy_step: int
    primary_sha256: str
    wrist_sha256: str | None
    proprio_sha256: str | None
    extractor_version: str = GEOMETRY_EXTRACTOR_VERSION
    source_role: str = SOURCE_ROLE
    hash_scheme: str = HASH_SCHEME
    predecision: bool = True

    def __post_init__(self):
        for name in ("pool_id", "entity_key", "anchor_key", "extractor_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"Explicit {name} required")
        if type(self.block_index) is not int or type(self.policy_step) is not int or min(self.block_index, self.policy_step) < 0:
            raise ValueError("Exact nonnegative block and policy step required")
        if self.source_role != SOURCE_ROLE or self.predecision is not True or self.hash_scheme != HASH_SCHEME:
            raise ValueError("Only PRE actual observation snapshots are admissible")
        _digest(self.primary_sha256, "primary_sha256")
        for name in ("wrist_sha256", "proprio_sha256"):
            if getattr(self, name) is not None:
                _digest(getattr(self, name), name)

    @property
    def identity_sha256(self):
        import json
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ActualObservationInputs:
    snapshot: SourceSnapshot
    primary: np.ndarray
    wrist: np.ndarray | None = None
    proprio: np.ndarray | None = None

    def __post_init__(self):
        for name, digest_name in (("primary", "primary_sha256"), ("wrist", "wrist_sha256"), ("proprio", "proprio_sha256")):
            value, expected = getattr(self, name), getattr(self.snapshot, digest_name)
            if value is None:
                if expected is not None:
                    raise ValueError(f"Missing actual {name}")
                continue
            array = np.asarray(value)
            if expected is None or array_sha256(array) != expected:
                raise ValueError(f"Actual {name} hash mismatch")
            if name in {"primary", "wrist"} and (array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8):
                raise ValueError("Current RGB must be uint8 H x W x 3")
            if name == "proprio" and array.ndim != 1:
                raise ValueError("Actual proprio must be a finite vector")
            array = array.copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def make_actual_inputs(*, pool_id, entity_key, anchor_key, block_index, policy_step,
                       actual_primary, actual_wrist=None, actual_proprio=None,
                       extractor_version=GEOMETRY_EXTRACTOR_VERSION):
    """Create the one per-pool source shared by every candidate."""
    snapshot = SourceSnapshot(pool_id, entity_key, anchor_key, block_index, policy_step,
        array_sha256(actual_primary), None if actual_wrist is None else array_sha256(actual_wrist),
        None if actual_proprio is None else array_sha256(actual_proprio), extractor_version)
    return ActualObservationInputs(snapshot, actual_primary, actual_wrist, actual_proprio)


def _assert_actual_inputs(inputs):
    if not isinstance(inputs, ActualObservationInputs):
        raise ValueError("Typed actual observation inputs required")
    for name in ("primary", "wrist", "proprio"):
        value, digest = getattr(inputs, name), getattr(inputs.snapshot, name + "_sha256")
        if (value is None) != (digest is None) or (value is not None and array_sha256(value) != digest):
            raise ValueError("Actual source mutated after snapshot admission")


def _geometry(relevance):
    array = np.asarray(relevance, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2 or not np.isfinite(array).all():
        raise ValueError("Actual relevance map must be finite H x W")
    array = np.clip(array, 0, 1)
    total = float(array.sum())
    if total <= 1e-8:
        return dict(x=.5, y=.5, spread=0., contrast=0., mass=0., peak=0.)
    yy, xx = np.mgrid[0:array.shape[0], 0:array.shape[1]].astype(float)
    xx /= array.shape[1]-1; yy /= array.shape[0]-1
    weights = array / total
    x, y = float((weights*xx).sum()), float((weights*yy).sum())
    q50, q90, q99 = np.quantile(array, (.5, .9, .99))
    contrast = float(np.clip(.75*max(0., q99-q50)/max(q99, .05) + .25*max(0., q99-q90)/max(q99, .05), 0, 1))
    return dict(x=x, y=y, spread=float(np.sqrt((weights*((xx-x)**2+(yy-y)**2)).sum())),
        contrast=contrast, mass=float(array.mean()), peak=float(q99))


def _pair(a, b):
    a, b = np.clip(np.asarray(a, float), 0, 1), np.clip(np.asarray(b, float), 0, 1)
    if a.shape != b.shape:
        raise ValueError("Current maps within a view must align")
    ga, gb = _geometry(a), _geometry(b)
    return dict(distance=float(np.hypot(ga["x"]-gb["x"], ga["y"]-gb["y"])),
        affinity=float(np.minimum(a, b).sum()/max(float(np.maximum(a, b).sum()), 1e-8)),
        localization_contrast=min(ga["contrast"], gb["contrast"]))


def current_proxies_from_maps(inputs: ActualObservationInputs, *, actual_maps: Mapping[str, Mapping[str, np.ndarray]]):
    """Export current-only diagnostic proxies; absolutely no fact admission.

    Maps are caller-produced actual-input maps.  This low-level interface
    makes no claim to authenticate the external map producer.  Use
    ``extract_current_proxies`` to invoke it on actual RGB only.
    """
    _assert_actual_inputs(inputs)
    if inputs.snapshot.extractor_version != GEOMETRY_EXTRACTOR_VERSION:
        raise ValueError("Explicit current geometry extractor version required")
    expected = {"primary"} | ({"wrist"} if inputs.wrist is not None else set())
    if set(actual_maps) != expected:
        raise ValueError("Only available actual primary/wrist views accepted")
    views = {}
    for view in sorted(expected):
        maps = actual_maps[view]
        if set(maps) != {"subject", "anchor", "gripper"}:
            raise ValueError("Only actual subject/anchor/gripper maps accepted")
        views[view] = dict(subject=_geometry(maps["subject"]), anchor=_geometry(maps["anchor"]),
            gripper=_geometry(maps["gripper"]), subject_anchor=_pair(maps["subject"], maps["anchor"]),
            subject_gripper=_pair(maps["subject"], maps["gripper"]))
    return dict(schema=GEOMETRY_EXTRACTOR_VERSION, snapshot=asdict(inputs.snapshot),
        snapshot_sha256=inputs.snapshot.identity_sha256, views=views, candidate_invariant=True,
        confidence_calibrated=False, physical_state_certified=False, predicate_facts=[],
        cross_view_identity_certified=False, role="current_geometry_proxy_only")


def extract_current_proxies(inputs: ActualObservationInputs, *, map_extractor: Callable):
    """Call an injected frozen localizer with actual RGB and language only."""
    _assert_actual_inputs(inputs)
    images = {"primary": inputs.primary}
    if inputs.wrist is not None:
        images["wrist"] = inputs.wrist
    maps = map_extractor(images=MappingProxyType(images), entity_key=inputs.snapshot.entity_key,
                         anchor_key=inputs.snapshot.anchor_key, gripper_key="robot gripper")
    return current_proxies_from_maps(inputs, actual_maps=maps)


@dataclass(frozen=True)
class RecomputedPredicate:
    value: str
    confidence: float
    evidence_id: str

    def __post_init__(self):
        if self.value not in {"true", "false", "unknown"} or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Invalid independently recomputed predicate")
        if not self.evidence_id:
            raise ValueError("Actual evidence identifier required")


@dataclass(frozen=True)
class ValidatedExtractor:
    version: str
    validation_report_sha256: str
    kinds: Mapping[str, frozenset[str]]
    validator: Callable[[Mapping, ActualObservationInputs], RecomputedPredicate]

    def __post_init__(self):
        _digest(self.validation_report_sha256, "validation_report_sha256")
        if not self.version or self.version == GEOMETRY_EXTRACTOR_VERSION or not callable(self.validator):
            raise ValueError("Proxy-only geometry cannot be registered as a fact extractor")
        kinds = {p: frozenset(k) for p, k in self.kinds.items()}
        if not kinds or any(p not in KINDS or not k or not k <= KINDS[p] for p, k in kinds.items()):
            raise ValueError("Predicate-specialized evidence kinds required")
        object.__setattr__(self, "kinds", MappingProxyType(kinds))


@dataclass(frozen=True)
class WitnessRegistry:
    """Reviewed receipts are explicit trusted configuration, never witness data."""
    approved_validation_sha256: frozenset[str] = frozenset()
    extractors: tuple[ValidatedExtractor, ...] = ()

    def __post_init__(self):
        receipts = frozenset(self.approved_validation_sha256)
        for digest in receipts:
            _digest(digest, "approved validation receipt")
        if len({e.version for e in self.extractors}) != len(self.extractors):
            raise ValueError("Duplicate fact extractor version")
        if any(e.validation_report_sha256 not in receipts for e in self.extractors):
            raise ValueError("Fact extractor lacks an independently reviewed validation receipt")
        object.__setattr__(self, "approved_validation_sha256", receipts)
        object.__setattr__(self, "extractors", tuple(self.extractors))


@dataclass(frozen=True)
class WitnessAdmission:
    accepted: bool
    reason: str
    fact: Mapping | None = None
    audit: Mapping = field(default_factory=dict)


def _contains_forbidden_proof_source(proof):
    if isinstance(proof, Mapping):
        for key, value in proof.items():
            words = str(key).lower()
            if any(term in words for term in ("forecast", "predicted", "candidate", "sim_state", "simulator",
                "ground_truth", "qpos", "qvel", "geom_id", "body_id", "physics", "intervention", "success", "outcome", "actions")):
                return True
            if _contains_forbidden_proof_source(value):
                return True
    elif isinstance(proof, (tuple, list)):
        return any(_contains_forbidden_proof_source(value) for value in proof)
    return False


def admit_current_witness(witness: Mapping, inputs: ActualObservationInputs,
                          registry: WitnessRegistry = WitnessRegistry()):
    """Admit a known actual predicate only after actual-input recomputation.

    UNKNOWN produces no write, including when the existing belief is FALSE.
    The receiving selector must apply returned facts before hard gating.
    """
    try:
        _assert_actual_inputs(inputs)
    except ValueError as error:
        return WitnessAdmission(False, "actual_source_validation_failed", audit={"error": str(error)})
    fields = {"predicate", "value", "confidence", "evidence_kind", "source_role", "predecision",
              "snapshot", "snapshot_sha256", "extractor_version", "validation_report_sha256", "proof"}
    if set(witness) != fields:
        return WitnessAdmission(False, "witness_fields_not_exact")
    if witness["source_role"] != SOURCE_ROLE or witness["predecision"] is not True:
        return WitnessAdmission(False, "not_runtime_actual_PRE_evidence")
    if not isinstance(witness["proof"], Mapping) or _contains_forbidden_proof_source(witness["proof"]):
        return WitnessAdmission(False, "proof_contains_nonactual_or_forbidden_source")
    if witness["snapshot"] != asdict(inputs.snapshot) or witness["snapshot_sha256"] != inputs.snapshot.identity_sha256:
        return WitnessAdmission(False, "source_entity_time_hash_or_extractor_mismatch")
    predicate, kind = witness["predicate"], witness["evidence_kind"]
    if predicate not in KINDS or kind not in KINDS[predicate]:
        return WitnessAdmission(False, "evidence_kind_not_valid_for_predicate")
    extractor = next((e for e in registry.extractors if e.version == witness["extractor_version"]), None)
    if extractor is None:
        return WitnessAdmission(False, "extractor_not_independently_validated_proxy_only_or_unknown")
    if witness["validation_report_sha256"] != extractor.validation_report_sha256 or kind not in extractor.kinds.get(predicate, ()):
        return WitnessAdmission(False, "reviewed_extractor_contract_mismatch")
    if witness["value"] not in {"true", "false", "unknown"} or not isinstance(witness["confidence"], (int, float)) or isinstance(witness["confidence"], bool):
        return WitnessAdmission(False, "invalid_witness_value_or_confidence")
    if not np.isfinite(witness["confidence"]) or not 0 <= witness["confidence"] <= 1:
        return WitnessAdmission(False, "invalid_witness_value_or_confidence")
    try:
        recomputed = extractor.validator(MappingProxyType(dict(witness)), inputs)
    except (ValueError, TypeError, KeyError) as error:
        return WitnessAdmission(False, "actual_input_validation_failed", audit={"error": str(error)})
    if not isinstance(recomputed, RecomputedPredicate):
        return WitnessAdmission(False, "validator_did_not_return_recomputed_predicate")
    if recomputed.value != witness["value"] or abs(recomputed.confidence-witness["confidence"]) > 1e-12:
        return WitnessAdmission(False, "actual_recomputation_disagrees_with_claim")
    if recomputed.value == "unknown":
        return WitnessAdmission(False, "unknown_is_no_fact_write")
    if recomputed.value == "true":
        # Receipt approval validates the extractor, not every prediction's
        # certainty. Restore a missing prerequisite only at the original
        # predicate-specific hard threshold; low-confidence TRUE is no write.
        from .reranker import PREDICATE_HARD_THRESHOLDS
        minimum = PREDICATE_HARD_THRESHOLDS.get(predicate, .75)
        if recomputed.confidence < minimum:
            return WitnessAdmission(False, "positive_witness_below_original_predicate_threshold",
                audit={"predicate": predicate, "confidence": recomputed.confidence,
                       "required_confidence": minimum})
    fact = dict(predicate=predicate, value=recomputed.value, confidence=recomputed.confidence,
        block_index=inputs.snapshot.block_index, source="current_observation",
        evidence_id=recomputed.evidence_id, predecision=True)
    return WitnessAdmission(True, "independently_validated_actual_witness",
        MappingProxyType(fact), audit={"snapshot_sha256": inputs.snapshot.identity_sha256,
            "validation_report_sha256": extractor.validation_report_sha256,
            "extractor_version": extractor.version, "evidence_kind": kind})
