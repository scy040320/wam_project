"""Runtime guardrails for four-step, dependency-aware decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .recovery import RecoveryDecision, RecoveryRouter
from .types import AttributionResult, Cause


class GuardLevel(str, Enum):
    CLEAR = "clear"
    WEAK = "weak"
    STRONG = "strong"


@dataclass(frozen=True)
class TemporalGuardResult:
    level: GuardLevel
    weak_streak: int
    guarded_reobserve: bool
    force_safe_stop: bool
    unknown_probability: float


class TemporalAbstainGate:
    """Episode-local hysteresis for online abstention; it never reads simulator GT."""

    def __init__(self, strong_unknown_probability: float, weak_confirmations: int = 2):
        if not 0.0 <= strong_unknown_probability <= 1.0:
            raise ValueError("strong_unknown_probability must be in [0, 1]")
        if weak_confirmations < 2:
            raise ValueError("weak_confirmations must be at least two")
        self.strong_unknown_probability = strong_unknown_probability
        self.weak_confirmations = weak_confirmations
        self.reset_episode()

    def reset_episode(self) -> None:
        self.weak_streak = 0

    def update(
        self,
        attribution: AttributionResult,
        immediate_safe_stop: bool = False,
    ) -> TemporalGuardResult:
        unknown_probability = float(attribution.cause_probabilities.get(Cause.UNKNOWN, 0.0))
        if immediate_safe_stop:
            self.weak_streak = self.weak_confirmations
            return TemporalGuardResult(
                GuardLevel.STRONG, self.weak_streak, False, True,
                unknown_probability,
            )
        if unknown_probability >= self.strong_unknown_probability:
            # Validation trajectories show that isolated high-confidence
            # unknown spikes occur in clean/known episodes.  Treat the first
            # spike as a guarded re-observation and stop only after temporal
            # confirmation, preserving the explicit safe exit without making
            # a single model spike terminal.
            self.weak_streak += 1
            confirmed = self.weak_streak >= self.weak_confirmations
            return TemporalGuardResult(
                GuardLevel.STRONG, self.weak_streak, not confirmed, confirmed,
                unknown_probability,
            )
        if attribution.abstained or attribution.cause is Cause.UNKNOWN:
            self.weak_streak += 1
            confirmed = self.weak_streak >= self.weak_confirmations
            return TemporalGuardResult(
                GuardLevel.WEAK, self.weak_streak, not confirmed, confirmed, unknown_probability,
            )
        self.weak_streak = 0
        return TemporalGuardResult(GuardLevel.CLEAR, 0, False, False, unknown_probability)


class AbstainPolicy:
    """Frozen validation thresholds for the explicit unknown safety exit."""

    def __init__(self, min_known_probability: float, max_entropy: float, max_residual_energy: float):
        self.min_known_probability = min_known_probability
        self.max_entropy = max_entropy
        self.max_residual_energy = max_residual_energy

    def apply(self, cause_probabilities: dict[Cause, float], predicted_cause: Cause, residual_energy: float) -> AttributionResult:
        known = {cause: value for cause, value in cause_probabilities.items() if cause is not Cause.UNKNOWN}
        maximum = max(known.values(), default=0.0)
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in cause_probabilities.values())
        abstained = (
            predicted_cause is Cause.UNKNOWN
            or maximum < self.min_known_probability
            or entropy > self.max_entropy
            or residual_energy > self.max_residual_energy
        )
        cause = Cause.UNKNOWN if abstained else predicted_cause
        return AttributionResult(cause, cause_probabilities, frozenset(), residual_energy, entropy, abstained)


@dataclass(frozen=True)
class CausalConsistencyResult:
    attribution: AttributionResult
    reason: str


class CausalConsistencyGuard:
    """Validation-frozen checks for observable execution/temporal evidence.

    These rules never read intervention metadata or simulator truth.  They use
    only the planned-versus-executed action residual and the change in residual
    energy between adjacent online decisions.
    """

    def __init__(self, action_residual_l2_threshold: float, object_energy_delta_threshold: float):
        if action_residual_l2_threshold < 0.0:
            raise ValueError("action_residual_l2_threshold must be non-negative")
        self.action_residual_l2_threshold = action_residual_l2_threshold
        self.object_energy_delta_threshold = object_energy_delta_threshold

    @staticmethod
    def _replace(
        attribution: AttributionResult,
        cause: Cause,
        affected_nodes: set[str],
        abstained: bool = False,
    ) -> AttributionResult:
        return AttributionResult(
            cause,
            attribution.cause_probabilities,
            frozenset(affected_nodes),
            attribution.residual_energy,
            attribution.entropy,
            abstained,
        )

    def apply(
        self,
        attribution: AttributionResult,
        action_residual_l2: float,
        energy_delta: float,
        object_nodes: set[str],
    ) -> CausalConsistencyResult:
        has_action_error = action_residual_l2 > self.action_residual_l2_threshold
        # Validation shows that model-level unknown plus an independently
        # observed execution deviation is specific to the mixed/OOD condition.
        # Preserve the unknown label but expose the evidence provenance so the
        # temporal gate can take the explicit safety exit immediately.
        if attribution.cause is Cause.UNKNOWN and has_action_error:
            nodes = set(attribution.affected_nodes) | {"action_segment_4", "effector_gripper"}
            return CausalConsistencyResult(
                self._replace(attribution, Cause.UNKNOWN, nodes, abstained=True),
                "model_unknown_with_execution_evidence",
            )
        if attribution.cause is Cause.UNKNOWN:
            return CausalConsistencyResult(attribution, "model_unknown")
        if has_action_error and attribution.cause is Cause.VISUAL_OCCLUSION:
            nodes = set(attribution.affected_nodes) | {
                "camera_primary", "action_segment_4", "effector_gripper",
            }
            return CausalConsistencyResult(
                self._replace(attribution, Cause.UNKNOWN, nodes, abstained=True),
                "simultaneous_visual_and_execution_evidence",
            )
        if has_action_error:
            nodes = set(attribution.affected_nodes) | {"action_segment_4", "effector_gripper"}
            return CausalConsistencyResult(
                self._replace(attribution, Cause.ACTION_NOISE, nodes),
                "observed_action_execution_delta",
            )
        if (
            attribution.cause in {Cause.NORMAL, Cause.ACTION_NOISE}
            and energy_delta > self.object_energy_delta_threshold
        ):
            return CausalConsistencyResult(
                self._replace(attribution, Cause.OBJECT_SHIFT, set(object_nodes)),
                "temporal_residual_energy_jump_without_action_delta",
            )
        return CausalConsistencyResult(attribution, "model_attribution")


class OnlineDecisionLoop:
    """Routes an already-computed model result without accessing simulator GT."""

    def __init__(self, router: RecoveryRouter, abstain_policy: AbstainPolicy):
        self.router = router
        self.abstain_policy = abstain_policy

    def decide(
        self,
        probabilities: dict[Cause, float],
        predicted_cause: Cause,
        affected_nodes: set[str],
        residual_energy: float,
        timestamp: int,
    ) -> tuple[AttributionResult, RecoveryDecision]:
        attribution = self.abstain_policy.apply(probabilities, predicted_cause, residual_energy)
        attribution = AttributionResult(
            attribution.cause, attribution.cause_probabilities, frozenset(affected_nodes),
            attribution.residual_energy, attribution.entropy, attribution.abstained,
        )
        return attribution, self.router.route(attribution, timestamp)
