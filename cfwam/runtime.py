"""Runtime guardrails for four-step, dependency-aware decisions."""

from __future__ import annotations

import math

from .recovery import RecoveryDecision, RecoveryRouter
from .types import AttributionResult, Cause


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
