"""Auditable best-seed integration and bounded recovery control.

This module does not change the frozen Cosmos generator.  It connects the
existing attribution-conditioned selector to Cosmos' ``best_seed`` decision
point and makes every fallback explicit and finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from .candidate_utility import CandidateUtilityModel
from .contracts import (
    AttributionOutput, BeliefState, CandidateVisualEvidence, ScoreWeights,
)
from .policy import SelectorMode, SelectionResult, select_hard_gate_value_tiebreak


class ClosedLoopCommand(str, Enum):
    EXECUTE = "execute"
    REOBSERVE = "reobserve"
    REQUERY = "requery"
    SAFE_REJECT = "safe_reject"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class RecoveryBudget:
    max_requeries: int = 1
    max_reobserves: int = 1
    max_safe_rejects: int = 1
    max_decisions: int = 4

    def __post_init__(self) -> None:
        values = (
            self.max_requeries, self.max_reobserves,
            self.max_safe_rejects, self.max_decisions,
        )
        if any(value < 0 for value in values):
            raise ValueError("recovery budgets must be non-negative")
        if self.max_decisions < 1:
            raise ValueError("max_decisions must be at least one")


@dataclass
class RecoveryState:
    decisions: int = 0
    requeries: int = 0
    reobserves: int = 0
    safe_rejects: int = 0
    terminal: bool = False


@dataclass(frozen=True)
class BestSeedSelection:
    selected_seed: int | None
    selection: SelectionResult


@dataclass(frozen=True)
class ClosedLoopDecision:
    command: ClosedLoopCommand
    selected_seed: int | None
    reason: str
    budget_before: Mapping[str, int | bool]
    budget_after: Mapping[str, int | bool]


def _state_record(state: RecoveryState) -> dict[str, int | bool]:
    return {
        "decisions": state.decisions,
        "requeries": state.requeries,
        "reobserves": state.reobserves,
        "safe_rejects": state.safe_rejects,
        "terminal": state.terminal,
    }


def select_best_seed(
    *,
    seed_ids: Sequence[int],
    mode: SelectorMode,
    belief: BeliefState,
    attribution: AttributionOutput,
    block_index: int,
    candidate_actions: Sequence[np.ndarray],
    official_values: Sequence[float],
    visual_support: Sequence[float] | None = None,
    visual_evidence: Sequence[CandidateVisualEvidence | None] | None = None,
    hard_confidence: float = 0.75,
    score_weights: ScoreWeights | None = None,
    utility_model: CandidateUtilityModel | None = None,
) -> BestSeedSelection:
    """Select a real Cosmos query seed through the frozen reranking policy."""
    if len(seed_ids) != len(candidate_actions):
        raise ValueError("seed_ids must align one-to-one with candidate actions")
    if len(set(seed_ids)) != len(seed_ids):
        raise ValueError("seed_ids must be unique")
    selection = select_hard_gate_value_tiebreak(
        mode=mode,
        belief=belief,
        attribution=attribution,
        block_index=block_index,
        candidate_actions=candidate_actions,
        official_values=official_values,
        visual_support=visual_support,
        visual_evidence=visual_evidence,
        hard_confidence=hard_confidence,
        score_weights=score_weights,
        utility_model=utility_model,
    )
    selected_seed = (
        None if selection.selected_candidate_id is None
        else int(seed_ids[selection.selected_candidate_id])
    )
    return BestSeedSelection(selected_seed, selection)


def apply_recovery_budget(
    selection: BestSeedSelection,
    state: RecoveryState,
    budget: RecoveryBudget,
) -> ClosedLoopDecision:
    """Convert one selector result into a bounded, auditable command."""
    before = _state_record(state)
    if state.terminal:
        return ClosedLoopDecision(
            ClosedLoopCommand.SAFE_STOP, None, "controller_already_terminal", before, before
        )
    if state.decisions >= budget.max_decisions:
        state.terminal = True
        after = _state_record(state)
        return ClosedLoopDecision(
            ClosedLoopCommand.SAFE_STOP, None, "decision_budget_exhausted", before, after
        )

    state.decisions += 1
    if selection.selected_seed is not None:
        after = _state_record(state)
        return ClosedLoopDecision(
            ClosedLoopCommand.EXECUTE,
            selection.selected_seed,
            "candidate_selected",
            before,
            after,
        )

    fallback = selection.selection.fallback
    if fallback == "requery" and state.requeries < budget.max_requeries:
        state.requeries += 1
        command, reason = ClosedLoopCommand.REQUERY, "selector_requery"
    elif fallback == "reobserve" and state.reobserves < budget.max_reobserves:
        state.reobserves += 1
        command, reason = ClosedLoopCommand.REOBSERVE, "selector_reobserve"
    elif fallback == "safe_reject" and state.safe_rejects < budget.max_safe_rejects:
        state.safe_rejects += 1
        command, reason = ClosedLoopCommand.SAFE_REJECT, "selector_safe_reject"
    else:
        state.terminal = True
        command = ClosedLoopCommand.SAFE_STOP
        reason = f"{fallback or 'missing'}_budget_exhausted"
    after = _state_record(state)
    return ClosedLoopDecision(command, None, reason, before, after)
