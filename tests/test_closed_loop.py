import numpy as np

from wam_reranking import (
    AttributionOutput, BestSeedSelection, ClosedLoopCommand, CoarseCause,
    ConsistencyFactor, EvidenceQuality, RecoveryBudget, RecoveryState,
    SelectionResult, SelectorMode, TriValue, apply_recovery_budget,
    initial_belief, select_best_seed,
)


def attribution(cause=CoarseCause.NORMAL):
    classes = {item.value: 0.0 for item in CoarseCause}
    classes[cause.value] = 1.0
    states = {item.value: TriValue.TRUE for item in ConsistencyFactor}
    probabilities = {item.value: 0.9 for item in ConsistencyFactor}
    return AttributionOutput(
        factor_probs=probabilities,
        class_probs=classes,
        projected_cause=cause,
        confidence=0.9,
        entropy=0.0,
        evidence_quality=EvidenceQuality(True, True, True),
        source_block_id="block-1",
        factor_states=states,
        factor_confidences=probabilities,
    )


def fallback_selection(name):
    result = SelectionResult(SelectorMode.LEARNED_HARD_GATE, None, name, (), {})
    return BestSeedSelection(None, result)


def test_best_seed_adapter_maps_candidate_index_to_seed():
    actions = [np.zeros((16, 7)), np.zeros((16, 7))]
    result = select_best_seed(
        seed_ids=[101, 205],
        mode=SelectorMode.LEARNED_HARD_GATE,
        belief=initial_belief(0),
        attribution=attribution(),
        block_index=1,
        candidate_actions=actions,
        official_values=[0.1, 0.9],
    )
    assert result.selected_seed == 205
    assert result.selection.selected_candidate_id == 1


def test_best_seed_rejects_duplicate_seed_ids():
    actions = [np.zeros((16, 7)), np.zeros((16, 7))]
    try:
        select_best_seed(
            seed_ids=[7, 7], mode=SelectorMode.LEARNED_HARD_GATE,
            belief=initial_belief(0), attribution=attribution(), block_index=1,
            candidate_actions=actions, official_values=[0.1, 0.2],
        )
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate query seeds must be rejected")


def test_execute_does_not_consume_fallback_budget():
    selection = BestSeedSelection(
        42, SelectionResult(SelectorMode.LEARNED_HARD_GATE, 0, None, (), {})
    )
    state = RecoveryState()
    decision = apply_recovery_budget(selection, state, RecoveryBudget())
    assert decision.command is ClosedLoopCommand.EXECUTE
    assert decision.selected_seed == 42
    assert state.requeries == state.reobserves == state.safe_rejects == 0


def test_requery_budget_is_finite():
    state = RecoveryState()
    budget = RecoveryBudget(max_requeries=1, max_decisions=4)
    first = apply_recovery_budget(fallback_selection("requery"), state, budget)
    second = apply_recovery_budget(fallback_selection("requery"), state, budget)
    assert first.command is ClosedLoopCommand.REQUERY
    assert second.command is ClosedLoopCommand.SAFE_STOP
    assert state.terminal


def test_reobserve_budget_is_finite():
    state = RecoveryState()
    budget = RecoveryBudget(max_reobserves=1, max_decisions=4)
    assert apply_recovery_budget(fallback_selection("reobserve"), state, budget).command is ClosedLoopCommand.REOBSERVE
    assert apply_recovery_budget(fallback_selection("reobserve"), state, budget).command is ClosedLoopCommand.SAFE_STOP


def test_safe_reject_budget_is_finite():
    state = RecoveryState()
    budget = RecoveryBudget(max_safe_rejects=1, max_decisions=4)
    assert apply_recovery_budget(fallback_selection("safe_reject"), state, budget).command is ClosedLoopCommand.SAFE_REJECT
    assert apply_recovery_budget(fallback_selection("safe_reject"), state, budget).command is ClosedLoopCommand.SAFE_STOP


def test_global_decision_budget_stops_mixed_retry_sequence():
    state = RecoveryState()
    budget = RecoveryBudget(max_requeries=3, max_reobserves=3, max_decisions=2)
    assert apply_recovery_budget(fallback_selection("requery"), state, budget).command is ClosedLoopCommand.REQUERY
    assert apply_recovery_budget(fallback_selection("reobserve"), state, budget).command is ClosedLoopCommand.REOBSERVE
    assert apply_recovery_budget(fallback_selection("requery"), state, budget).command is ClosedLoopCommand.SAFE_STOP


def test_terminal_controller_stays_stopped():
    state = RecoveryState(terminal=True)
    decision = apply_recovery_budget(fallback_selection("requery"), state, RecoveryBudget())
    assert decision.command is ClosedLoopCommand.SAFE_STOP
    assert decision.reason == "controller_already_terminal"


def test_missing_fallback_stops_safely():
    state = RecoveryState()
    decision = apply_recovery_budget(fallback_selection(None), state, RecoveryBudget())
    assert decision.command is ClosedLoopCommand.SAFE_STOP
    assert decision.reason == "missing_budget_exhausted"


def test_budget_audit_records_before_and_after_counts():
    state = RecoveryState()
    decision = apply_recovery_budget(fallback_selection("requery"), state, RecoveryBudget())
    assert decision.budget_before["requeries"] == 0
    assert decision.budget_after["requeries"] == 1
    assert decision.budget_after["decisions"] == 1
