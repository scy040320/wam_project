"""Public interfaces for belief-constrained WAM candidate reranking."""

from .belief import DEFAULT_GRAPH, DEFAULT_TASK_BINDINGS, DependencyGraph, initial_belief, update_belief
from .candidate_effects import parse_candidate_effect
from .calibration import HeadThreshold, build_attribution_output
from .contrastive import cross_task_factor_contrastive_loss
from .contracts import (
    ActionRefinementRequest, ActionRefinementResult, AttributionOutput, BeliefFact,
    BeliefState, CandidateDecision, CandidateEffect, CoarseCause, ConsistencyFactor,
    EvidenceQuality, ScoreWeights, Stage, TaskBinding, TriValue,
)
from .reranker import evaluate_candidate, select_candidate
from .target_localization import (
    CLIPSegTargetLocalizer, TargetResidualFeatures, canonical_target_prompt,
    pool_target_residual, structural_residual_grid, target_semantic_features,
)
from .paired_audit import PairedAuditDecision, PairedAuditThresholds, evaluate_clean_pair
from .evidence_routing import (
    CommandExecutionEvidence, EvidenceRoute, EvidenceRoutingDecision,
    LEARNED_FACTOR_NAMES, evaluate_command_execution, route_evidence,
)

__all__ = [
    "ActionRefinementRequest", "ActionRefinementResult", "AttributionOutput",
    "BeliefFact", "BeliefState", "CandidateDecision", "CandidateEffect",
    "CoarseCause", "ConsistencyFactor", "DEFAULT_GRAPH", "DEFAULT_TASK_BINDINGS",
    "DependencyGraph", "EvidenceQuality", "ScoreWeights", "Stage", "TaskBinding",
    "TriValue", "HeadThreshold", "build_attribution_output", "evaluate_candidate", "initial_belief", "parse_candidate_effect",
    "select_candidate", "update_belief", "PairedAuditDecision",
    "PairedAuditThresholds", "evaluate_clean_pair",
    "CommandExecutionEvidence", "EvidenceRoute", "EvidenceRoutingDecision",
    "LEARNED_FACTOR_NAMES", "evaluate_command_execution", "route_evidence",
    "CLIPSegTargetLocalizer", "TargetResidualFeatures", "canonical_target_prompt",
    "pool_target_residual", "structural_residual_grid", "target_semantic_features",
    "cross_task_factor_contrastive_loss",
]
