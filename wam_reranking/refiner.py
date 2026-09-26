"""Safety helpers for the planned external action-residual model."""

import numpy as np

from .contracts import ActionRefinementRequest, ActionRefinementResult


def apply_bounded_residual(request: ActionRefinementRequest, result: ActionRefinementResult) -> np.ndarray:
    """Apply an already-predicted bounded residual; this function does not train a model."""
    return np.asarray(request.candidate_actions, dtype=np.float64) + np.asarray(result.residual, dtype=np.float64)
