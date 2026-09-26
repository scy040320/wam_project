"""Auditable action-chunk semantics used before learned effect prediction."""

from __future__ import annotations

import numpy as np

from .contracts import CandidateEffect, Stage, TriValue


def parse_candidate_effect(candidate_id: int, actions: np.ndarray, *, visual_support: float = 0.5,
                           close_when_negative: bool = True) -> CandidateEffect:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (16, 7) or not np.isfinite(actions).all():
        raise ValueError("candidate actions must be finite with shape (16,7)")
    xyz = actions[:, :3]
    gripper = actions[:, 6]
    movement = float(np.linalg.norm(np.abs(xyz).sum(axis=0)))
    net_xyz = xyz.sum(axis=0)
    close_signal = -gripper if close_when_negative else gripper
    close_strength = float(np.max(close_signal))
    open_strength = float(np.max(-close_signal))
    closed_fraction = float(np.mean(close_signal > 0.25))
    lateral = float(np.linalg.norm(net_xyz[:2]))
    first_close = int(np.argmax(close_signal > 0.25)) if np.any(close_signal > 0.25) else 16
    upward_after_close = float(max(0.0, xyz[first_close:, 2].sum())) if first_close < 16 else 0.0
    if open_strength > 0.5 and closed_fraction > 0.2:
        stage = Stage.PLACE
    elif close_strength > 0.5 and upward_after_close > 0.08:
        stage = Stage.LIFT
    elif closed_fraction > 0.5 and lateral > 0.12:
        stage = Stage.TRANSPORT
    elif close_strength > 0.5:
        stage = Stage.GRASP
    elif movement > 0.08:
        stage = Stage.APPROACH
    else:
        stage = Stage.UNCERTAIN
    requirements = {
        Stage.APPROACH: {"target_visible": 0.5, "target_pose_current": 0.6},
        Stage.GRASP: {"target_pose_current": 0.8, "target_reachable": 0.8, "execution_consistent": 0.6},
        Stage.LIFT: {"grasped": 0.85, "execution_consistent": 0.8},
        Stage.TRANSPORT: {"grasped": 0.8, "lifted": 0.75},
        Stage.PLACE: {"lifted": 0.75, "receptacle_visible": 0.6, "place_ready": 0.7},
        Stage.UNCERTAIN: {}, Stage.OBSERVE: {},
    }
    effects = {
        Stage.APPROACH: {"target_reachable": (TriValue.TRUE, 0.55)},
        Stage.GRASP: {"grasped": (TriValue.TRUE, 0.55)},
        Stage.LIFT: {"lifted": (TriValue.TRUE, 0.55)},
        Stage.TRANSPORT: {"place_ready": (TriValue.TRUE, 0.45)},
        Stage.PLACE: {"placed": (TriValue.TRUE, 0.55)},
        Stage.UNCERTAIN: {}, Stage.OBSERVE: {},
    }
    confidence = float(np.clip(0.55 + 0.25 * abs(close_strength - open_strength) + 0.2 * visual_support, 0, 1))
    return CandidateEffect(candidate_id, stage, requirements[stage], effects[stage], confidence, {
        "movement_l1": movement, "net_upward": float(max(0.0, net_xyz[2])),
        "net_lateral": lateral, "close_strength": close_strength, "open_strength": open_strength,
        "closed_fraction": closed_fraction, "visual_support": float(visual_support),
    })
