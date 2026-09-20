"""Pre-registered development split and offline-only labels."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Cause


CONDITIONS = (
    Cause.NORMAL, Cause.VISUAL_OCCLUSION, Cause.OBJECT_SHIFT,
    Cause.ACTION_NOISE, Cause.UNKNOWN,
)
PHASES = ("phase_approach", "phase_grasp", "phase_transport")


@dataclass(frozen=True)
class DevelopmentSplit:
    train: tuple[int, ...] = tuple(range(0, 24))
    validation: tuple[int, ...] = tuple(range(24, 32))
    test: tuple[int, ...] = tuple(range(32, 40))

    def subset(self, episode_id: int) -> str:
        if episode_id in self.train:
            return "train"
        if episode_id in self.validation:
            return "validation"
        if episode_id in self.test:
            return "test"
        raise ValueError(f"episode_id must be within the frozen range [0, 39], got {episode_id}")


def expected_short_trajectory_count(task_count: int = 4) -> int:
    return task_count * len(PHASES) * len(CONDITIONS) * 40


def offline_mask_for_cause(cause: Cause, object_node: str | None = None, target_node: str | None = None) -> list[str]:
    """Ground-truth mask used offline only; never pass this to online inference."""
    if cause is Cause.NORMAL:
        return []
    if cause is Cause.VISUAL_OCCLUSION:
        return ["camera_primary"]
    if cause is Cause.OBJECT_SHIFT:
        return [node for node in (object_node, target_node) if node]
    if cause is Cause.ACTION_NOISE:
        return ["action_segment_4", "effector_gripper"]
    return ["camera_primary", *(node for node in (object_node, target_node) if node), "action_segment_4"]
