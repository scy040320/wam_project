"""On-disk contract for aligned four-step counterfactual samples."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OnlineStepRecord:
    run_id: str
    task_id: str
    split: str
    phase: str
    episode_id: int
    t_query: int
    t_real: int
    requested_prefix: int
    executed_prefix: int
    primary_current: str
    wrist_current: str
    primary_prediction: str
    wrist_prediction: str
    primary_actual: str
    wrist_actual: str
    action_plan: str
    action_executed: str
    action_execution_delta: str
    proprio_current: str
    proprio_actual: str
    value_prediction: float
    belief_snapshot: dict[str, dict[str, object]]
    intervention_description: str


@dataclass(frozen=True)
class OfflineLabelRecord:
    run_id: str
    cause: str
    affected_nodes: list[str]
    intervention_parameters: dict[str, Any]
    label_source: str = "simulator_offline_only"


def write_aligned_record(directory: Path, online: OnlineStepRecord, offline: OfflineLabelRecord) -> None:
    """Keep online deployment inputs separate from simulator-derived supervision."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "online_record.json").write_text(
        json.dumps(asdict(online), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / "offline_label.json").write_text(
        json.dumps(asdict(offline), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
