"""Simulator-independent audit contract for paired counterfactual branches.

The physical gate compares clean-A and clean-B after both branches undergo the
same snapshot-restore procedure.  A cached query observation compared with a
newly rendered observation is retained as diagnostic evidence only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PairedAuditThresholds:
    sim_qpos_max_abs: float = 1e-3
    sim_qpos_p95_abs: float = 2e-4
    robot_gripper_qvel_max_abs: float = 1e-3
    target_qvel_max_abs: float = 1e-3
    other_qvel_max_abs: float = 2e-2
    other_qvel_p95_abs: float = 5e-3
    image_mean_abs: float = 2.0
    image_p95_abs: float = 8.0
    image_fraction_gt5: float = 0.06
    image_psnr_min_db: float = 30.0
    proprio_max_abs: float = 1e-3


@dataclass(frozen=True)
class PairedAuditDecision:
    passed: bool
    failures: tuple[str, ...]
    cached_observation_diagnostics: Mapping[str, float]


def _image_pass(prefix: str, metrics: Mapping[str, float], t: PairedAuditThresholds,
                failures: list[str]) -> None:
    checks = (
        ("mean_abs", float(metrics["mean_abs"]) <= t.image_mean_abs),
        ("p95_abs", float(metrics["p95_abs"]) <= t.image_p95_abs),
        ("fraction_gt5", float(metrics["fraction_gt5"]) <= t.image_fraction_gt5),
        ("psnr_db", float(metrics["psnr_db"]) >= t.image_psnr_min_db),
    )
    failures.extend(f"{prefix}.{name}" for name, ok in checks if not ok)


def evaluate_clean_pair(
    clean_pair: Mapping[str, object],
    *,
    cached_observation_diagnostics: Mapping[str, float] | None = None,
    thresholds: PairedAuditThresholds = PairedAuditThresholds(),
) -> PairedAuditDecision:
    """Evaluate only clean-A/clean-B physical equivalence as hard gates.

    ``cached_observation_diagnostics`` may contain query-cache versus forced
    re-observation differences.  They are returned unchanged and never decide
    pass/fail because they do not compare two counterfactual branches.
    """

    failures: list[str] = []
    sim = clean_pair["sim_state"]
    scalar_checks = (
        ("sim_state.qpos_max_abs", float(sim["qpos_max_abs"]) <= thresholds.sim_qpos_max_abs),
        ("sim_state.qpos_p95_abs", float(sim["qpos_p95_abs"]) <= thresholds.sim_qpos_p95_abs),
        ("sim_state.robot_gripper_qvel_max_abs", float(sim["robot_gripper_qvel_max_abs"]) <= thresholds.robot_gripper_qvel_max_abs),
        ("sim_state.target_qvel_max_abs", float(sim["target_qvel_max_abs"]) <= thresholds.target_qvel_max_abs),
        ("sim_state.other_qvel_max_abs", float(sim["other_qvel_max_abs"]) <= thresholds.other_qvel_max_abs),
        ("sim_state.other_qvel_p95_abs", float(sim["other_qvel_p95_abs"]) <= thresholds.other_qvel_p95_abs),
        ("proprio_max_abs", float(clean_pair["proprio_max_abs"]) <= thresholds.proprio_max_abs),
    )
    failures.extend(name for name, ok in scalar_checks if not ok)
    _image_pass("primary", clean_pair["primary"], thresholds, failures)
    _image_pass("wrist", clean_pair["wrist"], thresholds, failures)
    return PairedAuditDecision(
        passed=not failures,
        failures=tuple(failures),
        cached_observation_diagnostics=dict(cached_observation_diagnostics or {}),
    )
