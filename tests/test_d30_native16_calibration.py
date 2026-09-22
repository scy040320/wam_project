import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_d30_native16_temporal_guard import load_rows, main


def _write_run(root: Path, condition: str, probabilities: dict[str, float]) -> None:
    run = root / f"task0_seed24_D_dependency_aware_{condition}"
    run.mkdir(parents=True)
    payload = {
        "control_horizon": 16,
        "condition": condition,
        "decisions": [{
            "block": 2,
            "intervention_injected": True,
            "raw_cause": condition,
            "cause_probabilities": probabilities,
            "entropy": 0.5,
            "residual_energy": 0.8,
        }],
    }
    (run / "online_decision_log.json").write_text(json.dumps(payload), encoding="utf-8")
    query = run / "query_002"
    query.mkdir()
    delta = np.zeros((16, 7), dtype=np.float32)
    if condition == "unknown":
        delta[:, :3] = 0.25
    np.save(query / "action_execution_delta.npy", delta)


def test_native16_calibration_freezes_validation_only_manifest(tmp_path, monkeypatch):
    source = tmp_path / "validation"
    common = {
        "normal": 0.8,
        "visual_occlusion": 0.05,
        "object_shift": 0.05,
        "action_noise": 0.05,
        "unknown": 0.05,
    }
    _write_run(source, "normal", common)
    _write_run(source, "unknown", {
        "normal": 0.01,
        "visual_occlusion": 0.01,
        "object_shift": 0.01,
        "action_noise": 0.01,
        "unknown": 0.96,
    })
    rows = load_rows(source)
    assert sum(row["true_unknown"] for row in rows) == 1

    output = tmp_path / "frozen"
    monkeypatch.setattr(sys, "argv", ["calibrate", "--input", str(source), "--output", str(output)])
    assert main() == 0
    manifest = (output / "d30_native16_temporal_guard_frozen.yaml").read_text(encoding="utf-8")
    assert "protocol: native16_prediction_real_alignment" in manifest
    assert "control_horizon: 16" in manifest
    assert "model_unknown_with_execution_evidence" in manifest


def test_native16_calibration_rejects_four_step_logs(tmp_path):
    run = tmp_path / "task0_seed24_D_dependency_aware_unknown"
    run.mkdir(parents=True)
    (run / "online_decision_log.json").write_text(json.dumps({
        "control_horizon": 4,
        "condition": "unknown",
        "decisions": [],
    }), encoding="utf-8")
    try:
        load_rows(tmp_path)
    except ValueError as error:
        assert "non-native horizon" in str(error)
    else:
        raise AssertionError("four-step calibration log was accepted")
