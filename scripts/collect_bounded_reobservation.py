"""Finite, constructed wrist-camera probes on two consumed development states.

This is 2 starts x 4 planned arms, each captured and re-executed twice for
immutable-image and repeat QC (24 physical executions). It is NOT a native
Cosmos candidate pool, independent test, policy improvement, or training run.
All branches keep the same persistent primary-view corruption. It is never
silently removed to manufacture recovery. There are no model/policy queries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

SOURCE = "known_task_recovery_v9_20261006_v5_paired_recovery"
ADAPTER = "known_task_recovery_observation_audit_20261006_v5_motion_typed_support"
REFERENCE = "candidate_reranking_d21_main20_stage_qualification_20261003_v2_split_compat"
BASE = "known_task_temporal_recovery_supervision_20261006_v2_scoped_segmentation"
ARMS = ("hold", "peek_left", "peek_right", "peek_up")
DEPENDENCIES = {
    f"research_runs/{ADAPTER}/collect_recovery_observation_evidence.py": "957e14e0e60bb4e2cae3db99311e2f2b4ee59d31b4fd49951b33c3f047a9d99e",
    f"research_runs/{ADAPTER}/observation_geometry.py": "f2eeca8e17d43f5939889a635a04252e4ee78bcd15632e9179a843e6843c7991",
    f"research_runs/{ADAPTER}/build_observable_recovery_labels.py": "7104ffa1faac715c63fc955d773896dac74efbcb836883bd356022334ca0c211",
    f"research_runs/{BASE}/replay_temporal_recovery_evidence.py": "3684243f669a27430b27f45673c138859548e160dff757de7269eb25128e8069",
    f"research_runs/{BASE}/temporal_recovery_teacher.py": "1c85d2b4667594d7a419501950bdcebbec134b6bc9fa2f4ea05ba476834d5053",
    f"research_runs/{REFERENCE}/collect_snapshot_fork.py": "e8588c5ba3718d6a9d1b1f64832bb66d9a835c3b935fd0ee8a450ac90e9c9070",
    f"research_runs/{REFERENCE}/base_collector_shift_calibrated.py": "681bea34e08b25ac4928c334b593d93921e44ef1067d985ee529998e3695a7c9",
}
SNAPSHOT_SHA = {10: "1469c584015586ed875f8d0647f991ee6c7568381fe177beb8e8273b3253df59",
                13: "d0aad671a4423882f2649dbdd5e64e32f1d18c7b639b04f1584356b175029176"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""): h.update(part)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def planned_probe(arm, original_planned):
    """Only frozen, pre-execution command channel defines gripper state."""
    if arm not in ARMS:
        raise ValueError("Probe arm is not predeclared")
    source = np.asarray(original_planned)
    if source.shape != (16, 7) or not np.isfinite(source).all():
        raise ValueError("Frozen finite 16x7 planned commands required")
    actions = np.zeros((16, 7), dtype=np.float32)
    # Cosmos commands may slightly exceed [-1,1]. The new constructed AUX
    # controller command is explicitly saturated; frozen source bytes stay
    # intact. Do not turn this engineering input contract into a pixel gate.
    actions[:, 6] = np.clip(source[0, 6], -1., 1.)
    # First two steps have no end-effector motion, providing a real causal
    # reference instead of backfilling the cached query image.
    if arm == "peek_left": actions[2:10, 0] = -.2
    if arm == "peek_right": actions[2:10, 0] = .2
    if arm == "peek_up": actions[2:10, 2] = .2
    return actions


def freeze(root):
    source = root / "outputs" / SOURCE
    report = json.loads((source / "completion_audit.json").read_text())
    if report.get("passed") is not True or report.get("candidate_first_blocks") != 160:
        raise ValueError("Original development source audit required")
    touched = {str((source / "completion_audit.json").relative_to(root)): sha(source / "completion_audit.json")}
    entries = []
    for state, split in ((10, "train"), (13, "val")):
        folder = source / "pools" / f"task57_state{state}_unknown"
        paths = [folder / n for n in ("snapshot.pkl", "primary_before.npy", "wrist_before.npy",
                                      "proprio_before.npy", "candidate_0/planned_actions.npy", "pool.json")]
        for path in paths: touched[str(path.relative_to(root))] = sha(path)
        if sha(folder / "snapshot.pkl") != SNAPSHOT_SHA[state]:
            raise ValueError("Trusted project snapshot changed")
        meta = json.loads((folder / "pool.json").read_text())
        if (meta["task"], meta["state"], meta["split"], meta["cause"]) != (57, state, split, "unknown"):
            raise ValueError("Source-scoped group/split mismatch")
        original = np.load(folder / "candidate_0/planned_actions.npy", allow_pickle=False)
        plans = {arm: planned_probe(arm, original).tolist() for arm in ARMS}
        entries.append(dict(task=57, state=state, split=split, source_directory=str(folder.relative_to(root)),
                            pool_id=f"task57_state{state}_unknown", planned_arms=plans))
    return dict(schema="bounded_reobservation_probe_contract_v1", source=SOURCE,
                role="constructed_auxiliary_not_native_cosmos_candidates", entries=entries,
                model_queries=0, raw_physical_executions=24, captured_probe_arms=8,
                repeated_qc_arms=16, persistent_primary_view_conflict=True,
                auxiliary_gripper_command="clip(source_preexecution_first_gripper,-1,1)",
                supervision_targets=["target_evidence_available"],
                no_cross_view_conflict_resolution_claim=True,
                policy_steps_per_execution=16, reference_step=2,
                keep_all_no_recovery_and_terminal_results=True,
                gt_is_supervision_only=True, no_future_execution_x=True,
                original_rows_and_models_untouched=True, training_started=False,
                ranking_training_ready=False, source_hashes=touched,
                dependencies=DEPENDENCIES, probe_script_sha256=sha(Path(__file__)))


def record_probe(env, fork, intervention, snapshot, folder, source_folder, actions):
    env.reset(); raw = fork.restore_runtime_snapshot(env, copy.deepcopy(snapshot))
    if not np.array_equal(np.asarray(env.get_sim_state()), snapshot["sim_state"]):
        raise RuntimeError("Exact common starting state required")
    images = {v: [np.load(source_folder / (v + "_before.npy"), allow_pickle=False)]
              for v in ("primary", "wrist")}
    proprio = [np.load(source_folder / "proprio_before.npy", allow_pickle=False)]
    length = 0
    for action in actions:
        raw, _, done, _ = env.step(action.tolist())
        observed, _ = intervention.InterventionEnv._shift_primary_view(copy.deepcopy(raw))
        images["primary"].append(np.flipud(observed["agentview_image"]))
        images["wrist"].append(np.flipud(observed["robot0_eye_in_hand_image"]))
        proprio.append(np.r_[raw["robot0_gripper_qpos"], raw["robot0_eef_pos"], raw["robot0_eef_quat"]])
        length += 1
        if done: break
    folder.mkdir(parents=True, exist_ok=False)
    for view, values in images.items(): np.save(folder / (view + ".npy"), np.stack(values))
    for name in ("requested", "applied", "planned_actions"): np.save(folder / (name + ".npy"), actions[:length])
    np.save(folder / "proprio.npy", np.stack(proprio))
    np.save(folder / "exact_start_state.npy", snapshot["sim_state"])
    dump(folder / "execution_metadata.json", dict(valid_length=length, terminal_seen=bool(done),
         terminal_success_not_used_as_recovery=True, no_padding=True))
    return length


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--collect", action="store_true")
    args = parser.parse_args(); root = args.root.resolve(); out = args.output.resolve()
    if out.exists() or root not in out.parents or root / "outputs" / SOURCE in out.parents:
        raise ValueError("New in-project output required; no overwrite/resume")
    protocol = freeze(root)
    for relative, digest in DEPENDENCIES.items():
        if sha(root / relative) != digest: raise ValueError("Frozen dependency changed: " + relative)
    out.mkdir(parents=True); dump(out / "protocol.json", protocol)
    dump(out / "status.json", dict(phase="contract_frozen", training_started=False, captured_arms=0))
    if not args.collect: return
    started = time.time(); completed = []; env = None
    try:
        # Every package/code origin is explicit. No root shadowed imports.
        adapter_dir = root / "research_runs" / ADAPTER
        sys.path.insert(0, str(adapter_dir))
        geometry = load("observation_geometry", adapter_dir / "observation_geometry.py")
        teacher = load("temporal_recovery_teacher", root / "research_runs" / BASE / "temporal_recovery_teacher.py")
        base = load("bounded_reobserve_base", root / "research_runs" / BASE / "replay_temporal_recovery_evidence.py")
        adapter = load("bounded_reobserve_adapter", adapter_dir / "collect_recovery_observation_evidence.py")
        certificate_helpers = load("bounded_reobserve_certificate_helpers", adapter_dir / "build_observable_recovery_labels.py")
        fork = load("bounded_reobserve_fork", root / "research_runs" / REFERENCE / "collect_snapshot_fork.py")
        intervention = load("bounded_reobserve_intervention", root / "research_runs" / REFERENCE / "base_collector_shift_calibrated.py")
        labeler = load("bounded_reobserve_labels", Path(__file__).parent / "reobservation_supervision.py")
        adapter.attach(base, root)
        from libero.libero import benchmark
        from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env
        task = benchmark.get_benchmark_dict()["libero_90"]().get_task(57)
        env, _ = get_libero_env(task, "cosmos", resolution=256)
        for entry in protocol["entries"]:
            folder = root / entry["source_directory"]
            # Only hash-verified project-generated pickles, never arbitrary user data.
            if sha(folder / "snapshot.pkl") != SNAPSHOT_SHA[entry["state"]]:
                raise RuntimeError("Snapshot changed before loading")
            with (folder / "snapshot.pkl").open("rb") as stream: snapshot = pickle.load(stream)
            for arm, planned in entry["planned_arms"].items():
                identity = dict(dataset=out.name, suite="libero90", task=57, state=entry["state"],
                                candidate_id=ARMS.index(arm), observed_block_id=f"{entry['pool_id']}/{arm}")
                dest = out / "auxiliary" / entry["pool_id"] / arm
                actions = np.asarray(planned, dtype=np.float32)
                n = record_probe(env, fork, intervention, snapshot, dest, folder, actions)
                if n != 16:
                    completed.append(dict(identity=identity, split=entry["split"], arm=arm,
                                          valid_length=n, complete=False, retained=True))
                    raise RuntimeError("Probe terminated early; retained, no padding or replacement")
                binding = base.bind(env, 57)
                base.install_scoped_segmentation_decoder(env)
                result = base.replay(env, fork, intervention, binding, dest, snapshot, "unknown")
                frames = result[0]
                measured = teacher.derive_temporal_recovery(frames, entity_kind="rigid")
                for source_frame, measured_frame in zip(frames, measured["frames"]):
                    measured_frame["observation_evidence"] = source_frame["observation_evidence"]
                dump(dest / "temporal_teacher.json", measured)
                labels = labeler.derive_reobservation_availability(frames, identity=identity, arm=arm,
                                                                  certificate_helpers=certificate_helpers)
                dump(dest / "observation_labels.json", labels)
                completed.append(dict(identity=identity, split=entry["split"], arm=arm, valid_length=n,
                    complete=True, directory=str(dest.relative_to(out)), recovered=labels["observed_availability_recovered"]))
                dump(out / "status.json", dict(phase="collecting", captured_arms=len(completed), total_arms=8,
                     current=identity, training_started=False, elapsed_seconds=time.time() - started))
                print(json.dumps(dict(captured_arms=len(completed), total_arms=8, recovered=labels["observed_availability_recovered"])), flush=True)
        for relative, digest in protocol["source_hashes"].items():
            if sha(root / relative) != digest: raise RuntimeError("Frozen development source changed")
        if len(completed) != 8 or any(not r["complete"] for r in completed):
            raise RuntimeError("Finite probe membership incomplete")
        dump(out / "records.json", completed)
        counts = {split: sum(r["recovered"] for r in completed if r["split"] == split) for split in ("train", "val")}
        report = dict(passed=True, captured_arms=8, raw_physical_executions=24, source_unchanged=True,
                      same_arm_repeat_qc_passed=True, recovery_positive_arms=counts,
                      scientific_recovery_coverage_ready=all(v > 0 for v in counts.values()),
                      cross_view_conflict_resolved_certificates=0, no_native_candidate_pool_claim=True,
                      training_started=False, ranking_training_ready=False)
        dump(out / "completion_audit.json", report)
        dump(out / "status.json", dict(phase="complete", **report))
        files = [p for p in out.rglob("*") if p.is_file() and p.name not in ("status.json", "SHA256SUMS.txt")]
        (out / "SHA256SUMS.txt").write_text("".join(f"{sha(p)}  {p.relative_to(out)}\n" for p in sorted(files)))
    except Exception as error:
        dump(out / "failure.json", dict(error_type=type(error).__name__, message=str(error),
              completed_arms=completed, training_started=False, no_auto_repair=True))
        dump(out / "status.json", dict(phase="failed", training_started=False, captured_arms=len(completed)))
        raise
    finally:
        if env is not None: env.close()


if __name__ == "__main__": main()
