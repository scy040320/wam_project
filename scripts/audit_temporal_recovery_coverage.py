"""Independent CPU coverage audit of verified replay, NOT a training gate.

Reads immutable replay/raw manifests and writes only a NEW audit directory.
Simulator measurement validity, positive events and observable supervision are
reported separately. Five context strata are audit metadata, never labels used
by the teacher. Pool variation includes ALL candidates; nothing is selected by
recovery outcome. Geometry occupancy cannot certify contact/joint visibility.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

SCHEMA = "temporal_recovery_coverage_audit_v1"
CONTEXTS = ("normal", "visual_occlusion", "object_shift", "execution_contact_deviation", "unknown")
PHYSICAL_HEADS = ("left_finger_target_contact", "right_finger_target_contact", "target_support_contact",
                  "carried_sufficient_evidence", "lifted_sufficient_evidence", "released_sufficient_evidence",
                  "holding_after_measured_release", "joint_state_changed")
FORBIDDEN_GROUPS = {("libero90", 0, 35), ("libero90", 0, 36)} | {
    ("libero90", t, s) for t in (9, 46, 57) for s in (14, 15)
}


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def bad_number(value):
        raise ValueError(f"Nonfinite JSON number in {path}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_number)


def inside(root, relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"Manifest path escapes source directory: {relative}")
    return path


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def suite_key(value):
    return str(value).lower().replace("_", "").replace("-", "")


def context_from_pool(pool):
    for context in CONTEXTS:
        if pool.endswith("_" + context):
            return context
    raise ValueError(f"Unknown context stratum: {pool}")


def semantic_measurements(result):
    """Reason wording is nonsemantic; values, masks and timing are not."""
    return dict(frames=[dict(step=row["step"], observability=row["observability"],
        physics={name: {key: atom[key] for key in ("value", "measurement_valid", "supervision_mask")}
                 for name, atom in row["physics"].items()}) for row in result["frames"]],
        first_measured_events=result["first_measured_events"], temporal_transitions=result["temporal_transitions"])


def empty_stat():
    return dict(frames=0, measurement_valid=0, measurement_positive=0, measurement_negative=0,
                supervision_mask=0, supervised_positive=0, supervised_negative=0,
                candidates=0, candidates_with_positive_measurement=0,
                candidates_with_supervised_positive=0)


def summarize_candidates(rows):
    by_atom, by_context = defaultdict(empty_stat), {c: defaultdict(empty_stat) for c in CONTEXTS}
    pools = defaultdict(list)
    observability = {c: defaultdict(Counter) for c in CONTEXTS}
    certificate_claims = Counter()
    for row in rows:
        pools[row["pool_id"]].append(row)
        context = row["context"]
        names = set().union(*(f["physics"] for f in row["measurements"]["frames"]))
        for name in names:
            items = [f["physics"][name] for f in row["measurements"]["frames"] if name in f["physics"]]
            for stats in (by_atom[name], by_context[context][name]):
                stats["candidates"] += 1
                stats["candidates_with_positive_measurement"] += any(a["measurement_valid"] and a["value"] is True for a in items)
                stats["candidates_with_supervised_positive"] += any(a["supervision_mask"] and a["value"] is True for a in items)
                for atom in items:
                    stats["frames"] += 1
                    stats["measurement_valid"] += atom["measurement_valid"]
                    stats["supervision_mask"] += atom["supervision_mask"]
                    stats["measurement_positive"] += atom["measurement_valid"] and atom["value"] is True
                    stats["measurement_negative"] += atom["measurement_valid"] and atom["value"] is False
                    stats["supervised_positive"] += atom["supervision_mask"] and atom["value"] is True
                    stats["supervised_negative"] += atom["supervision_mask"] and atom["value"] is False
        for frame in row["raw"]["frames"]:
            for name, value in frame["observability"].items():
                observability[context][name]["unknown" if value is None else str(value).lower()] += 1
            for name in ("identity_consistent", "cross_view_consistent", "contact_observable", "target_motion_observable", "joint_state_observable"):
                if frame["observability"].get(name) is True:
                    certificate_claims[name] += 1
    pool_reports = []
    for pool_id, group in sorted(pools.items()):
        names = set().union(*(f["physics"] for row in group for f in row["measurements"]["frames"]))
        atoms = {}
        for name in sorted(names):
            signatures, supervised_signatures, event_times = [], [], []
            positive, supervised_positive = [], []
            for row in sorted(group, key=lambda x: x["candidate_id"]):
                items = [f["physics"].get(name) for f in row["measurements"]["frames"]]
                signatures.append(json.dumps([(a["value"], a["measurement_valid"]) if a else None for a in items], sort_keys=True))
                supervised_signatures.append(json.dumps([a["value"] if a and a["supervision_mask"] else None for a in items], sort_keys=True))
                event_times.append(row["measurements"]["first_measured_events"].get(name))
                if any(a and a["measurement_valid"] and a["value"] is True for a in items):
                    positive.append(row["candidate_id"])
                if any(a and a["supervision_mask"] and a["value"] is True for a in items):
                    supervised_positive.append(row["candidate_id"])
            atoms[name] = dict(measurement_candidate_variation=len(set(signatures)) > 1,
                supervised_candidate_variation=len(set(supervised_signatures)) > 1,
                first_measured_event_variation=len(set(event_times)) > 1,
                first_event_steps_by_candidate=event_times, positive_event_candidates=positive,
                supervised_positive_event_candidates=supervised_positive)
        pool_reports.append(dict(pool_id=pool_id, context=group[0]["context"], split=group[0]["split"],
            candidate_ids=sorted(r["candidate_id"] for r in group), atoms=atoms))
    return dict(atom_coverage=dict(sorted(by_atom.items())), context_coverage={c:dict(v) for c,v in by_context.items()},
        observability_flags={c:{name:dict(count) for name,count in v.items()} for c,v in observability.items()},
        external_certificate_true_claims=dict(certificate_claims), pool_variation=pool_reports)


def numeric_image_audit(row):
    return (row["mean_abs"] <= 2 and row["p95"] <= 8 and row["fraction_gt5"] <= .06 and row["psnr"] >= 30)


def audit_replay_checks(checks):
    if len(checks) != 17 or [c["step"] for c in checks] != list(range(17)):
        raise ValueError("Replay alignment indices must be 0..16")
    for row in checks[1:]:
        if not (numeric_image_audit(row["primary"]) and numeric_image_audit(row["wrist"]) and row["proprio_max_abs"] <= 1e-3):
            raise ValueError("Original post-action RGB/proprio frozen threshold failed")


def validate_atoms(derived):
    if derived.get("deployment_features_created") is not False or derived.get("training_started") is not False:
        raise ValueError("Teacher role cannot create deployment features or start training")
    if not finite(derived) or len(derived["frames"]) != 17:
        raise ValueError("Invalid derived sequence")
    for index, frame in enumerate(derived["frames"]):
        if frame["step"] != index:
            raise ValueError("Derived step alignment failure")
        for atom in frame["physics"].values():
            if type(atom["measurement_valid"]) is not bool or type(atom["supervision_mask"]) is not bool:
                raise ValueError("Measurement validity and supervision masks must be explicit booleans")
            if atom["supervision_mask"] and not atom["measurement_valid"]:
                raise ValueError("Unmeasured physical state cannot be supervised")
            if atom["measurement_valid"] and atom["value"] is None:
                raise ValueError("Known measurement cannot be None")
            if atom.get("source") != "offline_simulator_measurement":
                raise ValueError("Physical source must remain offline-only simulator measurement")
    for name, step in derived["first_measured_events"].items():
        earliest = next((f["step"] for f in derived["frames"] if f["physics"].get(name, {}).get("value") is True and f["physics"][name]["measurement_valid"]), None)
        if step != earliest:
            raise ValueError("First-event timestamp was backfilled or inconsistent")


def inspect_arrays(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"primary", "wrist", "qpos", "qvel"}:
            raise ValueError("Raw teacher arrays incomplete")
        for key in ("primary", "wrist"):
            arr = arrays[key]
            if arr.shape != (17,256,256) or arr.dtype.kind not in "iu":
                raise ValueError("Segmentation must be 17 time-aligned integer geom-ID maps")
        for key in ("qpos", "qvel"):
            arr = arrays[key]
            if arr.ndim != 2 or arr.shape[0] != 17 or not np.isfinite(arr).all():
                raise ValueError("Raw simulator state must be 17 finite frames")
        return {key: np.array(arrays[key]) for key in arrays.files}


def perform_audit(args):
    import numpy as np
    source = args.input.resolve()
    teacher_path = args.teacher.resolve()
    spec = importlib.util.spec_from_file_location("coverage_frozen_temporal_teacher", teacher_path)
    teacher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = teacher
    spec.loader.exec_module(teacher)
    protocol, completion = read_json(source/"protocol.json"), read_json(source/"completion_audit.json")
    records = read_json(source/"records.json")
    if not completion.get("passed") or len(records) != args.expected_records or len(records) % 4:
        raise ValueError("Frozen replay cardinality or completion audit failed")
    hashes = {}
    for line in (source/"SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = inside(source, relative)
        actual = file_sha(path)
        if actual != expected:
            raise ValueError(f"Replay hash mismatch: {relative}")
        hashes[relative] = actual
    if args.replay_script and file_sha(args.replay_script) != protocol["script_sha256"]:
        raise ValueError("Replay script hash does not match the frozen protocol")
    source_hash_count = 0
    if args.source_root:
        original_root = args.source_root.resolve()
        for relative, expected in read_json(source/"source_sha256.json").items():
            if file_sha(inside(original_root, relative)) != expected:
                raise ValueError(f"Immutable original hash mismatch: {relative}")
            source_hash_count += 1
    group_roles, pool_candidates, seen, rows = defaultdict(set), defaultdict(set), set(), []
    repeat_count, unconfirmed_qc, entry_mismatch = 0, [], 0
    atomically_scoped_legacy_claims = []
    for row in records:
        folder = inside(source, row["directory"])
        raw, derived, replay = (read_json(folder/name) for name in ("temporal_teacher.json", "temporal_measurements.json", "replay_audit.json"))
        identity = raw["identity"]
        if not finite(raw) or identity["dataset"] != protocol["source"] or suite_key(identity["suite"]) != "libero90":
            raise ValueError("Raw finiteness/dataset/suite identity contract failed")
        if any(type(identity[name]) is not int for name in ("task", "state", "candidate_id")) or identity["state"] < 0:
            raise ValueError("Task/state/candidate IDs must be nonnegative integers, never booleans")
        if str(identity["task"]) not in protocol["bindings"] or raw.get("segmentation_not_model_input") is not True:
            raise ValueError("Task binding missing or GT segmentation role not isolated")
        key = (identity["dataset"], suite_key(identity["suite"]), identity["task"], identity["state"], identity["candidate_id"], identity["observed_block_id"])
        if key in seen or not 0 <= identity["candidate_id"] < 4 or identity["candidate_id"] != row["candidate_id"]:
            raise ValueError("Duplicate or mismatched source-scoped identity")
        seen.add(key)
        group = key[1:4]
        if group in FORBIDDEN_GROUPS:
            raise ValueError("Consumed training data leaks into reserved 32-scenario states")
        if raw["role"] != "offline_supervision_only" or raw["split"] not in ("train", "val") or raw["split"] != row["split"]:
            raise ValueError("Role/split incompatibility")
        if identity["observed_block_id"] != row["pool_id"]+"/first_block":
            raise ValueError("Pool/observed-block identity mismatch")
        group_roles[group].add(raw["split"])
        pool_candidates[row["pool_id"]].add(row["candidate_id"])
        if len(group_roles[group]) != 1:
            raise ValueError("A task/state group leaks across train/val")
        if not replay.get("passed") or not replay.get("exact_start") or not replay.get("source_unchanged"):
            raise ValueError("Replay provenance audit failed")
        audit_replay_checks(replay["checks"])
        entry_mismatch += not (numeric_image_audit(replay["checks"][0]["primary"]) and numeric_image_audit(replay["checks"][0]["wrist"]) and replay["checks"][0]["proprio_max_abs"] <= 1e-3)
        validate_atoms(derived)
        recalculated = teacher.derive_temporal_recovery(raw["frames"], entity_kind=raw["kind"], joint_unit=raw["joint_unit"])
        if semantic_measurements(recalculated) != semantic_measurements(derived):
            raise ValueError(f"Frozen teacher semantic replay mismatch: {row['directory']}")
        arrays = inspect_arrays(folder/raw["physical_teacher_arrays"])
        # A new source must not silently inherit old geometry-only claims as
        # stronger learnable certificates. Such claims are reported, not erased.
        claims = {name for f in raw["frames"] for name in ("contact_observable", "identity_consistent", "cross_view_consistent") if f["observability"].get(name) is True}
        if claims:
            atomically_scoped_legacy_claims.append(dict(directory=row["directory"], flags=sorted(claims)))
        if row["candidate_id"] == 0:
            repeat_count += 1
            report = read_json(folder/"repeat_qc.json")
            required = ("repeat_teacher.json", "repeat_teacher_arrays.npz")
            if any(not (folder/name).exists() for name in required):
                unconfirmed_qc.append(row["pool_id"])
            else:
                repeated = read_json(folder/"repeat_teacher.json")
                ra = inspect_arrays(folder/"repeat_teacher_arrays.npz")
                delta = np.abs(np.concatenate([arrays["qpos"],arrays["qvel"]],axis=1)-np.concatenate([ra["qpos"],ra["qvel"]],axis=1))
                if delta.max() > 1e-3 or np.quantile(delta,.95) > 1e-4 or np.any(delta>1e-3):
                    raise ValueError("Repeated all-step simulator trace exceeds frozen tolerance")
                audit_replay_checks(repeated["checks"])
                for a,b in zip(raw["frames"],repeated["frames"]):
                    if any(a["physics"].get(k) != b["physics"].get(k) for k in ("left_finger_target_contact","right_finger_target_contact","target_support_contact","target_anchor_contact")):
                        raise ValueError("Repeated contact-role atoms unstable")
                rd = teacher.derive_temporal_recovery(repeated["frames"],entity_kind=raw["kind"],joint_unit=raw["joint_unit"])
                if rd["first_measured_events"] != derived["first_measured_events"]:
                    raise ValueError("Repeated first-event timing unstable")
                if not report.get("passed"):
                    raise ValueError("Repeated audit report did not pass")
        rows.append(dict(pool_id=row["pool_id"],candidate_id=row["candidate_id"],context=context_from_pool(row["pool_id"]),split=row["split"],raw=raw,measurements=derived))
    if any(ids != set(range(4)) for ids in pool_candidates.values()):
        raise ValueError("Not every retained pool has candidates 0..3")
    summary = summarize_candidates(rows)
    return dict(schema=SCHEMA, audit_integrity_passed=not unconfirmed_qc,
        observed_candidates=len(rows), observed_frames=17*len(rows), pools=len(pool_candidates),
        context_candidates=dict(Counter(row["context"] for row in rows)), split_candidates=dict(Counter(row["split"] for row in rows)),
        source_scoped_unique=True, task_state_train_val_leakage=False, reserved_32_states_in_supervision=False,
        replay_file_hashes_checked=len(hashes), original_source_hashes_checked=source_hash_count,
        original_hash_verification_requested=args.source_root is not None,
        teacher_sha256=file_sha(teacher_path), audit_code_sha256=file_sha(Path(__file__)),
        repeat_qc_pools=repeat_count, unconfirmed_all_step_repeat_qc=unconfirmed_qc,
        cached_entry_mismatch_diagnostic_candidates=entry_mismatch,
        geometry_only_certificate_claims_requiring_separate_audit=atomically_scoped_legacy_claims,
        training_ready=False, training_started=False, label_contract_ready=False,
        limitations=["Geometry co-presence does not certify visible contact interfaces, force closure or discriminable joint state.",
            "Camera registration is not proof of semantic cross-view conflict resolution.",
            "Measurement candidate variation is not automatically observable or trainable candidate contrast.",
            "First measured carrying evidence is not the first physical grasp timestamp.",
            "No new Cosmos query, train/val selection or model-gate threshold adjustment is performed."], **summary)


class SummaryTests(unittest.TestCase):
    @staticmethod
    def row(cid, value, mask):
        atom = dict(value=value,measurement_valid=value is not None,supervision_mask=mask)
        return dict(pool_id="task9_state10_unknown",candidate_id=cid,context="unknown",split="train",
            raw=dict(frames=[dict(observability={"contact_observable":None})]),
            measurements=dict(frames=[dict(step=0,physics={"carried_sufficient_evidence":atom})],first_measured_events={"carried_sufficient_evidence":0} if value is True else {}))

    def test_simulator_positive_is_not_supervised_positive(self):
        s=summarize_candidates([self.row(0,True,False)])
        a=s["atom_coverage"]["carried_sufficient_evidence"]
        self.assertEqual(a["measurement_positive"],1)
        self.assertEqual(a["supervised_positive"],0)

    def test_unknown_is_not_negative(self):
        s=summarize_candidates([self.row(0,None,False)])
        self.assertEqual(s["atom_coverage"]["carried_sufficient_evidence"]["measurement_negative"],0)

    def test_all_masked_variation_does_not_become_supervised_contrast(self):
        s=summarize_candidates([self.row(0,True,False),self.row(1,None,False)])
        p=s["pool_variation"][0]["atoms"]["carried_sufficient_evidence"]
        self.assertTrue(p["measurement_candidate_variation"])
        self.assertFalse(p["supervised_candidate_variation"])

    def test_context_suffix_and_inheritance_scope(self):
        self.assertEqual(context_from_pool("task0_state31_execution_contact_deviation"),"execution_contact_deviation")
        self.assertEqual(suite_key("libero_90"),"libero90")


def main():
    if "--self-test" in sys.argv:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SummaryTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-records",type=int,default=160)
    parser.add_argument("--teacher",type=Path,default=Path(__file__).with_name("temporal_recovery_teacher.py"))
    parser.add_argument("--replay-script",type=Path)
    parser.add_argument("--source-root",type=Path)
    args=parser.parse_args()
    source,out=args.input.resolve(),args.output.resolve()
    if out.exists() or source in out.parents or out in source.parents:
        raise ValueError("Only a NEW separate audit directory may be written")
    out.mkdir(parents=True)
    try:
        report=perform_audit(args)
    except Exception as error:
        report=dict(schema=SCHEMA,audit_integrity_passed=False,error_type=type(error).__name__,error=str(error),training_ready=False,training_started=False)
        (out/"coverage_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        raise
    (out/"coverage_audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    report_hash=file_sha(out/"coverage_audit.json")
    (out/"SHA256SUMS.txt").write_text(report_hash+"  coverage_audit.json\n",encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("audit_integrity_passed","observed_candidates","pools","training_ready","label_contract_ready")},indent=2))


if __name__=="__main__":
    main()
