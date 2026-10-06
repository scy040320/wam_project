"""Join immutable physical/observation labels to PRE-execution candidate X.

This is a supervision preparation gate, not model fitting. Contact/movement
atoms are retained as atoms; they never silently become grasp/placement/pose
predicate labels. Missing labels stay masked. Historical terminal preferences
and the frozen selected-arm auxiliary set are kept, not replaced by 160 arms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from wam_reranking.direct_recovery import extract_raw_candidate_features
from wam_reranking.recovery_contract import FEATURE_NAMES, FEATURE_SCHEMA
from wam_reranking.recovery_label_contract import OBSERVABILITY_TARGETS, TARGETS
from wam_reranking.recovery_observability_contract import ATOM_NAMES

IDENTITY = ("dataset", "suite", "task", "state", "candidate_id", "observed_block_id")
SIDECAR_SCHEMAS = {"observable_recovery_labels_sidecars_v1", "observable_recovery_labels_sidecars_v2"}
ROLE_COUNTS = {"historical_selected_arm_auxiliary": 797, "paired_recovery_supervision": 160}
RANK_SPLIT = {"train": 576, "val": 192}
AUXILIARY_SPLIT = {"train": 718, "val": 239}
TARGET_NAMES = tuple(f"{name}.{endpoint}"
    for name in (*TARGETS, *OBSERVABILITY_TARGETS, "relation_support_2d")
    for endpoint in ("after", "transition"))


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or
        re.match(r"^[A-Za-z]:", name) or Path(name).is_absolute()):
        raise ValueError("Relative sidecar path required")
    result = (root / name).resolve()
    if root not in result.parents:
        raise ValueError("Sidecar path escapes frozen root")
    return result


def identity_key(value):
    if not isinstance(value, dict) or set(value) != set(IDENTITY):
        raise ValueError("Full source-scoped identity is mandatory")
    if any(not isinstance(value[k], str) or not value[k].strip() for k in
           ("dataset", "suite", "observed_block_id")):
        raise ValueError("Source identity dataset/suite/block must be explicit nonempty strings")
    if any(type(value[k]) is not int or value[k] < 0 for k in ("task", "state", "candidate_id")):
        raise ValueError("Source identity task/state/candidate must be literal nonnegative integers")
    return tuple(value[k] for k in IDENTITY)


def group(row):
    identity = row.get("identity", row)
    suite = identity.get("suite")
    if not isinstance(suite, str) or not suite.strip():
        raise ValueError("Explicit task suite required; naked IDs cannot define a split")
    if any(type(identity.get(k)) is not int or identity[k] < 0 for k in ("task", "state")):
        raise ValueError("Explicit nonnegative integer task/state required")
    normalized = suite.lower().replace("_", "").replace("-", "")
    for name in ("suite", "task", "state"):
        if name in row and name in identity:
            a, b = row[name], identity[name]
            if name == "suite":
                if not isinstance(a, str):
                    raise ValueError("Top-level task suite conflicts with source identity")
                a, b = a.lower().replace("_", "").replace("-", ""), normalized
            if a != b:
                raise ValueError("Top-level task/state/suite conflicts with source identity")
    return (normalized, identity["task"], identity["state"])


def paired_pool(identity):
    block = identity["observed_block_id"]
    if not block.endswith("/first_block") or not block[:-len("/first_block")]:
        raise ValueError("Paired source identity must name its frozen first-block pool")
    return block[:-len("/first_block")]


def validate_input(dataset):
    """Keep ALL old rows and reject accidental truncation or role promotion."""
    if (dataset.get("schema") != "recovery_ranker_training_input_v1" or
        dataset.get("feature_schema") != FEATURE_SCHEMA or
        tuple(dataset.get("feature_names", ())) != FEATURE_NAMES or
        tuple(dataset.get("target_names", ())) != TARGET_NAMES):
        raise ValueError("Frozen pre-execution feature/input schema drift")
    rank, auxiliary = dataset.get("rank_rows"), dataset.get("auxiliary_rows")
    if not isinstance(rank, list) or len(rank) != 768 or not isinstance(auxiliary, list) or len(auxiliary) != 957:
        raise ValueError("Must preserve exactly 768 rank and 957 auxiliary rows")
    if (dict(Counter(r.get("split") for r in rank)) != RANK_SPLIT or
        dict(Counter(r.get("split") for r in auxiliary)) != AUXILIARY_SPLIT or
        dict(Counter(r.get("role") for r in auxiliary)) != ROLE_COUNTS):
        raise ValueError("Frozen rank/auxiliary split or 797+160 role cardinality drift")
    splits, rank_pools, members, auxiliary_ids = {}, defaultdict(list), {}, set()
    for row in rank + auxiliary:
        role = row["split"]
        if role not in ("train", "val") or row.get("is_qc", False) or row.get("dataset_role") in (
            "qc", "clean_b", "confirmation", "qualification", "test"):
            raise ValueError("Assessment/QC cannot enter the inherited development pool")
        key = group(row)
        if splits.setdefault(key, role) != role:
            raise ValueError("Original cross-source task/state split leakage")
        for name in ("features", "no_dag_features", "shuffled_features"):
            values = np.asarray(row.get(name), dtype=np.float64)
            if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise ValueError("Invalid frozen pre-execution features: " + name)
    for row in rank:
        if (not isinstance(row.get("pool_id"), str) or not row["pool_id"] or
            type(row.get("candidate_id")) is not int or row["candidate_id"] not in range(4)):
            raise ValueError("Original rank candidate/pool identity missing")
        rank_pools[row["pool_id"]].append(row)
    if len(rank_pools) != 192:
        raise ValueError("Original 192 rank pools must remain intact")
    for rows in rank_pools.values():
        if (len(rows) != 4 or {r["candidate_id"] for r in rows} != set(range(4)) or
            len({(group(r), r["split"]) for r in rows}) != 1):
            raise ValueError("Original rank pools require exactly four same-group candidates")
    for row in auxiliary:
        targets = np.asarray(row.get("targets"), dtype=np.float64)
        masks = np.asarray(row.get("masks"))
        if (targets.shape != (len(TARGET_NAMES),) or masks.shape != targets.shape or
            not np.isfinite(targets).all() or not np.isin(masks, (0, 1)).all()):
            raise ValueError("Frozen auxiliary target/mask contract drift")
        for index, name in enumerate(TARGET_NAMES):
            if masks[index] and not ((-1. if name.endswith(".transition") else 0.) <= targets[index] <= 1.):
                raise ValueError("Original unmasked after/transition target outside its frozen domain")
        key = identity_key(row["identity"])
        if key in members or not isinstance(row.get("row_id"), str) or not row["row_id"] or row["row_id"] in auxiliary_ids:
            raise ValueError("Duplicate/missing source-scoped auxiliary identity")
        members[key] = row
        auxiliary_ids.add(row["row_id"])
    paired = [row for row in auxiliary if row["role"] == "paired_recovery_supervision"]
    pools = defaultdict(list)
    for row in paired:
        pools[paired_pool(row["identity"])].append(row)
    if len(pools) != 40 or dict(Counter(r["split"] for r in paired)) != {"train": 80, "val": 80}:
        raise ValueError("Original 160 paired arms/40 pools and 80/80 split must remain intact")
    for rows in pools.values():
        if (len(rows) != 4 or {r["identity"]["candidate_id"] for r in rows} != set(range(4)) or
            len({(group(r), r["split"], r["identity"]["dataset"]) for r in rows}) != 1):
            raise ValueError("Paired pool identity/split or candidate cardinality drift")
    return members, {identity_key(r["identity"]) for r in paired}


def manifest_snapshot(labels):
    hashes = {}
    for line in (labels / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            raise ValueError("Malformed observable SHA256 manifest")
        expected, name = line.split("  ", 1)
        if not re.fullmatch("[0-9a-fA-F]{64}", expected):
            raise ValueError("Malformed observable SHA256 digest")
        expected = expected.lower()
        if name in hashes or digest(relative(labels, name)) != expected:
            raise ValueError("Observable sidecar hash mismatch: " + name)
        hashes[name] = expected
    if not {"records.json", "observability_label_audit.json"} <= set(hashes):
        raise ValueError("Observable records and audit must be in the verified manifest")
    return hashes


def unchanged(input_path, input_sha, labels, manifest_sha, hashes):
    if digest(input_path) != input_sha or digest(labels / "SHA256SUMS.txt") != manifest_sha:
        raise ValueError("Frozen input/observation manifest changed during preparation")
    for name, expected in hashes.items():
        if digest(relative(labels, name)) != expected:
            raise ValueError("Frozen observation source changed during preparation: " + name)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare(input_path, labels, output):
    input_path, labels, output = (Path(p).resolve() for p in (input_path, labels, output))
    if (output.exists() or output in labels.parents or labels in output.parents or
        output == input_path.parent or input_path.parent in output.parents):
        raise ValueError("Refuse overwrite/mutation of supervision or input sources")
    input_bytes = input_path.read_bytes()
    input_sha = sha256(input_bytes).hexdigest()
    manifest_sha = digest(labels / "SHA256SUMS.txt")
    hashes = manifest_snapshot(labels)
    audit = read(labels / "observability_label_audit.json")
    if (audit.get("schema") not in SIDECAR_SCHEMAS or audit.get("passed") is not True or
        audit.get("integrity_passed") is not True or audit.get("no_original_masks_or_values_changed") is not True or
        audit.get("train_val_group_leakage") != 0 or audit.get("new_queries") != 0 or audit.get("new_candidates") != 0):
        raise ValueError("Observation-source integrity or immutable-source gate failed")
    dataset = json.loads(input_bytes.decode("utf-8"))
    members, paired_members = validate_input(dataset)
    rows, seen = [], set()
    counts = {"train": Counter(), "val": Counter()}
    timed_counts = {s: {name: {str(step): Counter() for step in range(17)} for name in ATOM_NAMES}
                    for s in ("train", "val")}
    same_time = defaultdict(list)
    source_counts = {name: Counter() for name in ATOM_NAMES}
    overlay_pools = defaultdict(list)
    for record in read(labels / "records.json"):
        key = identity_key(record["identity"])
        if key in seen or key not in members:
            raise ValueError("Duplicate/unmatched observable label identity")
        seen.add(key)
        parent = members[key]
        if record["split"] != parent["split"] or parent["role"] != "paired_recovery_supervision":
            raise ValueError("Observable label split/role differs from frozen candidate source")
        if (record.get("pool_id") != paired_pool(record["identity"]) or
            type(record.get("candidate_id")) is not int or record["candidate_id"] != record["identity"]["candidate_id"]):
            raise ValueError("Observable record pool/candidate identity differs from its frozen source")
        overlay_pools[record["pool_id"]].append(record)
        path = relative(labels, record["sidecar"])
        if (hashes.get(record["sidecar"]) != record["sidecar_sha256"] or digest(path) != record["sidecar_sha256"]):
            raise ValueError("Per-candidate certificate fingerprint mismatch")
        sidecar = read(path)
        if (sidecar.get("schema") != audit["schema"] or sidecar.get("role") != "offline_supervision_only" or
            identity_key(sidecar["identity"]) != key or sidecar["split"] != parent["split"] or
            sidecar.get("future_information_labels_only") is not True or
            sidecar.get("old_masks_and_values_unchanged") is not True or
            sidecar.get("deployment_features_created") is not False or
            sidecar.get("cached_frame0_diagnostic_only") is not True):
            raise ValueError("Certificate identity, split or temporal-source contract mismatch")
        expected_atom_schema = ("recovery_observability_certificate_v2" if sidecar["schema"].endswith("_v2")
                                else "recovery_observability_certificate_v1")
        raw = extract_raw_candidate_features(parent["features"],
            feature_names=dataset["feature_names"], feature_schema=dataset["feature_schema"])
        target_rows = []
        if not isinstance(sidecar.get("labels"), list) or len(sidecar["labels"]) != 17:
            raise ValueError("All 17 source frames must be kept")
        for expected_step, frame in enumerate(sidecar["labels"]):
            step = frame["step"]
            if type(step) is not int or step != expected_step:
                raise ValueError("Certificate step is not an original observation time")
            if not isinstance(frame.get("atoms"), dict) or set(frame["atoms"]) != set(ATOM_NAMES):
                raise ValueError("Every source frame must retain exactly seven physical atoms")
            targets = {}
            for name, atom in frame["atoms"].items():
                if (atom.get("schema") != expected_atom_schema or atom.get("name") != name or
                    identity_key(atom["identity"]) != key or type(atom.get("step")) is not int or atom["step"] != step or
                    atom.get("deployment_allowed") is not False or atom.get("role") != "offline_supervision_only" or
                    atom.get("future_actual_frames_used_only_as_labels") is not True):
                    raise ValueError("Future teacher atom masquerades as deployment input")
                if type(atom.get("new_mask")) is not bool or type(atom.get("measurement_valid")) is not bool:
                    raise ValueError("Observable mask and measurement validity must be literal booleans")
                mask = atom["new_mask"]
                value = atom["physical_value"]
                if value is not None and type(value) is not bool:
                    raise ValueError("Physical event truth must be literal boolean or unknown")
                if mask and (type(value) is not bool or atom["measurement_valid"] is not True):
                    raise ValueError("Unmasked physical atom needs measured boolean truth")
                if step == 0 and mask:
                    raise ValueError("Cached frame0 is diagnostic, not prefix-supervision truth")
                if not isinstance(atom.get("reasons"), list) or any(not isinstance(r, str) for r in atom["reasons"]):
                    raise ValueError("Every masked/unmasked atom needs its original reason list")
                targets[name] = dict(measured_value=value, supervision_mask=mask,
                    reasons=atom["reasons"], semantic_role="physical_event_atom_not_predicate_truth")
                tc = timed_counts[parent["split"]][name][str(step)]
                if mask:
                    counts[parent["split"]][name + (".positive" if value else ".negative")] += 1
                    tc["positive" if value else "negative"] += 1
                    source_counts[name]["certified_positive_frames" if value else "certified_negative_frames"] += 1
                    same_time[(record["pool_id"], parent["split"], step, name)].append((record["candidate_id"], value))
                else:
                    tc["masked"] += 1
                    source_counts[name]["masked_frames"] += 1
            target_rows.append(dict(step=step, targets=targets))
        if [r["step"] for r in target_rows] != list(range(17)):
            raise ValueError("All 17 source frames must be kept")
        rows.append(dict(identity=record["identity"], split=parent["split"], pool_id=record["pool_id"],
            preexecution_x=dict(schema=raw.schema, feature_names=list(raw.feature_names), values=raw.values.tolist(),
                fingerprint=raw.fingerprint(), visual_reconstruction_complete=raw.visual_reconstruction_complete),
            supervision_only=dict(physical_event_rows=target_rows, observation_sidecar_sha256=record["sidecar_sha256"]),
            no_future_actual_or_teacher_in_x=True))
    if (type(audit.get("completed_candidates")) is not int or
        len(rows) != audit["completed_candidates"] or len(rows) not in (32, 160) or
        audit.get("pools") != len(overlay_pools) or len(overlay_pools) != len(rows) // 4):
        raise ValueError("Frozen source cardinality drift")
    for records in overlay_pools.values():
        if (len(records) != 4 or {r["candidate_id"] for r in records} != set(range(4)) or
            len({(group(r), r["split"], r["identity"]["dataset"]) for r in records}) != 1):
            raise ValueError("Observation overlay must retain all four same-group source candidates per pool")
    if len(rows) == 160 and seen != paired_members:
        raise ValueError("Full overlay must join every one of the frozen 160 paired candidates")
    if audit.get("split_candidates") != dict(Counter(r["split"] for r in rows)):
        raise ValueError("Observation audit split counts differ from joined source rows")
    audited_counts = audit.get("physical_vs_certified_counts", {})
    for name in ATOM_NAMES:
        for field in ("certified_positive_frames", "certified_negative_frames", "masked_frames"):
            if audited_counts.get(name, {}).get(field) != source_counts[name][field]:
                raise ValueError("Observation audit certificate counts differ from retained atoms")
    for (pool, split, step, name), candidates in same_time.items():
        positives = [cid for cid, value in candidates if value]
        negatives = [cid for cid, value in candidates if not value]
        count = len(positives) * len(negatives)
        timed_counts[split][name][str(step)]["same_pool_positive_negative_pairs"] += count
        timed_counts[split][name][str(step)]["pools_with_supervised_difference"] += int(count > 0)
    # Never infer a pair from an early positive and a later endpoint negative.
    for split in timed_counts:
        for name in ATOM_NAMES:
            for step in range(17):
                cell = timed_counts[split][name][str(step)]
                for field in ("positive", "negative", "masked", "same_pool_positive_negative_pairs", "pools_with_supervised_difference"):
                    cell.setdefault(field, 0)
    atom_blockers = {}
    for name in ATOM_NAMES:
        cells = timed_counts["train"][name].values()
        positives = sum(c["positive"] for c in cells)
        negatives = sum(c["negative"] for c in timed_counts["train"][name].values())
        paired = sum(c["same_pool_positive_negative_pairs"] for c in timed_counts["train"][name].values())
        blockers = ["physical_atom_to_predicate_valid_at_use_time_mapping_not_audited"]
        if not positives:
            blockers.append("no_certified_train_positive")
        if not negatives:
            blockers.append("positive_only_or_no_certified_train_negative")
        if not paired:
            blockers.append("no_certified_train_same_pool_same_step_positive_negative_pair")
        atom_blockers[name] = blockers
    unchanged(input_path, input_sha, labels, manifest_sha, hashes)
    if output.exists():
        raise ValueError("Output was created concurrently")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema="observable_physical_event_overlay_v1", passed=True,
        inherited_rank_rows=len(dataset["rank_rows"]), inherited_auxiliary_rows=len(dataset["auxiliary_rows"]),
        overlay_candidates=len(rows), overlay_frames=17*len(rows), split=dict(Counter(r["split"] for r in rows)),
        certified_atoms_by_split={s:dict(c) for s,c in counts.items()},
        certified_atoms_by_split_and_step={s:{name:{step:dict(cell) for step,cell in times.items()}
            for name,times in atoms.items()} for s,atoms in timed_counts.items()},
        atom_training_blockers=atom_blockers,
        same_time_pair_definition="same frozen pool + same physical atom + same step + two certified opposing candidate values",
        early_positive_and_endpoint_negative_are_not_same_time_pair=True,
        source_identity_join="dataset/suite/task/state/candidate_id/observed_block_id", group_leakage=0,
        # This join only copies immutable parent X. Its own feature names
        # cannot prove that the earlier producer never used future/GT data.
        new_overlay_execution_x_leakage=False, future_execution_x_leakage=None,
        inherited_source_producer_audit_source="training_input.json:training_audit",
        inherited_source_producer_gt_leakage_claim=dataset.get("training_audit", {}).get("deployment_gt_leakage"),
        inherited_source_producer_audit_payload_sha256=sha256(json.dumps(
            dataset.get("training_audit", {}), sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest(),
        inherited_source_producer_source_hashes_count=len(dataset.get("training_audit", {}).get("source_hashes", {})),
        original_604_feature_gt_freedom_independently_confirmed=False,
        original_604_feature_temporal_provenance="inherited producer audit; not independently confirmed by overlay",
        producer_audit_claim_does_not_become_new_certification=True,
        original_input_sha256=input_sha,
        original_input_sha256_before=input_sha, original_input_sha256_after=digest(input_path),
        observation_manifest_sha256=manifest_sha,
        source_hashes_rechecked_before_output=True,
        event_atoms_do_not_implicitly_certify_predicates=True,
        missing_supervision_is_masked_not_negative=True, training_started=False, training_ready=False,
        next_gate="audited predicate validity AT use time; sufficient discriminator labels before fitting direct-score heads",
        earlier_event_does_not_prove_current_validity=True,
        original_rank_auxiliary_data_retained=True, inherited_rank_split=RANK_SPLIT,
        inherited_auxiliary_split=AUXILIARY_SPLIT, inherited_auxiliary_role_counts=ROLE_COUNTS,
        full_160_overlay_complete=len(rows)==160,
        paired_candidates_without_overlay=160-len(rows),
        raw_visual_unreconstructable_candidates=sum(not r["preexecution_x"]["visual_reconstruction_complete"] for r in rows),
        training_blockers=["physical_atom_to_predicate_valid_at_use_time_mapping_not_audited",
            "each_enabled_head_requires_certified_train_positive_negative_and_same_time_pairs",
            "positive_only_or_unobserved_targets_cannot_enable_direct_recovery_scores"],
        no_automatic_main_experiment=True)
    # Store every original row verbatim as JSON values. The read-only parent
    # remains hash-linked; the old ready flag is NOT inherited as new approval.
    dump(output / "inherited_training_rows.json", dict(schema="preserved_recovery_rank_and_auxiliary_rows_v1",
        original_input_sha256=input_sha, feature_schema=dataset["feature_schema"],
        feature_names=dataset["feature_names"], target_names=dataset["target_names"],
        rank_rows=dataset["rank_rows"], auxiliary_rows=dataset["auxiliary_rows"],
        training_ready=False, training_started=False))
    dump(output / "event_supervision_rows.json", dict(schema=report["schema"], rows=rows))
    unchanged(input_path, input_sha, labels, manifest_sha, hashes)
    dump(output / "input_audit.json", report)
    (output / "SHA256SUMS.txt").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(output.glob("*.json"))), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.input, a.labels, a.output), ensure_ascii=False))


if __name__ == "__main__": main()
