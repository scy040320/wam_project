"""Audit-only additive PRE bundle from authenticated shared-CURRENT sidecars.

No model fit, action selection, environment execution, label join or deployment.
All ordinary candidate capacity is preserved under learned-content masking.
Each whole-chain control uses its own immutable journal's attribution/history;
the input sidecar is not injected into any frozen model's feature schema.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import inspect
from pathlib import Path
import sys
import numpy as np

SCHEMA = "source_scoped_shared_current_PRE_bundle_audit_v2"
MODES = ("full", "ranker_command", "learned_no_dag", "cause_only", "quality_only",
         "shuffled", "masked_soft_only")


def load_payload(name, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError("Missing explicit isolated payload: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_preapproved_receipt(path, expected, sha):
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected) or sha(path) != expected):
        raise ValueError("Independent preapproved journal/report SHA256 receipt required")
    return expected


def import_frozen_source_pin(path, expected_sources, pins, sha):
    """Transfer an existing frozen audit entry, never create a self-hash receipt."""
    key = str(Path(path).resolve())
    expected = expected_sources.get(key)
    if (type(expected) is not str or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected)
            or sha(path) != expected
            or (key in pins and pins[key] != expected)):
        raise ValueError("Explicit source metadata lacks its independently frozen pin")
    pins[key] = expected
    return expected


def source_identity(key, block_meta, block_id, dataset):
    """Use explicit frozen aligned block/endpoint metadata, not fact timestamps."""
    index = block_meta.get("block_index")
    endpoint = block_meta.get("source_endpoint_reference", {}).get("endpoint_block_index")
    actual_step = block_meta.get("actual_observation_policy_step")
    predicted_step = block_meta.get("prediction_target_policy_step")
    if (type(index) is not int or type(endpoint) is not int or endpoint != index + 1
            or type(actual_step) is not int or type(predicted_step) is not int
            or min(actual_step, predicted_step) < 0
            or block_meta.get("alignment_valid_t_plus_H") is not True
            or block_meta.get("executed_steps") != 16
            or actual_step != predicted_step
            or block_id != f"libero90_task{key[0]}_block{index}"):
        raise ValueError("Original source/usage-time block identity is not aligned")
    return dict(dataset=dataset, suite="libero90", task=key[0], state=key[1], moment=key[2],
                condition=key[3], block_id=block_id, block_index=endpoint)


def reconstruct_attribution(helper, row, mode):
    if mode == "ranker_command":
        q = row["evidence_quality"]
        attr = helper.chain.absent_carrier(row["source_block_id"],
            (q["primary_reliable"], q["wrist_reliable"], q["execution_reliable"]))
    else:
        attr = helper.routing.ControlledAttribution(row["factor_probs"], row["class_probs"],
            helper.rt.CoarseCause(row["projected_cause"]), row["confidence"], row["entropy"],
            helper.rt.EvidenceQuality(**row["evidence_quality"]), row["source_block_id"],
            {k: helper.rt.TriValue(v) for k,v in row["factor_states"].items()},
            row["factor_confidences"], tuple(row.get("semantic_conflicts", ())))
    restored = helper.plain(attr)
    # A genuine frozen output has no control-only field. The duck-compatible
    # carrier adds an empty default, never a new conflict or relabeling.
    if "semantic_conflicts" not in row and restored.get("semantic_conflicts") == []:
        restored.pop("semantic_conflicts")
    if restored != row:
        raise ValueError("Original controlled attribution content changed during reconstruction")
    return attr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/wam_project/cosmos-policy"))
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-sha256", required=True)
    parser.add_argument("--journal-report-sha256", required=True)
    args = parser.parse_args()
    root, visual_root, output = args.root.resolve(), args.visual_root.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "outputs"):
        raise ValueError("Fresh additive outputs directory required")
    here = Path(__file__).resolve().parent
    extract = load_payload("source_scoped_RGB_sidecar_loader", here / "reextract_shared_current_forecast_evidence.py")
    helper = extract.load_helper(root)
    shared = extract.load_shared_payload(helper)
    # A separate module name leaves the original V8 parser/package module and
    # all frozen references untouched. Only this new feature API binds it.
    ordered = helper.rt.load("wam_reranking.source_scoped_ordered_parser", here / "candidate_effects.py")
    if "stage_semantics" not in inspect.signature(ordered.parse_candidate_effect).parameters:
        raise ValueError("Explicit new ordered candidate parser payload required")
    features = helper.rt.load("wam_reranking.source_scoped_forecast_features",
                             here / "source_scoped_forecast_features.py")
    features.parse_candidate_effect = ordered.parse_candidate_effect
    sha, read, dump = extract.sha, extract.read, extract.dump
    anchor = root / extract.ANCHOR
    verify_preapproved_receipt(anchor / "decision_journals.json", args.journal_sha256, sha)
    verify_preapproved_receipt(anchor / "root_replay_report.json", args.journal_report_sha256, sha)
    report = read(visual_root / "completion_audit.json")
    if (report.get("passed") is not True or report.get("schema") != extract.SCHEMA
            or report.get("phase") != "full" or report.get("pools") != 192
            or report.get("train") != 144 or report.get("val") != 48):
        raise ValueError("Completed authenticated full192 new-schema sidecar required")
    pins = dict(report["source_sha256"])
    for relative, expected in report["output_sha256"].items():
        path = extract.inside(visual_root, relative)
        if sha(path) != expected:
            raise ValueError("New visual sidecar bytes differ: " + relative)
        pins[str(path)] = expected
    for path in (Path(__file__), Path(extract.__file__), Path(shared.__file__), Path(features.__file__),
                 Path(ordered.__file__), anchor / "decision_journals.json",
                 anchor / "root_replay_report.json", visual_root / "completion_audit.json"):
        if str(path) in pins and pins[str(path)] != sha(path):
            raise ValueError("Pinned source changed before input audit")
        pins[str(path)] = sha(path)
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError("Immutable source/model/extractor bytes differ: " + path)
    pairs, sources, authoritative_pins = extract.source_index(helper, root)
    for path, expected in authoritative_pins.items():
        if path in pins and pins[path] != expected:
            raise ValueError("Original sources disagree with new sidecar pins")
        pins[path] = expected
    previous = read(anchor / "root_replay_report.json")
    if not previous["passed"] or previous["counts"]["original_identity_matches"] != 192:
        raise ValueError("Immutable whole-chain source audit is not passed")
    expected_sources = {}
    for filename in ("source_sha256.json", "addition_sha256.json"):
        manifest = anchor / filename
        if pins.get(str(manifest)) != sha(manifest):
            raise ValueError("Existing immutable source manifest changed")
        for path, digest in read(manifest).items():
            if path in expected_sources and expected_sources[path] != digest:
                raise ValueError("Existing immutable source manifests disagree")
            expected_sources[path] = digest
    journal_rows = read(anchor / "decision_journals.json")
    journals = {tuple(row["key"]): row for row in journal_rows}
    sidecars = {tuple(row["key"]): row for row in report["rows"]}
    if (len(journals) != len(journal_rows) or len(sidecars) != len(report["rows"])
            or set(journals) != set(pairs) or set(sidecars) != set(pairs)):
        raise ValueError("Whole-chain source/new sidecar pool identity differs")
    raw = {}
    for result_path in sorted((helper.SOURCE / "outcomes").glob("*/results.json")):
        for row in read(result_path):
            key = helper.io.key(row)
            if key in raw:
                raise ValueError("Duplicate source block pointer")
            # No terminal outcome field is accessed or joined here.
            raw[key] = row["evidence_block"]
    if set(raw) != set(pairs):
        raise ValueError("Original evidence block membership differs")
    identities, attrs = {}, {}
    for key in sorted(pairs):
        block_path = extract.inside(root, raw[key]) / "block.json"
        # The RGB reextractor only consumes RGB/maps/plans, so its output pin
        # subset legitimately omits block metadata. Import the existing entry
        # from the prior frozen full-scope audit, not a newly observed hash.
        import_frozen_source_pin(block_path, expected_sources, pins, sha)
        original = journals[key]["variants"]["full"]["attribution"]
        identities[key] = source_identity(key, read(block_path), original["source_block_id"], helper.SOURCE.name)
        attrs[key] = reconstruct_attribution(helper, original, "full")
    center = features.calibrate_train_channels([dict(identity=identities[key], split="train",
        source_sha256={"attribution":pins[str(anchor / "decision_journals.json")]}, attribution=attrs[key])
        for key in sorted(pairs) if pairs[key]["split"] == "train"])
    output.mkdir(parents=True)
    try:
        rows, matrix_pins = [], {}
        for key in sorted(pairs):
            sidecar = read(extract.inside(visual_root, sidecars[key]["visual_sidecar"]))
            if (sidecar["schema"] != extract.SCHEMA or sidecar["key"] != list(key)
                    or sidecar["split"] != pairs[key]["split"] or sidecar["suite"] != "libero90"
                    or sidecar["physical_facts_created"] or sidecar["actual_after_execution_images_used"]):
                raise ValueError("New PRE sidecar source/role mismatch")
            candidates = sidecar["candidates"]
            if [v["candidate_id"] for v in candidates] != list(range(4)):
                raise ValueError("K4 membership changed")
            relation = helper.bundle.FROZEN_RELATIONS[(key[0],key[2])].relation
            if sidecar["frozen_language_binding"]["relation"] != relation:
                raise ValueError("Frozen public language relation binding differs")
            if sidecar["source_rgb"]["current"] != [
                    {**ref, "path":str(path), "sha256":pins[str(path)]}
                    for ref,path in zip(sidecar["source_rgb"]["current"], sources[key]["current"], strict=True)]:
                raise ValueError("Sidecar CURRENT references differ from authoritative source")
            for cid,candidate in enumerate(sources[key]["candidates"]):
                expected_forecast = [dict(path=str(candidate[field]),sha256=pins[str(candidate[field])])
                                     for field in ("predicted_primary_path","predicted_wrist_path")]
                if [{k:ref[k] for k in ("path","sha256")} for ref in sidecar["source_rgb"]["forecast"][cid]] != expected_forecast:
                    raise ValueError("Candidate own-FORECAST references differ from authoritative source")
                if candidates[cid]["planned_actions_source"] != dict(
                        path=str(candidate["actions_path"]),sha256=pins[str(candidate["actions_path"])]):
                    raise ValueError("Candidate plan reference differs from authoritative source")
            plans = [np.load(c["actions_path"], allow_pickle=False) for c in sources[key]["candidates"]]
            evs = [helper.rt.CandidateVisualEvidence(**c["candidate_visual_evidence"]) for c in candidates]
            diagnostics = [c["fusion_diagnostic"] for c in candidates]
            named = dict(current_primary=pins[str(sources[key]["current"][0])],
                current_wrist=pins[str(sources[key]["current"][1])],
                localizer_model=pins[str(Path(extract.MODEL_CACHE) / "FROZEN_MANIFEST.json")],
                localizer_code=pins[inspect.getfile(helper.bundle.CLIPSegTargetLocalizer)],
                shared_fusion_code=pins[str(Path(shared.__file__).resolve())],
                candidate_parser_code=pins[str(Path(ordered.__file__).resolve())],
                attribution=pins[str(anchor / "decision_journals.json")])
            for cid, candidate in enumerate(sources[key]["candidates"]):
                for field, pathfield in (("plan","actions_path"),("predicted_primary","predicted_primary_path"),
                                        ("predicted_wrist","predicted_wrist_path")):
                    named[f"candidate_{cid}_{field}"] = pins[str(candidate[pathfield])]
            matrices, audits, complete_main = {}, {}, None
            for mode in MODES:
                base = "full" if mode == "masked_soft_only" else mode
                arm = journals[key]["variants"][base]
                attr = reconstruct_attribution(helper, arm["attribution"], base)
                snapshot = arm["live_belief"]
                context = dict(identity=identities[key], variant=mode, no_dag=mode=="learned_no_dag",
                               snapshot_sha256=features._digest(snapshot))
                kwargs = dict(attribution=attr, plans=plans, visual_evidences=evs, fusion_diagnostics=diagnostics,
                    identity=identities[key], source_sha256=named, split=pairs[key]["split"], variant=mode,
                    history_context=context, belief_snapshot=snapshot, centering=center, relation=relation,
                    mask_learned=mode in {"ranker_command","masked_soft_only"}, no_dag=mode=="learned_no_dag")
                main_matrix, interaction, audit = features.source_scoped_features(**kwargs)
                repeated_main, repeated_interaction, repeated_audit = features.source_scoped_features(**kwargs)
                if (not np.array_equal(main_matrix,repeated_main) or not np.array_equal(interaction,repeated_interaction)
                        or audit != repeated_audit):
                    raise ValueError("New PRE features are not deterministic")
                if complete_main is None:
                    complete_main = main_matrix
                if not np.array_equal(main_matrix,complete_main):
                    raise ValueError("Ordinary main capacity changed across attribution controls")
                matrices[f"{mode}_main"], matrices[f"{mode}_interaction"] = main_matrix, interaction
                audits[mode] = audit
            path = output / ("pool_" + "_".join(map(str,key)) + ".npz")
            np.savez_compressed(path, **matrices)
            matrix_pins[path.name] = sha(path)
            rows.append(dict(key=list(key),split=pairs[key]["split"],matrices=path.name,audits=audits))
        for path, expected in pins.items():
            if sha(path) != expected:
                raise ValueError("Source/model/code changed during PRE bundle audit")
        summary = dict(passed=True,schema=SCHEMA,pools=192,candidates=768,train=144,val=48,
            modes=list(MODES),main_features=len(features.MAIN_NAMES),interaction_features=len(features.INTERACTION_NAMES),
            source_sha256=pins,output_sha256=matrix_pins,centering=asdict(center),rows=rows,
            train_only_channel_mean=True,terminal_outcomes_consumed=False,terminal_labels_joined=False,
            ordinary_main_identical_across_controls=True,own_variant_history_used=True,
            physical_TRUE_facts_created=0,observation_registry_populated=False,
            unknown_or_forecast_proxy_admitted_as_fact=False,policy_queries=0,actions=0,fits=0,
            old_model_feature_schema_changed=False,model_deployed=False,closedloop_run=False,
            utility_improvement_claim=False,current_physical_certificate_limitation="original current witness registry remains empty")
        dump(output / "completion_audit.json",summary)
        dump(output / "source_sha256.json",pins)
        print({k:v for k,v in summary.items() if k not in {"source_sha256","output_sha256","rows"}})
    except BaseException as error:
        dump(output / "input_audit_failure.json",dict(schema=SCHEMA,error_type=type(error).__name__,error=str(error),
            queries=0,actions=0,fits=0,thresholds_changed=False))
        raise


if __name__ == "__main__":
    main()
