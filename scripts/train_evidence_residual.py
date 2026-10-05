"""Train only a cause residual on a frozen shared candidate backbone.

Input contains deployable feature/decision records plus separate supervised
outcomes. Outcome fields are read only for fitting and reporting, never by
the deployment policy. A failed gate is reported, not silently weakened.
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from wam_reranking.contracts import CandidateDecision
from wam_reranking.candidate_utility import CandidateUtilityModel, select_with_utility
from wam_reranking.evidence_residual import fit_evidence_residual, select_evidence_residual


def decision(record):
    return CandidateDecision(int(record["candidate_id"]), bool(record["accepted"]),
        float(record["official_value"]), record["total_score"],
        tuple(record["rejection_reasons"]), dict(record["components"]))


def evaluate(rows, backbone, residual):
    records=[]
    for row in rows:
        cs=row["candidates"]
        ds=[decision(c["decision"]) for c in cs]
        xs={c["candidate_id"]:np.asarray(c["backbone_features"]) for c in cs}
        zs={c["candidate_id"]:np.asarray(c["cause_features"]) for c in cs}
        chosen,scores=select_evidence_residual(decisions=ds,backbone_features=xs,
            cause_features=zs,backbone=backbone,residual=residual)
        unfiltered=[replace(d,accepted=True,rejection_reasons=(),
                            components={"dependency_risk":0.,"uncertainty":0.}) for d in ds]
        base,_=select_with_utility(unfiltered,xs,backbone)
        by_id={c["candidate_id"]:c for c in cs}
        anchor=max(cs,key=lambda c:(c["value"],-c["candidate_id"]))
        result=None if chosen is None else by_id[chosen.candidate_id]["outcome"]
        # No counterfactual fallback outcome is invented from a missing arm.
        success=None if result is None else bool(result["success"])
        records.append({"key":row["key"],"selected":None if chosen is None else chosen.candidate_id,
            "candidate_only_selected":base.candidate_id,"value_selected":anchor["candidate_id"],
            "success":success,"candidate_only_success":bool(by_id[base.candidate_id]["outcome"]["success"]),
            "value_success":bool(anchor["outcome"]["success"]),
            "covered":any(c["outcome"]["success"] for c in cs),
            "mixed":any(c["outcome"]["success"] for c in cs) and not all(c["outcome"]["success"] for c in cs),
            "nonzero_cause_correction":any(residual.score(z)!=0. for z in zs.values())})
    report={"pools":len(records),"full_success":sum(r["success"] is True for r in records),
        "candidate_only_success":sum(r["candidate_only_success"] for r in records),
        "value_success":sum(r["value_success"] for r in records),
        "unobserved_fallback":sum(r["success"] is None for r in records),
        "covered":sum(r["covered"] for r in records),"mixed":sum(r["mixed"] for r in records),
        "active_cause_pools":sum(r["nonzero_cause_correction"] for r in records),"rows":records}
    for label in ("candidate_only","value"):
        report["gains_vs_"+label]=sum(r["success"] is True and not r[label+"_success"] for r in records)
        report["harms_vs_"+label]=sum(r["success"] is False and r[label+"_success"] for r in records)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--backbone",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    dataset=json.loads(args.data.read_text())
    if dataset["schema"]!="evidence_residual_training_v1":raise ValueError("data schema mismatch")
    rows=dataset["rows"]
    train=[r for r in rows if r["split"]=="train"]
    val=[r for r in rows if r["split"]=="val"]
    if len(train)!=144 or len(val)!=48:raise ValueError("original split count drift")
    if {tuple(r["key"][:2]) for r in train}&{tuple(r["key"][:2]) for r in val}:raise ValueError("group leakage")
    if any(r["key"][1]>7 for r in rows):raise ValueError("closed-loop acceptance states cannot enter training")
    backbone=CandidateUtilityModel.from_record(json.loads(args.backbone.read_text()))
    pairs=[];support=[]
    for row in train:
        cs=row["candidates"]
        support.extend(np.asarray(c["cause_features"]) for c in cs)
        for better in cs:
            for worse in cs:
                bo,wo=better["outcome"],worse["outcome"]
                # Fit success preference only. Cost objectives are reported
                # later; do not trade baseline success for shorter failures.
                if bo["success"] and not wo["success"]:
                    margin=backbone.score(np.asarray(better["backbone_features"]))-backbone.score(np.asarray(worse["backbone_features"]))
                    pairs.append((np.asarray(better["cause_features"]),np.asarray(worse["cause_features"]),margin))
    fitted=fit_evidence_residual(pairs,support)
    calibration=[]
    # Fixed before any validation or consumed closed-loop replay.
    for gain in (0.,.125,.25,.5,1.):
        trial=replace(fitted,gain=gain)
        rr=evaluate(train,backbone,trial)
        calibration.append({"gain":gain,**{k:v for k,v in rr.items() if k!="rows"}})
    eligible=[r for r in calibration if r["harms_vs_candidate_only"]==0 and r["harms_vs_value"]==0 and r["unobserved_fallback"]==0]
    best=max(eligible,key=lambda r:(r["full_success"],-r["gain"])) if eligible else None
    selected=replace(fitted,gain=0. if best is None else best["gain"])
    train_report=evaluate(train,backbone,selected)
    # Validation is read only once after train-only calibration has finished.
    val_report=evaluate(val,backbone,selected)
    gate={"train_calibration_feasible":best is not None,
        "train_not_worse_than_candidate_only":train_report["full_success"]>=train_report["candidate_only_success"],
        "validation_not_worse_than_candidate_only":val_report["full_success"]>=val_report["candidate_only_success"],
        "train_not_worse_than_value":train_report["full_success"]>=train_report["value_success"],
        "validation_not_worse_than_value":val_report["full_success"]>=val_report["value_success"],
        "train_zero_success_harm":train_report["harms_vs_value"]==train_report["harms_vs_candidate_only"]==0,
        "validation_zero_success_harm":val_report["harms_vs_value"]==val_report["harms_vs_candidate_only"]==0,
        "all_outcomes_observed":train_report["unobserved_fallback"]==val_report["unobserved_fallback"]==0}
    gate["passed"]=all(gate.values())
    report={"role":"original_development_regression_not_independent_test",
        "train":train_report,"validation":val_report,"gate":gate,"calibration":calibration,
        "cause_contribution_demonstrated":train_report["gains_vs_candidate_only"]>0 and val_report["gains_vs_candidate_only"]>0,
        "gain_selected_from_train_only":True,"acceptance64_excluded_from_fitting":True,
        "zero_gain_is_shared_candidate_backbone_not_attribution_benefit":True}
    (args.output/"evidence_residual_model.json").write_text(json.dumps(selected.to_record(),indent=2)+"\n")
    (args.output/"training_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"train":{k:v for k,v in train_report.items() if k!="rows"},
        "validation":{k:v for k,v in val_report.items() if k!="rows"},"gain":selected.gain,
        "gate":gate,"cause_contribution_demonstrated":report["cause_contribution_demonstrated"]}))
    if not gate["passed"]:raise SystemExit(2)


if __name__=="__main__":main()
