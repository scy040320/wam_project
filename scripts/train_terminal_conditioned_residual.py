#!/usr/bin/env python3
"""Bounded terminal-utility training and score-channel ablation on consumed pools.

Never substitutes terminal success for signed physical-recovery supervision.
Runs a source audit and exact V8 replay BEFORE fitting; keeps all 192 pools.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from wam_reranking import (CandidateUtilityModel, CandidateVisualEvidence, SelectorMode,
    initial_belief, select_hard_gate_value_tiebreak)
from wam_reranking.candidate_effects import parse_candidate_effect


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected non-scalar JSON value: {type(value).__name__}")


def key(row):
    return (int(row["task_id"]), int(row["state"]), int(row["moment_id"]), str(row["condition"]))


def index_unique(rows):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError(f"Duplicate scoped source pool: {k}")
        result[k] = row
    return result


def clean_attribution(row, threshold):
    # Explicit whitelist: dataset truth, labels, condition and terminal fields
    # are never passed to the feature encoder. Source IDs are provenance only.
    fields = ("factor_prob", "unknown_prob", "predicted", "class_probs", "confidence",
              "primary_available", "wrist_available", "action_record_available")
    result = {k: row[k] for k in fields}
    result["unknown_threshold"] = threshold
    return result


def donor_map(pools):
    """Same split, different task/state, outcome-blind fixed cyclic mapping."""
    result = {}
    for split in ("train", "val"):
        items = [p for p in pools if p["split"] == split]
        groups = sorted({p["key"][:2] for p in items})
        if len(groups) < 2:
            raise ValueError("At least two distinct donor groups per split required")
        by_key = {p["key"]: p for p in items}
        for p in items:
            group = groups[(groups.index(p["key"][:2]) + 1) % len(groups)]
            dk = (*group, *p["key"][2:])
            if dk not in by_key:
                raise ValueError("No matched candidate-independent donor slot")
            result[p["key"]] = dk
    return result


def labels(outcome):
    # Outcomes are kept in a separate Y table, never merged into X.
    if type(outcome.get("success")) is not bool:
        raise ValueError("Actual terminal success must be bool")
    steps = outcome["executed_steps"]
    calls = outcome["continuation_wam_calls"]
    if type(steps) is not int or steps < 0 or type(calls) is not int or calls < 0:
        raise ValueError("Actual terminal cost missing")
    return {"success": outcome["success"], "steps": steps, "calls": calls,
            "risk_proxy": outcome.get("risk_proxy", {})}


def evaluate(pools, residual, mode, residual_api, margin):
    rows = []
    for p in pools:
        x = p["controls"][mode]
        deltas = residual.deltas(x) if mode != "masked" else np.zeros(len(x))
        chosen, scores = residual_api.select_residual(p["decisions"], p["base_scores"], deltas, margin)
        selected_id = None if chosen is None else chosen.candidate_id
        selected = p["fallback_y"] if selected_id is None else p["y"][selected_id]
        baseline = p["fallback_y"] if p["baseline_id"] is None else p["y"][p["baseline_id"]]
        value = p["y"][p["value_id"]]
        success = None if selected is None else selected["success"]
        row = {"key": list(p["key"]), "selected_candidate": selected_id,
            "selected_success": success, "v8_candidate": p["baseline_id"],
            "v8_success": None if baseline is None else baseline["success"],
            "value_candidate": p["value_id"], "value_success": value["success"],
            "oracle_success": any(y["success"] for y in p["y"]),
            "mixed": 0 < sum(y["success"] for y in p["y"]) < len(p["y"]),
            "observed": selected is not None, "fallback": p["fallback"] if selected_id is None else None,
            "steps": None if selected is None else selected["steps"],
            "continuation_wam_calls": None if selected is None else selected["calls"],
            "pool_generation_calls": p["generation_calls"],
            "deltas": deltas.tolist(), "scores": {str(k):v for k,v in scores.items()},
            "risk_proxy": None if selected is None else selected["risk_proxy"]}
        for name, ref in (("v8", baseline), ("value", value)):
            common = selected is not None and ref is not None and success and ref["success"]
            row[f"common_success_steps_delta_vs_{name}"] = selected["steps"] - ref["steps"] if common else None
            row[f"common_success_calls_delta_vs_{name}"] = selected["calls"] - ref["calls"] if common else None
        rows.append(row)
    result = {"pools": len(rows), "selected_successes": sum(r["selected_success"] is True for r in rows),
        "oracle_successes": sum(r["oracle_success"] for r in rows), "mixed_pools": sum(r["mixed"] for r in rows),
        "all_failed_pools": sum(not r["oracle_success"] for r in rows),
        "unobserved_fallbacks": sum(not r["observed"] for r in rows),
        "success_at_selection_on_covered": None,
        "nonzero_correction_pools": sum(any(abs(x)>1e-12 for x in r["deltas"]) for r in rows),
        "switches_vs_v8": sum(r["selected_candidate"]!=r["v8_candidate"] for r in rows), "rows": rows}
    for name in ("v8", "value"):
        result[f"{name}_successes"] = sum(r[f"{name}_success"] is True for r in rows)
        result[f"improvements_vs_{name}"] = sum(r["selected_success"] is True and r[f"{name}_success"] is False for r in rows)
        result[f"harms_vs_{name}"] = sum(r["selected_success"] is False and r[f"{name}_success"] is True for r in rows)
        ds = [r[f"common_success_steps_delta_vs_{name}"] for r in rows if r[f"common_success_steps_delta_vs_{name}"] is not None]
        dc = [r[f"common_success_calls_delta_vs_{name}"] for r in rows if r[f"common_success_calls_delta_vs_{name}"] is not None]
        result[f"common_success_vs_{name}"] = {"count":len(ds), "steps_delta_sum":sum(ds),
            "calls_delta_sum":sum(dc), "steps_delta_mean":float(np.mean(ds)) if ds else None,
            "calls_delta_mean":float(np.mean(dc)) if dc else None}
    success_rows = [r for r in rows if r["selected_success"] is True]
    result["selected_success_cost"] = {"count":len(success_rows),
        "mean_remaining_steps":float(np.mean([r["steps"] for r in success_rows])) if success_rows else None,
        "mean_continuation_wam_calls":float(np.mean([r["continuation_wam_calls"] for r in success_rows])) if success_rows else None,
        "mean_calls_with_common_pool_generation":float(np.mean([r["continuation_wam_calls"]+r["pool_generation_calls"] for r in success_rows])) if success_rows else None}
    if result["oracle_successes"]:
        result["success_at_selection_on_covered"] = sum(r["oracle_success"] and r["selected_success"] is True for r in rows)/result["oracle_successes"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = read(args.protocol)
    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=json_scalar)+"\n",encoding="utf-8")
    def status(phase, **values):
        write("status.json", {"phase": phase,"time_unix":time.time(),**values})
    pins = {}
    def pin(path):
        p = Path(path)
        pins[str(p)] = digest(p)
        return p
    try:
        status("source_audit")
        write("frozen_protocol.json", protocol)
        for path, expected in protocol["frozen_sha256"].items():
            if digest(pin(path)) != expected:
                raise ValueError(f"Frozen source mismatch: {path}")
        api = load_module("wam_reranking.terminal_conditioned_residual", pin(protocol["residual_module"]))
        prepare = load_module("frozen_prepare_conversion", pin(protocol["prepare_script"]))
        source_audit = read(pin(Path(protocol["source_root"])/"completion_audit.json"))
        bundle_audit = read(pin(Path(protocol["original_bundle"])/"bundle_audit.json"))
        if not source_audit.get("passed") or not bundle_audit.get("passed"):
            raise ValueError("Existing raw collection and original bundle audits must pass")
        original = index_unique(read(pin(Path(protocol["original_bundle"])/"paired_scenarios.json")))
        new_pair = index_unique(read(pin(Path(protocol["current_bundle"])/"paired_scenarios.json")))
        pred_payload = read(pin(Path(protocol["current_bundle"])/"learned_predictions.json"))
        predictions = index_unique(pred_payload["records"])
        if pred_payload["checkpoint_sha256"] != protocol["attributor_sha256"]:
            raise ValueError("Current predictions must identify the frozen approved attributor")
        if digest(pin(pred_payload["checkpoint"])) != protocol["attributor_sha256"]:
            raise ValueError("Attributor changed")
        if len(original)!=192 or set(original)!=set(new_pair) or set(original)!=set(predictions):
            raise ValueError("Exact same complete 192-pool source identities required")
        model = CandidateUtilityModel.from_record(read(pin(protocol["v8_model"])))
        old_report = read(pin(protocol["v8_report"]))
        expected = {tuple(r["key"]):r for split in ("train","validation") for r in old_report[split]["rows"]}
        source_rows = {}
        for path in sorted((Path(protocol["source_root"])/"outcomes").glob("*/results.json")):
            for raw in read(pin(path)):
                if key(raw) in source_rows:
                    raise ValueError("Duplicated raw outcome pool")
                source_rows[key(raw)] = (raw,path.parent)
        if set(source_rows)!=set(original):
            raise ValueError("Raw source membership mismatch")
        pools, provenance = [], []
        for k in sorted(original):
            old = original[k]; new = new_pair[k]
            raw, root = source_rows[k]
            split = old["split"]
            if split!=new["split"] or split!=("train" if k[1]<=5 else "val"):
                raise ValueError("Inherited group split changed")
            if old["candidate_visual_evidence"]!=new["candidate_visual_evidence"]:
                raise ValueError("Frozen before/predicted visual features changed")
            budget = raw["budget_contract"]
            if budget["official_episode_total_steps"]!=400 or budget["remaining_policy_steps"]!=400-budget["branch_endpoint_policy_step"]:
                raise ValueError("Budget contract mismatch")
            if [c["candidate_id"] for c in raw["candidates"]]!=[0,1,2,3]:
                raise ValueError("Complete original K4 candidates required")
            # Pin actual BEFORE and PREDICTIONS and bind copied training inputs to
            # their raw file sources. Terminal state remains only in hashed Y.
            for field in ("current_primary_path","current_wrist_path"):
                pin(root/raw[field])
            plans, visuals, effects, y, values = [], [], [], [], []
            for c, visual in zip(raw["candidates"],old["candidate_visual_evidence"]):
                cid = c["candidate_id"]
                for field in ("actions_path","predicted_primary_path","predicted_wrist_path"):
                    p = pin(root/c[field])
                    for bundle in (protocol["original_bundle"],protocol["current_bundle"]):
                        copied = Path(bundle)/"candidate_outcomes"/root.name/c[field]
                        if digest(pin(copied)) != pins[str(p)]:
                            raise ValueError("Copied candidate input source mismatch")
                a = np.load(root/c["actions_path"],allow_pickle=False)
                plans.append(a); values.append(float(c["value"]))
                visuals.append(CandidateVisualEvidence(**visual))
                effects.append(parse_candidate_effect(cid,a,visual_evidence=visuals[-1]))
                yy = labels(c["outcome"])
                if yy["steps"]>budget["remaining_policy_steps"]:
                    raise ValueError("Executed result exceeded frozen budget")
                y.append(yy)
            original_attr = prepare.attribution_output(clean_attribution(old["learned_attribution"],.64),f"original_v8:{k}")
            frozen_selection = select_hard_gate_value_tiebreak(mode=SelectorMode.LEARNED_HARD_GATE,
                belief=initial_belief(k[0],bindings=prepare.TASK_BINDINGS),attribution=original_attr,
                block_index=3,candidate_actions=plans,official_values=values,visual_evidence=visuals,utility_model=model)
            baseline_id = frozen_selection.selected_candidate_id
            baseline_fallback = frozen_selection.fallback
            if baseline_id!=expected[k]["selected_candidate"] or baseline_fallback!=expected[k]["fallback"]:
                raise ValueError(f"Original V8 exact policy replay failed: {k}")
            fallback_raw = raw.get("fallback_outcomes",{}).get(baseline_fallback)
            fallback_y = labels(fallback_raw) if fallback_raw is not None else None
            base_y = fallback_y if baseline_id is None else y[baseline_id]
            if base_y is None or base_y["success"]!=expected[k]["selected_success"]:
                raise ValueError("Original V8 terminal replay failed")
            base_scores = np.array([model.score(__import__('wam_reranking.candidate_utility',fromlist=['candidate_utility_features']).candidate_utility_features(e,v,original_attr))
                                   for e,v in zip(effects,values)])
            current = prepare.attribution_output(clean_attribution(predictions[k],pred_payload["unknown_threshold"]),f"current_frozen:{k}")
            p = {"key":k,"split":split,"plans":plans,"effects":effects,"attr":current,
                "relation":prepare.FROZEN_RELATIONS[(k[0],k[2])].relation,
                "prior":initial_belief(k[0],bindings=prepare.TASK_BINDINGS),
                "decisions":list(frozen_selection.decisions),"base_scores":base_scores,
                "baseline_id":baseline_id,"value_id":int(np.argmax(values)),"y":y,
                "fallback_y":fallback_y,"fallback":baseline_fallback,
                "generation_calls":int(raw["candidate_generation_wam_calls"])}
            pools.append(p)
        if {(p['key'][:2]) for p in pools if p['split']=='train'} & {(p['key'][:2]) for p in pools if p['split']=='val'}:
            raise ValueError("Task-state group leakage")
        donors = donor_map(pools); indexed = {p['key']:p for p in pools}
        for p in pools:
            controls = {}
            full_traces = []
            for mode in ('full','masked','shuffled','no_dag'):
                attr = indexed[donors[p['key']]]['attr'] if mode=='shuffled' else p['attr']
                matrix = []
                for e,a in zip(p['effects'],p['plans']):
                    x, trace = api.conditioned_features(prior=p['prior'],attribution=attr,effect=e,
                            actions=a,relation=p['relation'],no_dag=mode=='no_dag')
                    matrix.append(np.zeros_like(x) if mode=='masked' else x)
                    if mode=='full':full_traces.append(trace)
                controls[mode]=np.asarray(matrix)
            p['controls']=controls;p['x']=controls['full']
            provenance.append({'key':list(p['key']),'split':p['split'],'donor_key':list(donors[p['key']]),
                'candidate_traces':full_traces,'feature_hashes':{m:hashlib.sha256(x.tobytes()).hexdigest() for m,x in controls.items()}})
        train=[p for p in pools if p['split']=='train'];val=[p for p in pools if p['split']=='val']
        zero=api.TerminalResidual(np.ones(len(api.FEATURE_NAMES)),np.zeros(len(api.FEATURE_NAMES)),np.ones(len(api.FEATURE_NAMES)),0.)
        replay=evaluate(pools,zero,'full',api,model.switch_margin)
        if any(r['selected_candidate']!=r['v8_candidate'] for r in replay['rows']):
            raise ValueError("Zero residual does not preserve exact original V8")
        # Only X and its interface metadata are exported in this artifact.
        write('inference_inputs.json',{'schema':api.SCHEMA,'feature_names':list(api.FEATURE_NAMES),
            'pools':[{'key':list(p['key']),'split':p['split'],'controls':{m:x.tolist() for m,x in p['controls'].items()},
                      'base_scores':p['base_scores'].tolist(),'accepted':[d.candidate_id for d in p['decisions'] if d.accepted]} for p in pools]})
        audit={'passed':True,'exact_original_v8_replay':True,'zero_residual_replay_exact':True,
            'train_pools':len(train),'val_pools':len(val),'candidate_count':sum(len(p['y']) for p in pools),
            'all_failed_pools_preserved':True,'task_state_leakage':False,'future_evidence_in_x':False,
            'original_gates_shared_all_arms':True,'current_frozen_attributor_not_retrained':True,
            'recovery_probability_head_trained':False,'consumed_validation_not_independent':True,
            'ablation_scope':'new soft score channel only; original V8 gates/history/features retained',
            'active_candidate_difference_train_pools':sum(np.any(np.ptp(p['x'],axis=0)>1e-10) for p in train),
            'no_dag_changed_pools':sum(not np.array_equal(p['controls']['full'],p['controls']['no_dag']) for p in pools),
            'shuffled_changed_pools':sum(not np.array_equal(p['controls']['full'],p['controls']['shuffled']) for p in pools)}
        write('input_audit.json',audit);write('feature_provenance.json',provenance);write('source_sha256.json',pins)
        if audit['train_pools']!=144 or audit['val_pools']!=48 or audit['candidate_count']!=768:
            raise ValueError("Formal membership count mismatch")
        status('training',input_audit_passed=True)
        fit,fit_report=api.fit_terminal_pairs(train,**protocol['fit'])
        calibration=[]
        for gain in api.GAIN_GRID:
            r=evaluate(train,replace(fit,gain=gain),'full',api,model.switch_margin)
            calibration.append({'gain':gain,**{k:r[k] for k in ('selected_successes','v8_successes','value_successes','harms_vs_v8','harms_vs_value','unobserved_fallbacks')},
                'common_v8_steps':r['common_success_vs_v8']['steps_delta_sum'],'common_v8_calls':r['common_success_vs_v8']['calls_delta_sum']})
        feasible=[r for r in calibration if r['harms_vs_v8']==r['harms_vs_value']==r['unobserved_fallbacks']==0 and r['selected_successes']>=r['v8_successes']]
        if not feasible:raise ValueError('Even zero residual violates frozen baseline contract')
        best=min(feasible,key=lambda r:(-r['selected_successes'],r['common_v8_steps'],r['common_v8_calls'],r['gain']))
        final=replace(fit,gain=best['gain'])
        model_record=final.record();model_record.update({'frozen_v8_sha256':digest(protocol['v8_model']),
            'frozen_attributor_sha256':protocol['attributor_sha256'],'switch_margin_unchanged':model.switch_margin,
            'training_supervision':'actual_terminal_success_then_successful_cost_not_signed_recovery',
            'selection_gain_source':'train_only','chosen_gain':best,'formal_train_pools':144,'formal_val_pools':48})
        write('terminal_residual_model.json',model_record)
        frozen_model_hash=digest(args.output/'terminal_residual_model.json')
        status('frozen_evaluation',model_sha256=frozen_model_hash)
        # Validation is not consulted until the model AND gain are frozen.
        reports={split:{mode:evaluate(items,final,mode,api,model.switch_margin)
                   for mode in ('full','masked','shuffled','no_dag')}
                 for split,items in (('train',train),('validation',val))}
        full_train=reports['train']['full'];full_val=reports['validation']['full']
        gates={'train_noninferior_to_v8':full_train['selected_successes']>=full_train['v8_successes'],
            'val_noninferior_to_v8':full_val['selected_successes']>=full_val['v8_successes'],
            'train_zero_harm_vs_v8_and_value':full_train['harms_vs_v8']==full_train['harms_vs_value']==0,
            'val_zero_harm_vs_v8_and_value':full_val['harms_vs_v8']==full_val['harms_vs_value']==0,
            'all_selected_outcomes_observed':full_train['unobserved_fallbacks']==full_val['unobserved_fallbacks']==0}
        gates['passed']=all(gates.values())
        extra={'train_success_gain_vs_v8':full_train['selected_successes']-full_train['v8_successes'],
               'val_success_gain_vs_v8':full_val['selected_successes']-full_val['v8_successes'],
               'gain_parameter':final.gain,'nonzero_corrections_are_not_causal_gain_proof':True,
               'does_not_establish_entire_attributor_or_dag_necessity':True}
        # Confirm inputs and frozen models were not modified by fitting.
        changed=[path for path,h in pins.items() if digest(path)!=h]
        if changed:raise ValueError(f'Source mutation: {changed[:2]}')
        report={'schema':api.SCHEMA,'input_audit':audit,'fit':fit_report,'train_only_gain_calibration':calibration,
            'frozen_model_sha256':frozen_model_hash,'comparisons':reports,'development_gate':gates,
            'attribution_increment':extra,'scope':'consumed K4 pools; one-step offline selection followed by recorded frozen continuation; NOT new real closed-loop or main experiment',
            'cost_scope':'same 4 pool generation calls; continuation calls separate; risk proxy is not physical safety',
            'no_recovery_probability_training':True,'old_recovery_gate_and_failures_preserved':True}
        write('training_report.json',report)
        write('development_gate_report.json',gates)
        status('completed' if gates['passed'] else 'completed_development_gate_failed',model_sha256=frozen_model_hash,
               development_gate_passed=gates['passed'],attribution_increment=extra)
        paths=sorted(args.output.glob('*.json'))
        (args.output/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in paths),encoding='utf-8')
        print(json.dumps({'phase':'completed','gate':gates,'fit':fit_report,'increment':extra,
            'train':{m:{k:r[k] for k in ('selected_successes','v8_successes','value_successes','oracle_successes','harms_vs_v8','harms_vs_value','switches_vs_v8','common_success_vs_v8')} for m,r in reports['train'].items()},
            'validation':{m:{k:r[k] for k in ('selected_successes','v8_successes','value_successes','oracle_successes','harms_vs_v8','harms_vs_value','switches_vs_v8','common_success_vs_v8')} for m,r in reports['validation'].items()}},indent=2))
    except Exception as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc))
        write('failure.json',{'type':type(exc).__name__,'message':str(exc),'old_sources_modified':False})
        raise


if __name__=='__main__':
    main()
