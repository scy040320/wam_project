"""One bounded PRE-only interface-repair fit; original complete V8 is immutable."""
import argparse
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/root/autodl-tmp/wam_project/cosmos-policy')
OLD_CODE = ROOT / 'research_runs/known_task_full_v8_conditioned_ranker_20261007_v10_v1'
spec = importlib.util.spec_from_file_location('immutable_v10_runtime', OLD_CODE / 'full_v8_runtime.py')
rt = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rt; spec.loader.exec_module(rt)
old = rt.load('wam_reranking.persistent_conditioned_utility', ROOT / 'research_runs/known_task_full_v8_conditioned_ranker_20261008_v10_v2_persistent_evidence_env_resume/persistent_conditioned_utility.py')
api = rt.load('wam_reranking.command_predicate_utility', Path(__file__).with_name('command_predicate_utility.py'))
io = rt.load('immutable_v10_io', OLD_CODE / 'train_terminal_conditioned_residual.py')


def audit_persistent_histories(out, replay, bundle, backbone, residual):
    """No outcomes fit here; replay PRE-decision histories of the consumed V8."""
    counts = dict(journals=0, exact_zero_decisions=0, active_features=0,
                  historical_deficit_blocks=0, historical_normal_blocks=0,
                  normal_active_features=0, no_dag_feature_changes=0)
    for c in replay['checks']:
        name, block = c['scene'], c['block']; task = int(name.split('_')[0][4:])
        arm = rt.V8_OUT / 'scenarios' / name / 'full_repaired'
        record = rt.read(arm / c['journal'])
        folder = arm / f'recovery_pool_{block}'
        if c['journal'].endswith('_rejected.json') or not folder.exists():
            folder = rt.ORIGINAL / 'scenarios' / name / 'shared_root_pool' if block == 3 else arm / f'pool_{block}'
        plans = [np.load(folder / f'candidate_{i}' / 'actions.npy', allow_pickle=False) for i in range(4)]
        values = [float(rt.read(folder / f'candidate_{i}' / 'query.json')['value']) for i in range(4)]
        visuals = [rt.CandidateVisualEvidence(**v) for v in record['candidate_visual_evidence']]
        belief = rt.belief_from_snapshot(record['belief_before']); start = len(belief.history)
        attribution = rt.attribution_from_snapshot(record['attribution'])
        relation = bundle.FROZEN_RELATIONS[(task, 0)].relation
        effects, kw, reference, scores, _ = rt.build_context(belief, attribution, plans, values, visuals, relation, block, backbone, residual)
        matrices, traces = [], []
        for no_dag in (False, True):
            vectors, ts = [], []
            for e, a in zip(effects, plans):
                x, t = api.conditioned_features(prior=belief, attribution=attribution, effect=e, actions=a,
                    relation=relation, block_index=block, no_dag=no_dag, belief_already_updated=True, history_start=start)
                vectors.append(x); ts.append(t)
            matrices.append(np.array(vectors)); traces.append(ts)
        chosen, new_scores, _ = api.select_conditioned(**kw, traces=traces[0], deltas=np.zeros(4))
        if chosen != reference or new_scores != scores: raise ValueError('New wrapper fails exact historical V8 zero identity')
        historical = any(v['historical'] for t in traces[0] for v in t['deficits'].values())
        active = bool(np.any(matrices[0]))
        normal = attribution.projected_cause.value == 'normal'
        counts['journals'] += 1; counts['exact_zero_decisions'] += 1
        counts['active_features'] += active; counts['historical_deficit_blocks'] += historical
        counts['historical_normal_blocks'] += historical and normal
        counts['normal_active_features'] += active and normal
        counts['no_dag_feature_changes'] += not np.array_equal(*matrices)
    counts.update(passed=counts['journals'] == replay['decision_journals'], used_for_training=False,
                  physical_false_and_evidence_unknown_separate=True, future_prediction_writes=0)
    rt.dump(out / 'persistent_history_audit.json', counts)
    return counts


def evaluate(pools, model, mode):
    rows = []
    for p in pools:
        ds = model.deltas(p['controls'][mode])
        chosen, scores, route = api.select_conditioned(**p['kw'], deltas=ds, traces=p['traces'][mode])
        cid = None if chosen is None else chosen.candidate_id
        y = p['fallback_y'] if cid is None else p['y'][cid]
        ref = p['fallback_y'] if p['baseline_id'] is None else p['y'][p['baseline_id']]
        value = p['y'][p['value_id']]
        row = dict(key=list(p['key']), selected_candidate=cid, v8_candidate=p['baseline_id'],
            selected_success=None if y is None else y['success'], v8_success=None if ref is None else ref['success'],
            value_success=value['success'], oracle_success=any(x['success'] for x in p['y']),
            mixed=0 < sum(x['success'] for x in p['y']) < 4, observed=y is not None,
            deltas=ds.tolist(), scores=scores, route=route,
            steps=None if y is None else y['steps'], calls=None if y is None else y['calls'])
        for name, z in (('v8', ref), ('value', value)):
            common = y is not None and z is not None and y['success'] and z['success']
            row[f'common_steps_vs_{name}'] = y['steps'] - z['steps'] if common else None
            row[f'common_calls_vs_{name}'] = y['calls'] - z['calls'] if common else None
        rows.append(row)
    result = dict(pools=len(rows), selected_successes=sum(x['selected_success'] is True for x in rows),
        oracle_successes=sum(x['oracle_success'] for x in rows), mixed_pools=sum(x['mixed'] for x in rows),
        all_failed_pools=sum(not x['oracle_success'] for x in rows), unobserved_fallbacks=sum(not x['observed'] for x in rows),
        switches_vs_v8=sum(x['selected_candidate'] != x['v8_candidate'] for x in rows), rows=rows)
    for name in ('v8', 'value'):
        result[f'{name}_successes'] = sum(x[f'{name}_success'] is True for x in rows)
        result[f'harms_vs_{name}'] = sum(x['selected_success'] is False and x[f'{name}_success'] is True for x in rows)
        result[f'improvements_vs_{name}'] = sum(x['selected_success'] is True and x[f'{name}_success'] is False for x in rows)
        sd = [x[f'common_steps_vs_{name}'] for x in rows if x[f'common_steps_vs_{name}'] is not None]
        cd = [x[f'common_calls_vs_{name}'] for x in rows if x[f'common_calls_vs_{name}'] is not None]
        result[f'common_success_vs_{name}'] = dict(count=len(sd), steps_delta_sum=sum(sd), calls_delta_sum=sum(cd),
            steps_delta_mean=float(np.mean(sd)) if sd else None, calls_delta_mean=float(np.mean(cd)) if cd else None)
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=False)
    def status(phase, **kw): rt.dump(out / 'status.json', dict(phase=phase, time_unix=time.time(), **kw))
    # Predeclare before reading any terminal Y or evaluating validation.
    protocol = dict(schema=api.SCHEMA, fit_count=1, steps=1200, lr=.01, l2=.1,
        cap=api.CAP, minimum_switch_margin=.01, score_unit=old.SCORE_UNIT,
        gain_grid=list(old.GAIN_GRID), gain_calibration='train only; zero harm; lexicographic success/paired cost/smallest gain',
        split='original 144 train / 48 consumed validation', validation_hyperparameter_use=False,
        no_new_queries=True, no_new_actions=True, no_hard_gate_relaxation=True,
        control_scope='new residual only; V8 gates/routing remain frozen',
        command_threshold=api.COMMAND_THRESHOLD, contact_threshold=api.CONTACT_THRESHOLD,
        extra_control='without_command distinguishes measured-rule from learned attribution',
        correction_scope='source-comparable anchor alternatives with achievable fixed-cap score gaps',
        history_training_limit='root pools have no multi-block prehistory; audit persistent histories separately, never invent them',
        recovery_probability_head=False, real_closedloop_launch=False)
    rt.dump(out / 'frozen_repair_protocol.json', protocol)
    try:
        status('immutable_v8_identity_audit')
        replay, bundle = rt.audit_original_decisions(out)
        backbone, residual = rt.frozen_models()
        history_audit = audit_persistent_histories(out, replay, bundle, backbone, residual)
        source = ROOT / 'outputs/candidate_reranking_d21_formal_192_20261001_v7_balanced'
        current = ROOT / 'outputs/candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit/training_bundle'
        if not rt.read(source / 'completion_audit.json').get('passed'): raise ValueError('192 raw audit failed')
        pairs = io.index_unique(rt.read(current / 'paired_scenarios.json'))
        payload = rt.read(current / 'learned_predictions.json'); preds = io.index_unique(payload['records'])
        if payload['checkpoint_sha256'] != '68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec' or rt.sha(payload['checkpoint']) != payload['checkpoint_sha256']:
            raise ValueError('Frozen complete V8 attributor identity failed')
        pins = {str(current / 'paired_scenarios.json'): rt.sha(current / 'paired_scenarios.json'),
                str(current / 'learned_predictions.json'): rt.sha(current / 'learned_predictions.json'),
                payload['checkpoint']: payload['checkpoint_sha256']}
        source_rows = {}
        for path in sorted((source / 'outcomes').glob('*/results.json')):
            pins[str(path)] = rt.sha(path)
            for raw in rt.read(path):
                k = io.key(raw)
                if k in source_rows: raise ValueError('Duplicate raw pool')
                source_rows[k] = (raw, path.parent)
        if len(pairs) != 192 or set(pairs) != set(preds) or set(pairs) != set(source_rows):
            raise ValueError('Pool membership mismatch')
        pools = []
        for k in sorted(pairs):
            row = pairs[k]; raw, folder = source_rows[k]
            if row['split'] != ('train' if k[1] <= 5 else 'val'): raise ValueError('Original split changed')
            if [c['candidate_id'] for c in raw['candidates']] != list(range(4)): raise ValueError('K4 mismatch')
            if raw['budget_contract']['official_episode_total_steps'] != 400 or raw['budget_contract']['remaining_policy_steps'] != 400 - raw['budget_contract']['branch_endpoint_policy_step']:
                raise ValueError('Original budget changed')
            plans, values, labels = [], [], []
            for c in raw['candidates']:
                for field in ('actions_path', 'predicted_primary_path', 'predicted_wrist_path'):
                    path = folder / c[field]; copied = current / 'candidate_outcomes' / folder.name / c[field]
                    pins[str(path)] = rt.sha(path)
                    if rt.sha(copied) != pins[str(path)]: raise ValueError('Prediction/action copy mismatch')
                plans.append(np.load(folder / c['actions_path'], allow_pickle=False)); values.append(float(c['value']))
                yy = io.labels(c['outcome'])
                if yy['steps'] > raw['budget_contract']['remaining_policy_steps']: raise ValueError('Outcome budget violation')
                labels.append(yy)  # Terminal Y only; never passed to feature or admission APIs.
            block = ROOT / raw['evidence_block']; meta_path = block / 'block.json'; meta = rt.read(meta_path)
            requested_path, applied_path = block / 'requested_actions.npy', block / 'applied_actions.npy'
            for path in (meta_path, requested_path, applied_path): pins[str(path)] = rt.sha(path)
            for name, path in (('requested_actions', requested_path), ('applied_actions', applied_path)):
                if meta[name]['sha256'] != pins[str(path)]: raise ValueError('Previous-block command hash mismatch')
            endpoint = raw['budget_contract']['branch_endpoint_policy_step']
            if meta['prediction_target_policy_step'] != endpoint: raise ValueError('Previous command/candidate timing mismatch')
            command = api.command_evidence(requested=np.load(requested_path, allow_pickle=False),
                applied=np.load(applied_path, allow_pickle=False),
                record_available=bool(meta['action_record_available'] and preds[k]['action_record_available']),
                source_block_id=f'libero90_task{k[0]}_block{meta["block_index"]}',
                source_block_index=meta['block_index'], use_block_index=3, policy_step_t=meta['policy_step_t'],
                endpoint_policy_step=endpoint, observation_policy_step=meta['actual_observation_policy_step'],
                executed_steps=meta['executed_steps'], alignment_valid=meta['alignment_valid_t_plus_H'],
                evidence_ids=(pins[str(requested_path)], pins[str(applied_path)]))
            if not command.valid: raise ValueError('Incomplete preceding command record')
            attr = bundle.attribution_output(io.clean_attribution(preds[k], payload['unknown_threshold']), command.source_block_id)
            prior = rt.initial_belief(k[0], block_index=3, bindings=bundle.TASK_BINDINGS)
            before = copy.deepcopy(prior); hstart = len(prior.history)
            relation = bundle.FROZEN_RELATIONS[(k[0], k[2])].relation
            visuals = [rt.CandidateVisualEvidence(**v) for v in row['candidate_visual_evidence']]
            effects, kw, chosen, scores, fallback = rt.build_context(prior, attr, plans, values, visuals, relation, 3, backbone, residual)
            # The stored legacy belief_snapshot is POST-attribution, not a prior.
            # Build the original V8 once and reuse that post-update belief.
            fallback_raw = raw.get('fallback_outcomes', {}).get(fallback)
            pools.append(dict(key=k, split=row['split'], prior=before, current_belief=prior, history_start=hstart,
                attr=attr, command=command, relation=relation, visuals=visuals, plans=plans, values=values, effects=effects,
                kw=kw, decisions=kw['decisions'], base_scores=np.array([scores.get(i, 0.) for i in range(4)]),
                baseline_id=None if chosen is None else chosen.candidate_id, value_id=int(np.argmax(values)),
                y=labels, fallback_y=io.labels(fallback_raw) if fallback_raw is not None else None))
        train = [p for p in pools if p['split'] == 'train']; val = [p for p in pools if p['split'] == 'val']
        if len(train) != 144 or len(val) != 48 or {p['key'][:2] for p in train} & {p['key'][:2] for p in val}:
            raise ValueError('Group isolation/count failed')
        donors = io.donor_map(pools); index = {p['key']: p for p in pools}; provenance = []
        deterministic = True
        for p in pools:
            controls, traces = {}, {}
            for mode in ('full', 'masked', 'shuffled', 'no_dag', 'without_command'):
                attr = index[donors[p['key']]]['attr'] if mode == 'shuffled' else p['attr']
                belief = copy.deepcopy(p['current_belief'])
                if mode == 'shuffled':
                    belief = copy.deepcopy(p['prior'])
                    rt.build_context(belief, attr, p['plans'], p['values'], p['visuals'], p['relation'], 3, backbone, residual)
                xx, tt = [], []
                for e, a in zip(p['effects'], p['plans']):
                    args = dict(prior=belief, attribution=attr, effect=e, actions=a, relation=p['relation'],
                        no_dag=mode == 'no_dag', belief_already_updated=True, history_start=p['history_start'],
                        command=None if mode == 'without_command' else index[donors[p['key']]]['command'] if mode == 'shuffled' else p['command'])
                    x, t = api.conditioned_features(**args); repeat, _ = api.conditioned_features(**args)
                    deterministic &= np.array_equal(x, repeat)
                    xx.append(np.zeros_like(x) if mode == 'masked' else x); tt.append(t)
                controls[mode], traces[mode] = np.array(xx), tt
            p['controls'], p['traces'], p['x'] = controls, traces, controls['full']
            p['comparable'] = api.comparable_ids(decisions=p['decisions'], backbone_features=p['kw']['backbone_features'],
                attribution=p['attr'], traces=traces['full'], reference_id=p['baseline_id'])
            zero, zero_scores, _ = api.select_conditioned(**p['kw'], deltas=np.zeros(4), traces=traces['full'])
            if (None if zero is None else zero.candidate_id) != p['baseline_id']: raise ValueError('Zero residual identity changed')
            if any(abs(zero_scores.get(i, 0.) - p['base_scores'][i]) > 1e-12 for i in range(4)): raise ValueError('Zero scores changed')
            provenance.append(dict(key=list(p['key']), split=p['split'], donor_key=list(donors[p['key']]),
                                   comparable=sorted(p['comparable']), traces=traces['full']))
        pref, pref_audit = old.preference_pairs(train)
        repair_pairs = [(pool, better, worse) for pool, better, worse, weight, margin in pref
            if train[pool]['y'][better]['success'] and not train[pool]['y'][worse]['success']
            and better != train[pool]['baseline_id']]
        pref_audit['corrective_success_pairs'] = len(repair_pairs)
        pref_audit['protective_success_pairs'] = pref_audit['success'] - len(repair_pairs)
        audit = dict(passed=bool(history_audit['passed'] and deterministic and pref_audit['corrective_success_pairs'] > 0), deterministic_features=bool(deterministic),
            train_pools=144, val_pools=48, candidates=768, group_leakage=False, executed_future_in_inputs=False,
            zero_identity_pools=192, original_zero_identity_journals=replay['decision_journals'],
            frozen_attributor_sha256=payload['checkpoint_sha256'], original_v8_gates_unchanged=True,
            train_deployment_comparable_preferences=pref_audit,
            active_train_pools=sum(bool(np.any(np.ptp(p['x'], axis=0) > 1e-10)) for p in train),
            active_val_pools=sum(bool(np.any(np.ptp(p['x'], axis=0) > 1e-10)) for p in val),
            validation_consumed=True, history_not_invented=True, control_scope=protocol['control_scope'])
        rt.dump(out / 'input_audit.json', audit); rt.dump(out / 'feature_provenance.json', provenance)
        rt.dump(out / 'source_sha256.json', pins)
        rt.dump(out / 'inference_inputs.json', dict(feature_names=list(api.FEATURE_NAMES), pools=[
            dict(key=list(p['key']), split=p['split'], controls={m: x.tolist() for m, x in p['controls'].items()}) for p in pools]))
        if not audit['passed']: raise ValueError('Pre-fit activity/valid supervision audit failed; NO FIT')
        status('single_training', input_audit_passed=True, epoch=0, total_steps=protocol['steps'])
        fit, fit_report = old.fit_once(train, steps=protocol['steps'], lr=protocol['lr'], l2=protocol['l2'])
        fit_report.update(corrective_success_pairs=len(repair_pairs),
            protective_success_pairs=pref_audit['protective_success_pairs'])
        calibration = []
        for gain in old.GAIN_GRID:
            r = evaluate(train, replace(fit, gain=gain), 'full')
            calibration.append(dict(gain=gain, success=r['selected_successes'], v8_success=r['v8_successes'],
                harms_v8=r['harms_vs_v8'], harms_value=r['harms_vs_value'], unobserved=r['unobserved_fallbacks'],
                steps=r['common_success_vs_v8']['steps_delta_sum'], calls=r['common_success_vs_v8']['calls_delta_sum']))
        feasible = [r for r in calibration if r['harms_v8'] == r['harms_value'] == r['unobserved'] == 0 and r['success'] >= r['v8_success']]
        if not feasible: raise ValueError('Zero V8 violated admission; no calibration may bypass')
        best = min(feasible, key=lambda r: (-r['success'], r['steps'], r['calls'], r['gain']))
        model = replace(fit, gain=best['gain']); record = model.record()
        record['schema'] = api.SCHEMA
        record['command_threshold'] = api.COMMAND_THRESHOLD
        record['predicate_mask_contract'] = 'source-specific .55 contact / .5 visual; frozen hard gates'
        record['fit_count'] = 1
        record.update(baseline='complete 64-scene frozen V8 strategy/models/routing', calibration='train_only',
                      frozen_attributor_sha256=payload['checkpoint_sha256'], formal_train_pools=144, formal_val_pools=48)
        rt.dump(out / 'conditional_utility_model.json', record); model_hash = rt.sha(out / 'conditional_utility_model.json')
        status('frozen_evaluation', chosen_gain=model.gain, model_sha256=model_hash)
        reports = {split: {mode: evaluate(ps, model, mode) for mode in ('full', 'masked', 'shuffled', 'no_dag', 'without_command')}
                   for split, ps in (('train', train), ('validation', val))}
        # Gain=1 is a declared shadow diagnosis, not an alternate deployed model
        # or another fit. No validation information may change the chosen gain.
        shadows = {split: {mode: evaluate(ps, replace(fit, gain=1.), mode) for mode in ('full', 'masked', 'shuffled', 'no_dag', 'without_command')}
                   for split, ps in (('train', train), ('validation', val))}
        for path, h in pins.items():
            if rt.sha(path) != h: raise ValueError('Immutable input changed')
        gates = {}
        for split in ('train', 'validation'):
            r = reports[split]['full']; gates[split + '_noninferior_v8'] = r['selected_successes'] >= r['v8_successes']
            gates[split + '_zero_harm_v8_value'] = r['harms_vs_v8'] == r['harms_vs_value'] == 0
            gates[split + '_observed'] = r['unobserved_fallbacks'] == 0
        quality = all(gates.values()); tr = reports['train']['full']
        gain_evidence = tr['improvements_vs_v8'] > 0 or (tr['common_success_vs_v8']['steps_delta_sum'] < 0 and tr['common_success_vs_v8']['calls_delta_sum'] <= 0)
        gates.update(base_preservation_pass=quality, additional_train_utility_gain=bool(gain_evidence),
                     real_closedloop_recommendable=bool(quality and gain_evidence), passed=bool(quality and gain_evidence))
        report = dict(protocol=protocol, input_audit=audit, fit=fit_report, chosen_gain=model.gain,
            train_only_gain_calibration=calibration, frozen_model_sha256=model_hash, comparisons=reports,
            unit_gain_shadow_diagnosis=shadows, development_gate=gates,
            scope='192 consumed single-choice counterfactual pools, NOT 64-scene closed-loop results',
            no_gate3_claim=True, overall_attributor_or_dag_necessity_not_proven=True)
        report['learned_attribution_increment_not_established_by_command_gain'] = True
        rt.dump(out / 'training_report.json', report); rt.dump(out / 'development_gate_report.json', gates)
        status('completed', development_gate_passed=gates['passed'], chosen_gain=model.gain,
               real_closedloop_launched=False, model_sha256=model_hash)
        (out / 'SHA256SUMS.txt').write_text(''.join(f'{rt.sha(p)}  {p.name}\n' for p in sorted(out.glob('*.json'))))
        print(rt.json.dumps(dict(gates=gates, fit=fit_report, gain=model.gain,
            comparisons={s: {m: {k: v for k, v in r.items() if k != 'rows'} for m, r in rs.items()} for s, rs in reports.items()},
            shadow={s: {m: {k: v for k, v in r.items() if k != 'rows'} for m, r in rs.items()} for s, rs in shadows.items()}), default=rt.scalar))
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error)); raise


if __name__ == '__main__': main()
