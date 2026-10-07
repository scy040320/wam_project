"""Original TRAIN only. No model fit, new query, action or validation tuning."""
import os
from pathlib import Path
ROOT = Path('/root/autodl-tmp/wam_project/cosmos-policy')
for k, v in {'LIBERO_CONFIG_PATH': str(ROOT / '.libero'), 'HF_HUB_OFFLINE': '1',
             'TRANSFORMERS_OFFLINE': '1', 'MUJOCO_GL': 'egl'}.items(): os.environ.setdefault(k, v)
import argparse
import copy
import importlib.util
import inspect
import sys
import time
from collections import Counter
import numpy as np

RUNTIME = ROOT / 'research_runs/known_task_full_v8_conditioned_ranker_20261008_v10_v2_persistent_evidence_env_resume'
spec = importlib.util.spec_from_file_location('readonly_persistent_runtime', RUNTIME / 'train_persistent_conditioned_utility.py')
run = importlib.util.module_from_spec(spec); sys.modules[spec.name] = run; spec.loader.exec_module(run)
rt, io, old = run.rt, run.io, run.api
api = rt.load('wam_reranking.command_predicate_utility', Path(__file__).with_name('command_predicate_utility.py'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output; out.mkdir(parents=True, exist_ok=False)
    status = lambda phase, **kw: rt.dump(out / 'status.json', dict(phase=phase, time_unix=time.time(), **kw))
    protocol = dict(schema=api.SCHEMA, original_train_only=True, train_pools=144, candidates=576,
        training_fit_count=0, new_queries=0, new_action_steps=0, validation_evaluation=False,
        command_threshold=api.COMMAND_THRESHOLD, contact_threshold=api.CONTACT_THRESHOLD,
        residual_visual_threshold=api.VISUAL_THRESHOLD, fixed_cap=api.CAP, minimum_switch_margin=.01,
        original_hard_gates_unchanged=True, correction_scope='frozen accepted alternatives only',
        physical_recovery_head=False, complete_v8_reference='64-scene frozen strategy/models/routing',
        model_and_raw_outputs_immutable=True, command_measurement_is_not_contact_truth=True,
        forecast_writes_current_belief=False)
    rt.dump(out / 'frozen_protocol.json', protocol)
    try:
        status('source_and_feature_audit')
        current = ROOT / 'outputs/candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit/training_bundle'
        source = ROOT / 'outputs/candidate_reranking_d21_formal_192_20261001_v7_balanced'
        if not rt.read(source / 'completion_audit.json')['passed']: raise ValueError('Raw audit failed')
        paired = io.index_unique(rt.read(current / 'paired_scenarios.json'))
        payload = rt.read(current / 'learned_predictions.json'); predictions = io.index_unique(payload['records'])
        expected = '68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec'
        if payload['checkpoint_sha256'] != expected or rt.sha(payload['checkpoint']) != expected: raise ValueError('Attributor identity failed')
        converter = rt.frozen.paired.legacy.load('readonly_source_adapter', rt.frozen.paired.legacy.parent.REF / 'prepare_d21_training_bundle_v7.py')
        backbone, residual = rt.frozen_models()
        pins = {str(p): rt.sha(p) for p in (current / 'paired_scenarios.json', current / 'learned_predictions.json',
            Path(payload['checkpoint']), Path(inspect.getsourcefile(rt.parse_candidate_effect)),
            Path(inspect.getsourcefile(converter.attribution_output)),
            rt.V8_OUT / 'candidate_only_model.json', rt.V8_OUT / 'evidence_residual_model.json',
            Path(inspect.getsourcefile(old)), Path(__file__).with_name('command_predicate_utility.py'), Path(__file__))}
        raws = {}
        for p in sorted((source / 'outcomes').glob('*/results.json')):
            pins[str(p)] = rt.sha(p)
            for row in rt.read(p):
                key = io.key(row)
                if key in raws: raise ValueError('Duplicate raw pool')
                raws[key] = (row, p.parent)
        if len(paired) != 192 or set(paired) != set(raws) or set(paired) != set(predictions): raise ValueError('Pool IDs mismatch')
        pools, rows, counts = [], [], Counter()
        train_keys = {k for k, r in paired.items() if r['split'] == 'train'}
        if len(train_keys) != 144 or any(k[1] > 5 for k in train_keys): raise ValueError('Original TRAIN split changed')
        for key in sorted(train_keys):
            row = paired[key]; raw, folder = raws[key]
            # Raw results do not carry split; the frozen training bundle does.
            # Verify BOTH its assignment and the raw source identity/state.
            if (row['split'] != 'train' or io.key(raw) != key or raw['state'] not in range(6)
                or [c['candidate_id'] for c in raw['candidates']] != list(range(4))):
                raise ValueError('Membership/K changed')
            block = ROOT / raw['evidence_block']; meta_path = block / 'block.json'; meta = rt.read(meta_path)
            requested, applied = block / 'requested_actions.npy', block / 'applied_actions.npy'
            for p in (meta_path, requested, applied): pins[str(p)] = rt.sha(p)
            for name, p in (('requested_actions', requested), ('applied_actions', applied)):
                if meta[name]['sha256'] != pins[str(p)]: raise ValueError('Command metadata hash mismatch')
            use_block, endpoint = 3, raw['budget_contract']['branch_endpoint_policy_step']
            if meta['prediction_target_policy_step'] != endpoint: raise ValueError('Command endpoint differs from candidate start')
            command = api.command_evidence(requested=np.load(requested, allow_pickle=False), applied=np.load(applied, allow_pickle=False),
                record_available=bool(meta['action_record_available'] and predictions[key]['action_record_available']),
                source_block_id=f'libero90_task{key[0]}_block{meta["block_index"]}', source_block_index=meta['block_index'],
                use_block_index=use_block, policy_step_t=meta['policy_step_t'], endpoint_policy_step=endpoint,
                observation_policy_step=meta['actual_observation_policy_step'], executed_steps=meta['executed_steps'],
                alignment_valid=meta['alignment_valid_t_plus_H'], evidence_ids=(pins[str(requested)], pins[str(applied)]))
            if not command.valid: raise ValueError(f'Invalid preceding command record: {key}')
            # Everything up to labels below is PRE-only. No intervention condition,
            # simulator state, success, steps or continuation cost enters APIs.
            attr = converter.attribution_output(io.clean_attribution(predictions[key], payload['unknown_threshold']), command.source_block_id)
            attr_before = copy.deepcopy(attr)
            plans, values = [], []
            for c in raw['candidates']:
                for field in ('actions_path', 'predicted_primary_path', 'predicted_wrist_path'):
                    p = folder / c[field]; pins[str(p)] = rt.sha(p)
                    if rt.sha(current / 'candidate_outcomes' / folder.name / c[field]) != pins[str(p)]: raise ValueError('Frozen copy differs')
                plans.append(np.load(folder / c['actions_path'], allow_pickle=False)); values.append(float(c['value']))
            visuals = [rt.CandidateVisualEvidence(**v) for v in row['candidate_visual_evidence']]
            belief = rt.initial_belief(key[0], block_index=3, bindings=converter.TASK_BINDINGS)
            start = len(belief.history); relation = converter.FROZEN_RELATIONS[(key[0], key[2])].relation
            effects, kw, reference, scores, fallback = rt.build_context(belief, attr, plans, values, visuals, relation, 3, backbone, residual)
            snapshot = copy.deepcopy(belief.snapshot())
            variants = {}
            for mode in ('old', 'masks_only', 'full', 'no_dag'):
                xs, ts = [], []
                for effect, plan in zip(effects, plans):
                    args = dict(prior=belief, attribution=attr, effect=effect, actions=plan, relation=relation,
                        block_index=3, belief_already_updated=True, history_start=start, no_dag=mode == 'no_dag')
                    if mode == 'old': x, t = old.conditioned_features(**args)
                    else:
                        args['command'] = None if mode == 'masks_only' else command
                        x, t = api.conditioned_features(**args)
                        repeat, _ = api.conditioned_features(**args)
                        if not np.array_equal(x, repeat): raise ValueError('Features nondeterministic')
                    xs.append(x); ts.append(t)
                x = np.array(xs); rid = None if reference is None else reference.candidate_id
                comparable = (old if mode == 'old' else api).comparable_ids(decisions=kw['decisions'],
                    backbone_features=kw['backbone_features'], attribution=attr, traces=ts, reference_id=rid)
                variants[mode] = dict(x=x, traces=ts, comparable=comparable)
            zero, zero_scores, _ = api.select_conditioned(**kw, deltas=np.zeros(4), traces=variants['full']['traces'])
            if zero != reference or zero_scores != scores: raise ValueError('Complete V8 zero identity failed')
            if snapshot != belief.snapshot() or attr_before != attr: raise ValueError('Frozen inputs mutated')
            counts['exact_zero_identity_pools'] += 1
            counts['reliable_command_pools'] += command.valid
            counts['measured_command_deviation_pools'] += command.deviated
            counts['hard_command_negative_neural_execution_pools'] += command.deviated and attr.factor_probs['execution_contact_consistent'] > .5
            for i, t in enumerate(variants['full']['traces']):
                old_trace = variants['old']['traces'][i]
                counts['contact_source_open_candidates'] += t['predicate_usable']['grasped']
                counts['relation_source_open_candidates'] += t['predicate_usable']['place_ready']
                counts['contact_restored_from_broad_mask_candidates'] += (not old_trace['visual_usable'] and t['predicate_usable']['grasped'])
            # Read terminal Y strictly AFTER constructing masks/deficits/admission.
            y = [io.labels(c['outcome']) for c in raw['candidates']]
            pool = dict(key=key, split='train', y=y, baseline_id=rid, kw=kw,
                base_scores=np.array([scores.get(i, 0.) for i in range(4)]), variants=variants)
            pools.append(pool)
            rows.append(dict(key=list(key), baseline_id=rid, source_command=command.record(),
                raw_neural_execution_consistent=attr.factor_probs['execution_contact_consistent'],
                baseline_success=None if rid is None else y[rid]['success'],
                successful_candidates=[i for i, label in enumerate(y) if label['success']],
                accepted_candidates=[d.candidate_id for d in kw['decisions'] if d.accepted],
                variants={m: dict(comparable=sorted(v['comparable']), active=bool(np.any(np.ptp(v['x'], axis=0) > 1e-10)),
                    traces=v['traces']) for m, v in variants.items()}))
        if len(pools) != 144: raise ValueError('TRAIN count incorrect')
        status('train_preference_audit', completed_train_pools=144, new_queries=0, fits=0)
        supervision = {}
        for mode in ('old', 'masks_only', 'full', 'no_dag'):
            ps = [{**p, 'x': p['variants'][mode]['x'], 'comparable': p['variants'][mode]['comparable']} for p in pools]
            pairs, audit = old.preference_pairs(ps)
            repairs, protections = [], []
            for index, better, worse, weight, margin in pairs:
                p = ps[index]
                if p['y'][better]['success'] and not p['y'][worse]['success']:
                    record = dict(key=list(p['key']), better=better, worse=worse,
                        reference=p['baseline_id'], needed_correction=float(p['base_scores'][worse]-p['base_scores'][better]+margin),
                        feature_l1_difference=float(np.abs(p['x'][better]-p['x'][worse]).sum()))
                    (protections if better == p['baseline_id'] else repairs).append(record)
            supervision[mode] = dict(preference_counts=audit, repair_pairs=repairs, protection_pairs=protections,
                repair_pair_count=len(repairs), protection_pair_count=len(protections),
                distinct_repair_pools=len({tuple(x['key']) for x in repairs}),
                active_pools=sum(bool(np.any(np.ptp(p['x'], axis=0) > 1e-10)) for p in ps))
        misses = []
        for p, r in zip(pools, rows):
            rid = p['baseline_id']
            if rid is None or p['y'][rid]['success'] or not any(v['success'] for v in p['y']): continue
            alternatives = []
            for cid in r['successful_candidates']:
                v = p['variants']['full']; needed = float(p['base_scores'][rid]-p['base_scores'][cid]+max(.01, backbone.switch_margin))
                alternatives.append(dict(candidate=cid, frozen_gate_accepted=cid in r['accepted_candidates'],
                    comparable=cid in v['comparable'], features_differ=bool(np.any(np.abs(v['x'][cid]-v['x'][rid])>1e-10)),
                    needed_correction=needed, fixed_cap_sufficient=needed < api.CAP))
            misses.append({**r, 'alternatives': alternatives})
        for p, h in pins.items():
            if rt.sha(p) != h: raise ValueError('Immutable source changed')
        report = dict(protocol=protocol, counts=dict(counts), supervision=supervision,
            missed_train_pools=misses, original_train_pools=144, all_failed_train_pools=sum(not any(v['success'] for v in p['y']) for p in pools),
            sources_unchanged=True, source_count=len(pins), raw_model_probabilities_unchanged=True,
            input_audit_passed=True, corrective_supervision_exists=supervision['full']['repair_pair_count'] > 0,
            training_started=False, actual_selection_improvement_proven=False,
            claim='feature/admission readiness only, NOT a trained or closed-loop benefit')
        rt.dump(out/'readiness_report.json', report); rt.dump(out/'train_feature_provenance.json', rows)
        rt.dump(out/'source_sha256.json', pins)
        status('completed', corrective_supervision_exists=report['corrective_supervision_exists'], fits=0, new_queries=0, new_action_steps=0)
        (out/'SHA256SUMS.txt').write_text(''.join(f'{rt.sha(p)}  {p.name}\n' for p in sorted(out.glob('*.json'))))
        print('READINESS_SUMMARY='+rt.json.dumps({k:v for k,v in report.items() if k!='missed_train_pools'}, default=rt.scalar))
        print('MISSED_TRAIN_SUMMARY='+rt.json.dumps([{k:v for k,v in p.items() if k not in ('variants',)} for p in misses], default=rt.scalar))
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error)); raise


if __name__ == '__main__': main()
