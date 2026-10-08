"""One predeclared terminal-utility fit on the audited original PRE pools.

This does not run Cosmos, infer a physical witness, change a classifier, or
overwrite V8. AFTER here means candidate FORECAST, never actual-after RGB.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path('/root/autodl-tmp/wam_project/cosmos-policy')
HERE = Path(__file__).resolve().parent
REPLAY = ROOT / 'outputs/known_task_evidence_interface_recheck_20261008_v4_final_boundaries_roots'
HELPER = ROOT / ('research_runs/known_task_predicate_scoped_replay_20261008_v3_preserve_history/'
                 'replay_evidence_scoped_routes.py')
MODES = ('full', 'ranker_command', 'learned_no_dag', 'cause_only', 'quality_only', 'shuffled',
         'masked_soft_only')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def load_helper():
    spec = importlib.util.spec_from_file_location('immutable_soft_fit_root_helper', HELPER)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    def status(phase, **kw): dump(out/'status.json', dict(phase=phase, time_unix=time.time(), **kw))
    protocol = dict(schema='continuous_attribution_terminal_utility_trial_v2_full_competition',
        fit_count=1, steps=1200, lr=.01, l2=.1, success_steps=900,
        gain=1., pairwise_cap=.1, switch_margin=.01,
        gain_search=False, validation_hyperparameter_use=False,
        backbone='unchanged latest source-scoped PRE selector; frozen V8 remains archived',
        splits='original 144 train / 48 consumed val; task/state groups unchanged',
        features='continuous attribution x forecast AFTER-only effects x source-quality; no cached BEFORE/delta/motion',
        supervised_target='terminal selection success; conditional successful cost; NOT recovery probability',
        masks='source availability is hard; forecast quality is a continuous soft weight',
        modes=list(MODES), all_failed_and_unobserved_pools_retained=True,
        physical_fact_writes=0, physical_witnesses_generated=0,
        hard_gate_threshold_changes=0, new_queries=0, new_actions=0,
        real_closedloop_launched=False, gate3_claim=False)
    protocol.update(previous_single_fit_preserved=
        'outputs/known_task_continuous_evidence_utility_20261009_v3_training',
        supporting_diagnosis_split='train only: successful candidate beat anchor but lost to another failed competitor',
        corrective_target='success beats every hard-feasible failed competitor AND actual deployment choice succeeds',
        configuration_changed=False, hard_thresholds_changed=False,
        total_new_training_attempts_this_repair=2)
    dump(out/'protocol.json', protocol)
    try:
        status('source_and_PRE_input_audit')
        h = load_helper()
        from wam_reranking.contracts import CandidateDecision
        soft = h.rt.load('wam_reranking.soft_attribution_utility', HERE/'soft_attribution_utility.py')
        contracts = h.routing
        pins = read(REPLAY/'source_sha256.json') | read(REPLAY/'addition_sha256.json')
        for p in (Path(__file__), HERE/'soft_attribution_utility.py', HELPER,
                  REPLAY/'root_replay_report.json', REPLAY/'decision_journals.json'):
            pins[str(p)] = sha(p)
        for p, digest in pins.items():
            if sha(p) != digest: raise ValueError('Immutable source changed before fit: '+p)
        report, journals = read(REPLAY/'root_replay_report.json'), read(REPLAY/'decision_journals.json')
        if not report['passed'] or report['counts']['original_identity_matches'] != 192:
            raise ValueError('Previous original-192 audit failed')
        pair_path = h.CACHE/'paired_scenarios.json'
        pins[str(pair_path)] = sha(pair_path)
        pairs = h.io.index_unique(read(pair_path))
        js = {tuple(j['key']): j for j in journals}
        if len(js) != 192 or set(js) != set(pairs): raise ValueError('PRE identity/count mismatch')
        raw_index = {}
        for path in sorted((h.SOURCE/'outcomes').glob('*/results.json')):
            pins[str(path)] = sha(path)
            for raw in read(path):
                key = h.io.key(raw)
                if key in raw_index: raise ValueError('Duplicate actual outcomes row')
                raw_index[key] = raw, path.parent
        if set(raw_index) != set(pairs): raise ValueError('Label identity mismatch')
        prepared, inference_rows = [], []
        for key in sorted(pairs):
            split = pairs[key]['split']
            if split != ('train' if key[1] <= 5 else 'val'): raise ValueError('Original split changed')
            relation = h.bundle.FROZEN_RELATIONS[(key[0], key[2])].relation
            raw, folder = raw_index[key]
            if [c['candidate_id'] for c in raw['candidates']] != list(range(4)):
                raise ValueError('Original K4 membership changed')
            # These are candidate forecasts/action plans, authenticated in the
            # prior audit. No outcome is passed to the following APIs.
            effects = [h.rt.parse_candidate_effect(i,
                np.load(folder/c['actions_path'], allow_pickle=False),
                visual_evidence=h.rt.CandidateVisualEvidence(**v))
                for i, (c, v) in enumerate(zip(raw['candidates'], pairs[key]['candidate_visual_evidence'], strict=True))]
            variants, matrices, audits = {}, {}, {}
            for mode in MODES:
                base_mode = 'full' if mode == 'masked_soft_only' else mode
                variant = js[key]['variants'][base_mode]
                a = variant['attribution']
                # Duck-compatible controlled output preserves raw frozen route,
                # all probabilities and explicit partial-control conflicts.
                if base_mode == 'ranker_command':
                    # The audited no-learned carrier has confidence=0 and a
                    # uniform class distribution. It is deliberately NOT a
                    # frozen classifier prediction. Never force its confidence
                    # through the genuine-prediction converter or relabel it.
                    q = a['evidence_quality']
                    attr = h.chain.absent_carrier(a['source_block_id'],
                        (q['primary_reliable'], q['wrist_reliable'], q['execution_reliable']))
                    if h.plain(attr) != a:
                        raise ValueError('No-learned carrier reconstruction changed its original content')
                else:
                    attr = contracts.ControlledAttribution(
                        a['factor_probs'], a['class_probs'], h.rt.CoarseCause(a['projected_cause']),
                        a['confidence'], a['entropy'], h.rt.EvidenceQuality(**a['evidence_quality']),
                        a['source_block_id'], {k:h.rt.TriValue(v) for k,v in a['factor_states'].items()},
                        a['factor_confidences'], tuple(a.get('semantic_conflicts', ())))
                masked = mode in ('ranker_command', 'masked_soft_only')
                x, audit = soft.soft_features(attr, effects, relation,
                    no_dag=mode=='learned_no_dag', masked=masked,
                    belief_snapshot=variant['live_belief'])
                repeat, _ = soft.soft_features(attr, effects, relation,
                    no_dag=mode=='learned_no_dag', masked=masked,
                    belief_snapshot=variant['live_belief'])
                if not np.array_equal(x, repeat): raise ValueError('Nondeterministic PRE feature construction')
                decisions = tuple(CandidateDecision(**{**d,
                    'rejection_reasons':tuple(d['rejection_reasons'])}) for d in variant['decisions'])
                if [d.candidate_id for d in decisions] != list(range(4)):
                    raise ValueError('Gate membership changed')
                score = np.array([variant['scores'].get(str(i), 0.) for i in range(4)])
                matrices[mode], audits[mode] = x, audit
                variants[mode] = dict(split=split, key=key, x=x, decisions=decisions,
                    accepted=np.array([d.accepted for d in decisions]), base_scores=score,
                    official_values=np.array([c['value'] for c in raw['candidates']]),
                    reference_id=variant['selected_id'], value_id=int(np.argmax([c['value'] for c in raw['candidates']])),
                    margin=.01, fallback_outcome=None)
            # Terminal supervision joins only AFTER every PRE feature/gate.
            yy = [h.io.labels(c['outcome']) for c in raw['candidates']]
            for v in variants.values(): v['outcomes'] = yy
            prepared.append(dict(key=key, split=split, variants=variants))
            inference_rows.append(dict(key=list(key), split=split,
                matrices={m:x.tolist() for m,x in matrices.items()}, audits=audits,
                forecast_features_only=True, terminal_labels_in_features=False))
        train = [p['variants']['full'] for p in prepared if p['split']=='train']
        val = [p for p in prepared if p['split']=='val']
        if len(train)!=144 or len(val)!=48: raise ValueError('Original train/val counts changed')
        if {p['key'][:2] for p in train} & {p['key'][:2] for p in val}: raise ValueError('Group leakage')
        audit = dict(passed=True, train=144, val=48, before_fields_consumed=0,
            new_physical_facts=0, deterministic=True, all_failed_pools=report['counts']['all_failed_pools'],
            source_sha256_count=len(pins), models_and_raw_routes_unchanged=True,
            fallback_unknown_is_failure=False,
            masked_control_keeps_shared_candidate_effect_capacity=True,
            original_candidate_dependent_before_quarantined_from_new_channel=True)
        dump(out/'input_audit.json', audit)
        dump(out/'inference_inputs.json', dict(feature_names=list(soft.FEATURE_NAMES), rows=inference_rows))
        dump(out/'source_sha256.json', pins)
        status('single_fixed_training', input_audit_passed=True)
        model, fit_audit = soft.fit_once(train)
        dump(out/'soft_utility_model.json', model.to_record())
        status('frozen_evaluation', model_sha256=sha(out/'soft_utility_model.json'))
        comparisons = {}
        for split in ('train','val'):
            comparisons[split] = {}
            for mode in MODES:
                rows = []
                for p in prepared:
                    if p['split'] != split: continue
                    v = p['variants'][mode]
                    result = soft.select_soft_utility(v['decisions'], v['base_scores'], v['official_values'],
                        v['reference_id'], v['x'], model, margin=v['margin'])
                    cid, rid = result['selected_id'], v['reference_id']
                    y = None if cid is None else v['outcomes'][cid]
                    baseline_y = None if rid is None else v['outcomes'][rid]
                    value_y = v['outcomes'][v['value_id']]
                    rows.append(dict(key=list(p['key']), selected_id=cid, reference_id=rid,
                        observed=y is not None, success=None if y is None else y['success'],
                        reference_success=None if baseline_y is None else baseline_y['success'],
                        value_success=value_y['success'], deltas=result['deltas'],
                        scores=result['scores'], reason=result['reason'],
                        common_steps_delta=(y['steps']-baseline_y['steps']) if y and baseline_y and y['success'] and baseline_y['success'] else None,
                        common_calls_delta=(y['calls']-baseline_y['calls']) if y and baseline_y and y['success'] and baseline_y['success'] else None))
                common = [r for r in rows if r['common_steps_delta'] is not None]
                comparisons[split][mode] = dict(pools=len(rows), observed=sum(r['observed'] for r in rows),
                    success=sum(r['success'] is True for r in rows),
                    baseline_success=sum(r['reference_success'] is True for r in rows),
                    unobserved=sum(not r['observed'] for r in rows),
                    changes=sum(r['selected_id']!=r['reference_id'] for r in rows),
                    improvements=sum(r['success'] is True and r['reference_success'] is False for r in rows),
                    harms=sum(r['success'] is False and r['reference_success'] is True for r in rows),
                    common_success=len(common), common_steps_delta_sum=sum(r['common_steps_delta'] for r in common),
                    common_calls_delta_sum=sum(r['common_calls_delta'] for r in common), rows=rows)
        for p,digest in pins.items():
            if sha(p)!=digest: raise ValueError('Immutable source changed during fit: '+p)
        full = comparisons['train']['full']
        gates = dict(train_zero_added_success_harm=full['harms']==0,
            train_has_extra_success=full['improvements']>0,
            val_zero_added_success_harm=comparisons['val']['full']['harms']==0,
            unresolved_current_witness_exits_remain=True, new_closedloop=False, gate3=False)
        dump(out/'training_report.json', dict(protocol=protocol, fit=fit_audit,
            input_audit=audit, comparisons=comparisons, checks=gates,
            scope='consumed original PRE pools and recorded official continuation; NOT new real closedloop',
            learned_attribution_benefit_requires_whole_chain_controls=True))
        status('completed', fit_count=1, checks=gates,
            summary={s:{m:{k:v for k,v in r.items() if k!='rows'} for m,r in modes.items()} for s,modes in comparisons.items()})
        print(json.dumps(read(out/'status.json')))
    except Exception as error:
        status('stopped_no_further_fit', error_type=type(error).__name__, error=str(error),
               no_auto_retrain=True, new_queries=0, new_actions=0)
        raise


if __name__=='__main__': main()
