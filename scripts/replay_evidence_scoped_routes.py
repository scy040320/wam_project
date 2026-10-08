"""CPU-only, frozen-parameter PRE replay; no fitting, queries or actions.

Root-pool outcomes are labels for the recorded frozen continuation only.
Consumed trajectory decisions are audited separately, never fabricated as a
new closed-loop rollout. Existing data, code and reports remain immutable.
"""
import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path('/root/autodl-tmp/wam_project/cosmos-policy')
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'research_runs/known_task_full_v8_conditioned_ranker_20261007_v10_v1'
spec = importlib.util.spec_from_file_location('evidence_replay_frozen_runtime', OLD / 'full_v8_runtime.py')
rt = importlib.util.module_from_spec(spec); sys.modules[spec.name] = rt; spec.loader.exec_module(rt)
old = rt.load('wam_reranking.persistent_conditioned_utility', ROOT / 'research_runs/known_task_full_v8_conditioned_ranker_20261008_v10_v2_persistent_evidence_env_resume/persistent_conditioned_utility.py')
command_api = rt.load('wam_reranking.command_predicate_utility', ROOT / 'research_runs/known_task_full_v8_priority_utility_20261008_v10_v4/command_predicate_utility.py')
priority = rt.load('wam_reranking.priority_conditioned_utility', ROOT / 'research_runs/known_task_full_v8_priority_utility_20261008_v10_v4/priority_conditioned_utility.py')
io = rt.load('evidence_replay_frozen_io', OLD / 'train_terminal_conditioned_residual.py')
chain = rt.load('wam_reranking.whole_chain_attribution', ROOT / 'research_runs/known_task_whole_chain_closedloop_20261008_v3_episode_resume/whole_chain_attribution.py')
bundle = rt.frozen.paired.legacy.load('evidence_replay_immutable_bundle', rt.frozen.paired.legacy.parent.REF / 'prepare_d21_training_bundle_v7.py')
native = rt.load('wam_reranking.native_before_adapter', HERE / 'native_before_adapter.py')
routing = rt.load('wam_reranking.attribution_routing_contract', HERE / 'attribution_routing_contract.py')
policy = rt.load('wam_reranking.evidence_scoped_selection', HERE / 'evidence_scoped_selection.py')
FIT = ROOT / 'outputs/known_task_full_v8_priority_utility_20261008_v10_v4_training'
SOURCE = ROOT / 'outputs/candidate_reranking_d21_formal_192_20261001_v7_balanced'
CACHE = ROOT / 'outputs/candidate_reranking_d21_frozen_d18v8_development_20261004_v2_unobserved_explicit/training_bundle'
MODES = policy.MODES
MODEL_SHA = '225b911358b15806d4a51bafbb8be5bb6012d7ec444800b5e6f34ed188b922ef'
ATTRIBUTOR_SHA = '68b2b638f73c959fb170d39caecad63572427f7716f78fe4c9459ccaf8012bec'


def plain(value):
    if isinstance(value, Enum): return value.value
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def frozen_models(pin=None):
    pin = pin or (lambda p: Path(p))
    backbone, residual = rt.frozen_models()
    for name in ('candidate_only_model.json', 'evidence_residual_model.json'):
        pin(rt.V8_OUT / name)
    path = pin(FIT / 'conditional_utility_model.json')
    if rt.sha(path) != MODEL_SHA: raise ValueError('Frozen conditional identity changed')
    record = rt.read(path)
    if record['gain'] != .875: raise ValueError('Frozen gain changed')
    model = old.ConditionalUtility(np.asarray(record['scale']), np.asarray(record['weights']),
        np.asarray(record['support']), record['gain'])
    return backbone, residual, model


def raw_input(record, threshold=.64):
    # NO labels/task/condition/outcome/private physics in selector inputs.
    result = io.clean_attribution(record, threshold)
    for key in ('normal_prob', 'factor_threshold'):
        if key in record: result[key] = record[key]
    return result


def repaired_variants(kw, attribution, donor):
    results, audits = {}, {}
    for mode in MODES:
        controlled = attribution
        if mode in ('cause_only', 'quality_only', 'shuffled'):
            controlled, audit = routing.replace_attribution_channels(attribution, donor,
                mode='all' if mode == 'shuffled' else mode,
                source_block_id=kw['command'].source_block_id, availability=kw['availability'])
            audits[mode] = audit
        args = {**kw, 'attribution': controlled, 'mode': mode}
        result = policy.select_scoped_chain(**args)
        if plain(result) != plain(policy.select_scoped_chain(**args)):
            raise ValueError('Nondeterministic repaired replay')
        results[mode] = result
    # Remove learned information across the entire new chain, not just scores.
    masked = policy.select_scoped_chain(**{**kw, 'attribution': donor, 'mode': 'ranker_command'})
    if plain(masked) != plain(results['ranker_command']):
        raise ValueError('Learned output leaked into no-learned arm')
    return results, audits


def compare(a, b):
    facts = [p for p in a['live_belief']['facts'] if a['live_belief']['facts'][p] != b['live_belief']['facts'][p]]
    gates = [i for i in range(len(a['decisions'])) if
        (a['decisions'][i].accepted, a['decisions'][i].rejection_reasons) !=
        (b['decisions'][i].accepted, b['decisions'][i].rejection_reasons)]
    score_ids = [i for i in range(len(a['decisions'])) if abs(a['scores'].get(i, 0) - b['scores'].get(i, 0)) > 1e-12
                 or (i in a['scores']) != (i in b['scores'])]
    return dict(belief_changed=facts, gate_changed=gates, scores_changed=score_ids,
        reference_changed=a['reference_id'] != b['reference_id'],
        choice_changed=(a['selected_id'], plain(a['fallback'])) != (b['selected_id'], plain(b['fallback'])))


def summary(rows, mode, reference='legacy_full'):
    common = [r for r in rows if r[mode]['success'] is True and r[reference]['success'] is True]
    return dict(pools=len(rows), observed=sum(r[mode]['observed'] for r in rows),
        successes=sum(r[mode]['success'] is True for r in rows),
        fallback_unobserved=sum(not r[mode]['observed'] for r in rows),
        choice_changes=sum(r[mode]['selected_id'] != r[reference]['selected_id'] for r in rows),
        improvements=sum(r[mode]['success'] is True and r[reference]['success'] is False for r in rows),
        harms=sum(r[mode]['success'] is False and r[reference]['success'] is True for r in rows),
        unobserved_vs_successful_reference=sum(not r[mode]['observed'] and r[reference]['success'] is True for r in rows),
        common_success=dict(count=len(common), steps_delta_sum=sum(r[mode]['steps'] - r[reference]['steps'] for r in common),
            calls_delta_sum=sum(r[mode]['calls'] - r[reference]['calls'] for r in common)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    out = ap.parse_args().output; out.mkdir(parents=True, exist_ok=False)
    pins = {}
    def pin(path):
        path = Path(path); pins[str(path)] = rt.sha(path); return path
    def status(phase, **kw): rt.dump(out / 'status.json', dict(phase=phase, time_unix=time.time(), **kw))
    protocol = dict(schema=policy.SCHEMA, original_pools=192, train=144, consumed_val=48,
        modes=list(MODES), new_queries=0, new_actions=0, fits=0, gain=.875,
        classifier_unknown_threshold=.64, factor_threshold=.5, command_threshold=1e-6,
        hard_predicate_thresholds_unchanged=True, all_failed_pools_retained=True,
        old_code_and_models_unchanged=True, no_scenario_specific_exceptions=True,
        current_pose_certificates='none invented from aggregate observability or candidate predictions',
        reference='same learned candidate backbone inside actual hard-feasible set; no UNKNOWN value override',
        partial_controls='same fixed same-split donor; cause and quality independently replaced; actual command retained',
        scope='one root selection + recorded frozen official continuation; NOT new closed-loop performance')
    rt.dump(out / 'protocol.json', protocol)
    try:
        status('input_audit')
        for module in (rt, old, command_api, priority, io, chain, bundle, native, routing, policy): pin(module.__file__)
        backbone, residual, model = frozen_models(pin)
        pairs = io.index_unique(rt.read(pin(CACHE / 'paired_scenarios.json')))
        payload = rt.read(pin(CACHE / 'learned_predictions.json')); preds = io.index_unique(payload['records'])
        if payload['checkpoint_sha256'] != ATTRIBUTOR_SHA or rt.sha(pin(payload['checkpoint'])) != ATTRIBUTOR_SHA:
            raise ValueError('Frozen attributor changed')
        if not rt.read(pin(SOURCE / 'completion_audit.json'))['passed']: raise ValueError('Source audit failed')
        previous = rt.read(pin(FIT / 'training_report.json'))
        oldrows = {tuple(r['key']): r for split in ('train', 'validation') for r in previous['comparisons'][split]['full']['rows']}
        rawrows = {}
        for path in sorted((SOURCE / 'outcomes').glob('*/results.json')):
            for row in rt.read(pin(path)):
                key = io.key(row)
                if key in rawrows: raise ValueError('Duplicate source pool')
                rawrows[key] = row, path.parent
        if set(rawrows) != set(pairs) or set(pairs) != set(preds) or len(pairs) != 192:
            raise ValueError('Identity/count mismatch')
        donors = priority.cross_group_donors([dict(key=k, split=pairs[k]['split']) for k in sorted(pairs)])
        attrs, oldattrs, prepared, contracts = {}, {}, {}, {}
        for k in sorted(pairs):
            pair, (row, folder) = pairs[k], rawrows[k]
            if pair['split'] != ('train' if k[1] <= 5 else 'val'): raise ValueError('Split changed')
            if pairs[donors[k]]['split'] != pair['split'] or donors[k][:2] == k[:2]: raise ValueError('Donor leakage')
            if [c['candidate_id'] for c in row['candidates']] != list(range(4)): raise ValueError('Pool membership changed')
            budget = row['budget_contract']
            if budget['official_episode_total_steps'] != 400 or budget['remaining_policy_steps'] != 400 - budget['branch_endpoint_policy_step']:
                raise ValueError('Budget changed')
            plans, values = [], []
            for c in row['candidates']:
                for name in ('actions_path', 'predicted_primary_path', 'predicted_wrist_path'):
                    p = pin(folder / c[name])
                    if rt.sha(CACHE / 'candidate_outcomes' / folder.name / c[name]) != pins[str(p)]:
                        raise ValueError('Candidate cache differs')
                plans.append(np.load(folder / c['actions_path'], allow_pickle=False)); values.append(c['value'])
            block = ROOT / row['evidence_block']; meta = rt.read(pin(block / 'block.json'))
            requested, applied = pin(block / 'requested_actions.npy'), pin(block / 'applied_actions.npy')
            if meta['requested_actions']['sha256'] != pins[str(requested)] or meta['applied_actions']['sha256'] != pins[str(applied)]:
                raise ValueError('Command hashes changed')
            command = command_api.command_evidence(requested=np.load(requested, allow_pickle=False),
                applied=np.load(applied, allow_pickle=False), record_available=bool(meta['action_record_available'] and preds[k]['action_record_available']),
                source_block_id=f'libero90_task{k[0]}_block{meta["block_index"]}', source_block_index=meta['block_index'],
                use_block_index=3, policy_step_t=meta['policy_step_t'], endpoint_policy_step=budget['branch_endpoint_policy_step'],
                observation_policy_step=meta['actual_observation_policy_step'], executed_steps=meta['executed_steps'],
                alignment_valid=meta['alignment_valid_t_plus_H'], evidence_ids=(pins[str(requested)], pins[str(applied)]))
            if not command.valid or meta['prediction_target_policy_step'] != budget['branch_endpoint_policy_step']:
                raise ValueError('Command/observation alignment failed')
            availability = (bool(preds[k]['primary_available']), bool(preds[k]['wrist_available']), command.valid)
            oldattrs[k] = bundle.attribution_output(io.clean_attribution(preds[k], payload['unknown_threshold']), command.source_block_id)
            attrs[k], contracts[k] = routing.decode_frozen_attribution(raw_input(preds[k]), command.source_block_id, availability)
            contracts[k]['legacy_projected_cause'] = oldattrs[k].projected_cause.value
            contracts[k]['legacy_confidence'] = oldattrs[k].confidence
            effects = [rt.parse_candidate_effect(i, a, visual_evidence=rt.CandidateVisualEvidence(**v))
                       for i, (a, v) in enumerate(zip(plans, pair['candidate_visual_evidence'], strict=True))]
            prepared[k] = dict(prior=rt.initial_belief(k[0], block_index=3, bindings=bundle.TASK_BINDINGS), effects=effects,
                plans=plans, values=values, relation=bundle.FROZEN_RELATIONS[(k[0], k[2])].relation,
                block_index=3, command=command, availability=availability, backbone=backbone, frozen_residual=residual, conditional_model=model)
        rows, journals, changes = [], [], []
        for index, k in enumerate(sorted(pairs)):
            kw, (raw, _) = prepared[k], rawrows[k]
            legacy = chain.select_chain(**kw, learned_attribution=oldattrs[k], mode='full')
            expected = oldrows[k]
            if legacy['selected_id'] != expected['selected_candidate'] or any(abs(legacy['scores'][int(i)]-s) > 1e-10 for i,s in expected['scores'].items()):
                raise ValueError(f'Original frozen deployment identity mismatch {k}')
            variants, controls = repaired_variants(kw, attrs[k], attrs[donors[k]])
            variant_compare = {m: compare(r, variants['full']) for m,r in variants.items()}
            changes.append(dict(key=list(k), split=pairs[k]['split'], donor_key=list(donors[k]),
                legacy_vs_repaired=compare(variants['full'], legacy), controls_vs_full=variant_compare,
                contract=contracts[k], control_audits=controls))
            # Outcomes joined after all selector calls and invariance checks.
            ys = [io.labels(c['outcome']) for c in raw['candidates']]
            row = dict(key=list(k), split=pairs[k]['split'], oracle_success=any(y['success'] for y in ys),
                       mixed=0 < sum(y['success'] for y in ys) < 4)
            for mode, result in {'legacy_full':legacy, **variants}.items():
                cid = result['selected_id']; y = None if cid is None else ys[cid]
                row[mode] = dict(selected_id=cid, observed=y is not None, success=None if y is None else y['success'],
                    steps=None if y is None else y['steps'], calls=None if y is None else y['calls'])
            rows.append(row); journals.append(dict(key=list(k), variants=plain(variants), legacy_full=plain(legacy)))
            if (index+1) % 24 == 0: status('root_replay', completed=index+1, total=192)
        for path, digest in pins.items():
            if rt.sha(path) != digest: raise ValueError('Source mutated during audit')
        summary_rows = {s:{m:summary([r for r in rows if r['split']==s],m) for m in ('legacy_full',*MODES)} for s in ('train','val')}
        report = dict(passed=True, protocol=protocol, counts=dict(pools=192,train=144,val=48,
            mixed_pools=sum(r['mixed'] for r in rows), all_failed_pools=sum(not r['oracle_success'] for r in rows),
            original_identity_matches=192, repaired_choice_changes=sum(c['legacy_vs_repaired']['choice_changed'] for c in changes),
            legacy_double_projection=sum(c['contract']['legacy_projected_cause'] != c['contract']['frozen_final_cause'] for c in changes)),
            comparisons=summary_rows, rows=rows, links=changes, current_certificates_synthesized=0,
            no_future_inputs=True, frozen_sources_unchanged=True, group_leakage=False,
            fits_started=0, queries_started=0, actions_executed=0, no_new_closedloop_claim=True)
        rt.dump(out/'root_replay_report.json',report);rt.dump(out/'decision_journals.json',journals);rt.dump(out/'source_sha256.json',pins)
        status('completed',passed=True,counts=report['counts']);print(plain(dict(counts=report['counts'],comparisons=summary_rows)))
    except Exception as error:
        status('failed',error=repr(error));raise


if __name__ == '__main__': main()
