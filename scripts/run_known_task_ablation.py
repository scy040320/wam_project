"""Execute a frozen four-arm multiblock ablation from one source per group.

The external execution contract is an audited Cosmos/LIBERO adapter. It is
not shipped as a second competing policy implementation. Pickles are trusted
project-generated snapshots only. No post-execution outcomes enter selection.
"""
import argparse
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image

from wam_reranking import initial_belief, localize_candidate_visual_evidence, parse_candidate_effect
from wam_reranking.candidate_utility import select_with_utility
from wam_reranking.closedloop_metrics import execution_cost_summary
from wam_reranking.evidence_arbitration import select_evidence_arbitration
from wam_reranking.evidence_residual import candidate_backbone_features, cause_residual_features, typed_gate_effect
from wam_reranking.mechanism_ablation import METHODS, prepare_mechanism_decisions
from wam_reranking.reranker import select_candidate
from wam_reranking.shared_query_cache import file_sha


def dump(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(record, indent=2, default=str) + '\n')
    temp.replace(path)


def summarize(rows):
    result = {'scenarios': len(rows), 'methods': {}}
    for method in METHODS:
        out = {'success': sum(bool(r['methods'][method]['success']) for r in rows),
               'fallback_episodes': sum(bool(r['methods'][method]['fallback_events']) for r in rows)}
        for control in METHODS:
            out['gains_vs_' + control] = sum(r['methods'][method]['success'] and not r['methods'][control]['success'] for r in rows)
            out['harms_vs_' + control] = sum(not r['methods'][method]['success'] and r['methods'][control]['success'] for r in rows)
        result['methods'][method] = out
    return result


def load_execution(root, protocol, out):
    path = root / protocol['execution_contract']
    spec = importlib.util.spec_from_file_location('frozen_matched_execution', path)
    paired = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = paired
    spec.loader.exec_module(paired)
    for mod in (paired, paired.repair, paired.legacy):
        mod.OUT, mod.VERSION = out, protocol['version']
    return paired


def experiment_class(paired, root, out):
    class Experiment(paired.PairedExperiment):
        def execute(self, *args, **kwargs):
            if shutil.disk_usage(root).free < 15 * 1024**3:
                raise RuntimeError('Frozen disk floor: less than 15 GiB')
            return super().execute(*args, **kwargs)

        @paired.legacy.frozen_precision()
        def choose_mode(self, pool, obs, record, belief, block, mode):
            if self.arm == 'candidate_only':
                # Exact frozen candidate-only implementation, not retrained.
                return paired.legacy.Experiment.choose_mode(self, pool, obs, record, belief, block, 'candidate_only')
            if self.arm not in ('no_dependency', 'full'):
                raise RuntimeError('Unexpected attribution arm')
            if self.last_action_stage is not None:
                belief.task_stage = self.last_action_stage
            started = time.monotonic()
            visuals = [localize_candidate_visual_evidence(localizer=self.attributor.localizer,
                current_primary=Image.fromarray(np.asarray(obs['primary_image']).astype(np.uint8)),
                current_wrist=Image.fromarray(np.asarray(obs['wrist_image']).astype(np.uint8)),
                predicted_primary=Image.fromarray(np.asarray(c['future_image_predictions']['future_image']).astype(np.uint8)),
                predicted_wrist=Image.fromarray(np.asarray(c['future_image_predictions']['future_wrist_image']).astype(np.uint8)),
                target_prompt=self.relation.subject, anchor_prompt=self.relation.anchor,
                relation=self.relation.relation) for c in pool]
            attr = self.bundle.attribution_output(record, f'libero90_task{self.task_id}_block{block-1}')
            before = belief.snapshot()
            values = [float(c['value_prediction']) for c in pool]
            effects = [parse_candidate_effect(i, c['actions'], visual_evidence=v)
                       for i, (c, v) in enumerate(zip(pool, visuals, strict=True))]
            decisions = prepare_mechanism_decisions(method=self.arm, belief=belief,
                attribution=attr, block_index=block, effects=effects, values=values,
                relation=self.relation.relation, backbone=self.candidate_model)
            xs = {i: candidate_backbone_features(e, values[i]) for i, e in enumerate(effects)}
            zs = {i: cause_residual_features(typed_gate_effect(e, self.relation.relation), attr, decisions[i])
                  for i, e in enumerate(effects)}
            chosen, scores = select_evidence_arbitration(decisions=decisions, backbone_features=xs,
                cause_features=zs, backbone=self.candidate_model, residual=self.residual, attribution=attr)
            fallback = None
            if chosen is None: _, fallback = select_candidate(decisions, attr)
            else: self.last_action_stage = effects[chosen.candidate_id].stage
            controls = [replace(d, accepted=True, rejection_reasons=(),
                        components={'dependency_risk': 0., 'uncertainty': 0.}) for d in decisions]
            candidate_anchor, _ = select_with_utility(controls, xs, self.candidate_model)
            return None if chosen is None else chosen.candidate_id, fallback, {
                'mode': self.arm, 'attribution': asdict(attr), 'belief_before': before,
                'selection': {'decisions': [asdict(d) for d in decisions], 'belief_snapshot': belief.snapshot()},
                'selected_candidate_id': None if chosen is None else chosen.candidate_id,
                'selected_route_components': None if chosen is None else dict(chosen.components),
                'candidate_only_anchor_at_same_observation': candidate_anchor.candidate_id,
                'official_value_anchor_at_same_observation': int(np.argmax(values)),
                'candidate_visual_evidence': [asdict(v) for v in visuals], 'scores': scores,
                'cause_corrections': {i: self.residual.score(z) for i, z in zs.items()},
                'selection_compute_seconds': time.monotonic() - started,
                'graph_edges_enabled': self.arm == 'full', 'no_gt_selection_inputs': True}

        def common_source(self, state):
            # Inherited method generates only one prefix/query/snapshot per
            # task/state. Its OUT is redirected to this experiment only.
            return paired.legacy.Experiment.common_source(self, state)

        def scenario(self, state, condition):
            path = out / 'scenarios' / f'task{self.task_id}_state{state}_{condition}'
            if path.exists(): raise RuntimeError('No overwrite or scientific retry')
            payload, common = self.common_source(state)
            path.mkdir(parents=True, exist_ok=False)
            prefix = copy.deepcopy(payload['prefix'])
            dump(path / 'common_source_reference.json', common)

            def early(reason):
                row = {'task': self.task_id, 'state': state, 'condition': condition,
                       'early_terminal': True, 'intervention_not_reached': True, 'reason': reason,
                       'methods': {m: {'mode': m, 'success': True, 'budget_ok': True,
                           'total_policy_steps': sum(x['steps'] for x in prefix),
                           'wam_calls_after_fork': 0, 'common_prefix_wam_calls': len(prefix),
                           'executed_steps_after_fork': 0, 'fallback_events': [], 'outcome_observed': True,
                           'real_multiblock_updates': True, 'selection_updates': 0} for m in METHODS}}
                dump(path / 'scenario.json', row)
                return row

            if payload.get('early_terminal'): return early(payload['reason'])
            source, query_obs, action = payload['snapshot'], payload['query_obs'], payload['source_result']
            with (path / 'pre_intervention_runtime_snapshot.pkl').open('wb') as f: pickle.dump(source, f)
            clean = []
            for arm in ('clean_a', 'clean_b'):
                self.restore(source)
                end, done, n, _, _ = self.execute(action, 16, path / arm, query_obs, 2)
                if n != 16 or done:
                    prefix.append({'block': 2, 'steps': n, 'seconds': payload['source_query_seconds']})
                    return early('source_block_terminal')
                clean.append({'observation': end, 'sim': np.r_[self.env.sim.data.qpos.copy(), self.env.sim.data.qvel.copy()]})
            qc = paired.legacy.parent.compare_pair(*clean)
            dump(path / 'clean_pair_audit.json', qc)
            if not qc['passed']: raise RuntimeError('Frozen clean A/B pairing audit failed')
            self.restore(source)
            variant = {'clean': 'clean', 'visual_occlusion': 'primary_center_mask',
                       'object_shift': 'target_pose_shift', 'execution_contact_deviation': 'cartesian_action_deviation_4_steps'}[condition]
            wrapper = self.base.InterventionEnv(self.env, {'name': condition,
                'label': 'normal' if condition == 'clean' else condition, 'variant': variant})
            wrapper.step_count = 42
            if condition != 'clean': wrapper.arm(block_index=2, block_start=42, offset=4,
                duration=4 if condition == 'execution_contact_deviation' else None)
            end, done, n, episode, _ = self.execute(action, 16, path / 'source_block', query_obs, 2, wrapper)
            dump(path / 'offline_intervention.json', {'event': wrapper.event, 'never_deployment_input': True})
            if n != 16 or done:
                prefix.append({'block': 2, 'steps': n, 'seconds': payload['source_query_seconds']})
                return early('intervention_source_block_terminal')
            if condition != 'clean' and wrapper.event is None: raise RuntimeError('Intervention not executed')
            snapshot = self.fork.capture_runtime_snapshot(self.env)
            with (path / 'root_runtime_snapshot.pkl').open('wb') as f: pickle.dump(snapshot, f)
            record = self.attributor.infer(episode)
            obs = copy.deepcopy(end)
            pool, seconds = self.pool(obs, state, 3, 0, path / 'shared_root_pool')
            order_index = (self.protocol['conditions'].index(condition) + self.protocol['task_states'][str(self.task_id)].index(state)) % len(METHODS)
            order = METHODS[order_index:] + METHODS[:order_index]
            outcomes = {}
            for arm in order:
                self.arm, self.last_action_stage = arm, None
                paired.legacy.status(self.lane, 'executing_arm', task=self.task_id, state=state,
                                     condition=condition, arm=arm, arms_completed=len(outcomes))
                mode = arm if arm in ('value_only', 'candidate_only') else 'full'
                begun = time.monotonic()
                result = self.rollout(snapshot, copy.deepcopy(obs), copy.deepcopy(pool), copy.deepcopy(record),
                                      state, path / arm, mode, root_seconds=seconds)
                result.update(mode=arm, rollout_wall_seconds=time.monotonic() - begun,
                              all_methods_reexecuted=True)
                outcomes[arm] = result
                dump(path / 'partial_methods.json', outcomes)
            row = {'task': self.task_id, 'state': state, 'condition': condition,
                   'early_terminal': False, 'methods': outcomes, 'qc_passed': True,
                   'root_snapshot_sha256': file_sha(path / 'root_runtime_snapshot.pkl'),
                   'root_observation_sha256': self.base.obs_hash(obs), 'actual_execution_order': order,
                   'all_methods_share_root_snapshot_and_pool': True,
                   'all_methods_reexecuted': True, 'root_coverage': None,
                   'root_oracle_not_collected': True, 'no_outcome_selected_scenarios': True}
            dump(path / 'scenario.json', row)
            return row
    return Experiment


def audit(out, protocol, pilot):
    rows = []
    for phase in ('pilot',) if pilot else ('pilot', 'full'):
        for lane in (0, 1):
            part = json.loads((out / f'lane{lane}_{phase}_results.json').read_text())
            if len(part) != 8: raise RuntimeError('Fixed per-lane scene count mismatch')
            rows.extend(part)
    expected = 16 if pilot else 32
    identities = {(r['task'], r['state'], r['condition']) for r in rows}
    if len(rows) != expected or len(identities) != expected: raise RuntimeError('Scene identity mismatch')
    for r in rows:
        if set(r['methods']) != set(METHODS): raise RuntimeError('Missing arm')
        for o in r['methods'].values():
            if not all(o[k] for k in ('budget_ok', 'outcome_observed', 'real_multiblock_updates')):
                raise RuntimeError('Unobserved outcome/budget failure')
            if not r['early_terminal'] and (o['total_policy_steps'] != 48 + o['executed_steps_after_fork'] or o['common_prefix_wam_calls'] != 3):
                raise RuntimeError('Execution cost prefix mismatch')
    costs = execution_cost_summary(rows, METHODS, full_method='full')
    sm = summarize(rows)
    report = {'quality_passed': True, 'summary': sm, 'costs': costs,
              'early_terminal_scenarios': sum(r['early_terminal'] for r in rows),
              'new_state_selection_outcome_independent': True,
              'original_gate3_pass_claim': False, 'generalization_to_unseen_tasks_claim': False,
              'no_automatic_policy_iteration': True, 'root_oracle_not_collected': True,
              'scientific_criteria': protocol['scientific_criteria']}
    dump(out / ('pilot_quality_audit.json' if pilot else 'completion_audit.json'), report)
    if not pilot:
        dump(out / 'results.json', rows)
        dump(out / 'execution_cost_report.json', costs)
        dump(out / 'per_task_report.json', {str(t): summarize([r for r in rows if r['task'] == t]) for t in protocol['tasks']})
        dump(out / 'per_condition_report.json', {c: summarize([r for r in rows if r['condition'] == c]) for c in protocol['conditions']})
    dump(out / 'status.json', {'phase': 'pilot_quality_passed' if pilot else 'complete',
         'scenarios_completed': len(rows), 'total_scenarios': 32, 'quality_passed': True,
         'scientific_success_not_required_for_continuation': True, 'original_gate3_pass_claim': False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lane', type=int, choices=(0, 1))
    ap.add_argument('--phase', choices=('pilot', 'full', 'pilot-audit', 'audit'), required=True)
    args = ap.parse_args()
    root, out = args.project_root.resolve(), args.output.resolve()
    protocol = json.loads((out / 'frozen_protocol.json').read_text())
    for name, digest in protocol['frozen_inputs'].items():
        if file_sha(root / name) != digest: raise RuntimeError('Frozen input changed: ' + name)
    if args.phase in ('pilot-audit', 'audit'):
        audit(out, protocol, args.phase == 'pilot-audit')
        return
    if args.lane is None: raise ValueError('Lane is required')
    paired = load_execution(root, protocol, out)
    exp = experiment_class(paired, root, out)(args.lane, protocol)
    rows = []
    try:
        for task in protocol['lane_ownership'][str(args.lane)]:
            exp.set_task(task)
            states = protocol['task_states'][str(task)]
            for state in states[:1] if args.phase == 'pilot' else states[1:]:
                for condition in protocol['conditions']:
                    rows.append(exp.scenario(state, condition))
                    dump(out / f'lane{args.lane}_{args.phase}_results.json', rows)
                    paired.legacy.status(args.lane, args.phase, scenarios_completed=len(rows), summary=summarize(rows))
        paired.legacy.status(args.lane, args.phase + '_complete', scenarios_completed=len(rows))
    except Exception as error:
        paired.legacy.status(args.lane, 'failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if exp.env is not None: exp.env.close()


if __name__ == '__main__': main()
