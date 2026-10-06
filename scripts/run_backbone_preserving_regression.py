"""Consumed-scene regression; frozen adapters plus one named policy change."""
import argparse
import json
from pathlib import Path
import traceback

import run_known_task_ablation as execution
from wam_reranking.epistemic_backbone_policy import select_epistemic_backbone
from wam_reranking.closedloop_metrics import execution_cost_summary


def verify(root, protocol):
    for relative, digest in protocol['frozen_inputs'].items():
        if execution.file_sha(root / relative) != digest:
            raise RuntimeError('Frozen repair input changed: ' + relative)


def audit(out, protocol):
    rows = []
    for lane in (0, 1):
        part = json.loads((out / f'lane{lane}_results.json').read_text())
        if len(part) != 16: raise RuntimeError('Lane membership mismatch')
        rows.extend(part)
    expected = {(int(t), s, c) for t, states in protocol['task_states'].items()
                for s in states for c in protocol['conditions']}
    if len(rows) != 32 or {(r['task'], r['state'], r['condition']) for r in rows} != expected:
        raise RuntimeError('Missing or duplicated scene')
    for row in rows:
        if set(row['methods']) != set(execution.METHODS): raise RuntimeError('Missing method')
        for method in row['methods'].values():
            if not all(method[k] for k in ('budget_ok', 'outcome_observed', 'real_multiblock_updates')):
                raise RuntimeError('Invalid observed outcome or budget')
    summary = execution.summarize(rows)
    full = summary['methods']['full']
    historical = json.loads((Path(protocol['parent_output_absolute']) / 'results.json').read_text())
    old = {(r['task'], r['state'], r['condition']): r for r in historical}
    controls = ('value_only', 'candidate_only')
    control_outcome_changes = sum(
        bool(r['methods'][m]['success']) != bool(old[(r['task'],r['state'],r['condition'])]['methods'][m]['success'])
        for r in rows for m in controls)
    old_full_harms = sum(old[(r['task'],r['state'],r['condition'])]['methods']['full']['success']
                         and not r['methods']['full']['success'] for r in rows)
    costs = execution_cost_summary(rows, execution.METHODS, full_method='full')
    target = (full['success'] >= 18 and full['harms_vs_value_only'] == 0
              and full['harms_vs_candidate_only'] == 0 and old_full_harms == 0)
    record = {'quality_passed': True, 'development_repair_target_passed': target,
        'summary': summary, 'costs': costs, 'lost_parent_full_successes': old_full_harms,
        'reexecuted_control_success_changes': control_outcome_changes,
        'independent_validation_claim': False, 'original_gate3_pass_claim': False,
        'attribution_necessity_not_demonstrated_by_baseline_recovery': True,
        'no_automatic_policy_iteration': True}
    execution.dump(out / 'completion_audit.json', record)
    execution.dump(out / 'results.json', rows)
    execution.dump(out / 'execution_cost_report.json', costs)
    execution.dump(out / 'per_task_report.json', {str(t): execution.summarize([r for r in rows if r['task'] == t]) for t in protocol['tasks']})
    execution.dump(out / 'per_condition_report.json', {c: execution.summarize([r for r in rows if r['condition'] == c]) for c in protocol['conditions']})
    execution.dump(out / 'status.json', {'phase':'complete', 'scenarios_completed':32,
        'quality_passed':True, 'development_repair_target_passed':target,
        'original_gate3_pass_claim':False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--lane', type=int, choices=(0, 1))
    ap.add_argument('--audit', action='store_true')
    args = ap.parse_args()
    root, out = args.project_root.resolve(), args.output.resolve()
    protocol = json.loads((out / 'frozen_protocol.json').read_text())
    verify(root, protocol)
    if args.audit: return audit(out, protocol)
    if args.lane is None: raise ValueError('Lane is required')
    # Only this new process changes its selector binding; no frozen parent
    # code file, checkpoint, graph, threshold or cause gain is rewritten.
    execution.select_evidence_arbitration = select_epistemic_backbone
    paired = execution.load_execution(root, protocol, out)
    exp = execution.experiment_class(paired, root, out)(args.lane, protocol)
    rows = []
    try:
        scenes = protocol['lane_scenes'][str(args.lane)]
        current_task = None
        for task, state, condition in scenes:
            if task != current_task:
                exp.set_task(task); current_task = task
            row = exp.scenario(state, condition)
            row['role'] = 'consumed_development_regression'
            rows.append(row)
            execution.dump(out / f'lane{args.lane}_results.json', rows)
            paired.legacy.status(args.lane, 'regression', scenarios_completed=len(rows),
                summary=execution.summarize(rows))
        paired.legacy.status(args.lane, 'complete', scenarios_completed=len(rows))
    except Exception as error:
        paired.legacy.status(args.lane, 'failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        if exp.env is not None: exp.env.close()


if __name__ == '__main__': main()
