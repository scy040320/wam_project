"""Read-only recovery-evidence salvage audit; no label inference or promotion.

Historical selected-arm trajectories may support future auxiliary supervision,
but cannot substitute for unexecuted candidates in a paired ranking pool.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import numpy as np

from wam_reranking.shared_query_cache import file_sha

VERSION='known_task_trusted_evidence_arbitration_20261005_v8'
SEALED={'task0':{35,36},'task9':{14,15},'task46':{14,15},'task57':{14,15}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();source=root/'outputs'/VERSION/'scenarios'
    counts=Counter();problems=Counter();rows=[];hashes={};groups=set();cast_max_abs=0.
    def load(path):
        if not path.is_file():raise FileNotFoundError(str(path))
        hashes[str(path.relative_to(root))]=file_sha(path)
        return np.load(path,allow_pickle=False)
    for path in sorted(source.rglob('trajectory.npz')):
        match=re.fullmatch(r'task(\d+)_state(\d+)_(.+)',path.relative_to(source).parts[0])
        if not match:raise ValueError('Unrecognized historical group identity')
        task,state,condition=int(match[1]),int(match[2]),match[3]
        if state in SEALED.get('task'+str(task),set()):
            raise ValueError('32-scene validation state entered auxiliary audit')
        groups.add((task,state));counts['raw_saved_blocks']+=1
        hashes[str(path.relative_to(root))]=file_sha(path)
        failures=[];folder=path.parent;block=folder/'evidence/block'
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={'primary','wrist','requested','applied'}:
                failures.append('trajectory_field_contract')
            n=len(data['requested'])
            if n!=16:failures.append('terminal_short_or_incomplete_block')
            else:
                for view in ('primary','wrist'):
                    frames=data[view]
                    if frames.shape!=(16,256,256,3) or not np.isfinite(frames).all():
                        failures.append(view+'.trajectory_shape')
                    for prefix in ('O_t_','actual_observed_'):
                        if not (block/(prefix+view+'.npy')).is_file():
                            failures.append(view+'.missing_actual_endpoint')
                    if (block/('actual_observed_'+view+'.npy')).is_file():
                        actual=load(block/('actual_observed_'+view+'.npy'))
                        if not np.array_equal(frames[-1],actual):
                            failures.append(view+'.endpoint_time_alignment')
                    if (block/('O_t_'+view+'.npy')).is_file():load(block/('O_t_'+view+'.npy'))
                for name in ('requested','applied'):
                    observed=load(block/(name+'_actions.npy'))
                    if data[name].shape!=(16,7) or not np.array_equal(data[name],observed):
                        failures.append(name+'.command_identity')
                plan=load(block/'planned_actions.npy')
                # Frozen executor iterates np.asarray(result['actions'], np.float32).
                # Verify that exact documented transport cast, NOT a relaxed
                # numeric tolerance or an arbitrary action transformation.
                if data['requested'].dtype!=np.float32 or not np.array_equal(
                        np.asarray(plan,np.float32),data['requested']):
                    failures.append('documented_float32_planned_requested_identity')
                elif not np.array_equal(plan,data['requested']):
                    counts['exact_after_documented_float32_transport_cast']+=1
                    cast_max_abs=max(cast_max_abs,float(np.abs(plan-data['requested']).max()))
                for name in ('O_t_proprio','actual_observed_proprio'):
                    if not (block/(name+'.npy')).is_file():failures.append('missing_endpoint_proprio')
                    else:load(block/(name+'.npy'))
                # Post-block attribution is not an observed recovery label and
                # may legitimately be absent at terminal success. The original
                # decision supplies the BEFORE-block attribution for features.
                decision=folder.parent/f'decision_{folder.name.split("_")[-1]}.json'
                if not decision.is_file():failures.append('missing_before_block_decision')
                else:
                    hashes[str(decision.relative_to(root))]=file_sha(decision)
                    metadata=json.loads(decision.read_text())
                    if 'attribution' not in metadata:failures.append('missing_before_block_attribution')
        counts['complete_16_step_blocks']+=int(n==16)
        counts['aligned_potential_auxiliary_blocks']+=int(not failures)
        problems.update(failures)
        rows.append(dict(source_key=VERSION,suite='libero90',task=task,state=state,
            condition=condition,path=str(path.relative_to(root)),length=n,
            potential_auxiliary=not failures,failures=failures,
            role='not_promoted_not_labelled_not_training',unexecuted_candidates_have_no_labels=True))
    report=dict(source=VERSION,counts=dict(counts),groups=sorted(groups),problems=dict(problems),
        excluded32_states={task:sorted(states) for task,states in SEALED.items()},
        recovery_labels_generated=False,training_started=False,
        promotion_requires_new_membership_and_label_quality_audit=True,
        per_step_proprio_not_saved=True,endpoint_proprio_may_support_endpoint_labels=True,
        action_transport_contract='frozen_execute_np_asarray_result_actions_float32_exact',
        max_plan_to_float32_quantization=cast_max_abs,
        full_counterfactual_candidate_execution_not_available=True,
        scope='selected_full_method_historical_consumed_development_blocks_only',
        source_hashes=hashes,entries=rows)
    for path,sha in hashes.items():
        if file_sha(root/path)!=sha:raise ValueError('Historical source changed during audit')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise ValueError('Do not overwrite historical audit')
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes','entries')},indent=2))


if __name__=='__main__':main()
