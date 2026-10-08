"""Single-fit, success-first terminal utility with frozen deployment margins.

Training-only helpers below read terminal supervision. Inference remains the
unchanged PRE-only command_predicate_utility API; no label enters that API.
"""
from dataclasses import replace
import numpy as np
from . import persistent_conditioned_utility as old

SCHEMA = 'priority_constrained_command_predicate_utility_v1'
FIT_STEPS = 1200
SUCCESS_STEPS = 900
LR = .01
L2 = .1
REPAIR_MARGIN_MULTIPLIER = 2.
BACKTRACK_STEPS = 12
CONTROL_MODES = ('full', 'masked', 'shuffled', 'no_dag', 'without_command',
                 'masked_learned', 'shuffled_learned')


def cross_group_donors(pools):
    """Fixed within-split permutation, rotating slots as well as groups.

    No outcomes/probabilities are consulted. The legacy same-condition slot
    permutation was ineffective on the only corrective training example.
    Source keys establish identity/order only, never score input.
    """
    result = {}
    for split in ('train','val'):
        groups = {}
        for p in sorted((p for p in pools if p['split']==split),key=lambda p:p['key']):
            groups.setdefault(p['key'][:2],[]).append(p['key'])
        names=sorted(groups)
        if len(names)<2: raise ValueError('Two donor groups per split required')
        if len({len(v) for v in groups.values()})!=1: raise ValueError('Unequal source slots')
        for i,g in enumerate(names):
            donor=groups[names[(i+1)%len(names)]]
            for j,k in enumerate(groups[g]): result[k]=donor[(j+1)%len(donor)]
    if len(set(result.values()))!=len(pools): raise ValueError('Donor mapping is not a permutation')
    return result


def fixed_selection(base, delta, references, comparable, official, margins):
    """Vectorized equivalent of the unchanged frozen-reference switch rule.

    Terminal labels are deliberately not arguments. Must be cross-checked
    against the deployment API on every final/calibration pool by the runner.
    """
    base,delta,official=np.asarray(base),np.asarray(delta),np.asarray(official)
    refs=np.asarray(references,dtype=int); n=len(refs); safe=np.maximum(refs,0)
    if base.shape!=delta.shape or base.shape!=official.shape: raise ValueError('Candidate alignment')
    if not np.isfinite(base).all() or not np.isfinite(delta).all(): raise ValueError('Nonfinite scores')
    if np.any(np.ptp(delta,axis=1)>old.CAP+1e-12): raise ValueError('Frozen cap exceeded')
    scores=base+delta
    eligible=np.asarray(comparable,bool).copy()
    eligible[np.arange(n),safe]=False
    eligible &= (delta>delta[np.arange(n),safe,None]+1e-12)
    eligible &= scores>scores[np.arange(n),safe,None]+np.asarray(margins)[:,None]
    eligible[refs<0,:]=False
    top=np.max(np.where(eligible,scores,-np.inf),axis=1)
    tied=eligible & (scores>=top[:,None]-np.asarray(margins)[:,None])
    candidate=np.argmax(np.where(tied,official,-np.inf),axis=1)
    return np.where(np.any(eligible,axis=1),candidate,refs)


def _arrays(pools):
    base=np.array([p['base_scores'] for p in pools])
    refs=np.array([-1 if p['baseline_id'] is None else p['baseline_id'] for p in pools])
    mask=np.array([[i in p['comparable'] and p['decisions'][i].accepted for i in range(4)] for p in pools])
    official=np.array([p['values'] for p in pools])
    margins=np.array([max(.01,float(p['kw']['backbone'].switch_margin)) for p in pools])
    y=np.array([[v['success'] for v in p['y']] for p in pools],bool)
    ref_y=np.array([False if p['baseline_id'] is None else p['y'][p['baseline_id']]['success'] for p in pools])
    value_y=np.array([p['y'][p['value_id']]['success'] for p in pools])
    return base,refs,mask,official,margins,y,ref_y|value_y


def fit_once(train_pools, *, steps=FIT_STEPS, lr=LR, l2=L2):
    if (steps,lr,l2)!=(FIT_STEPS,LR,L2): raise ValueError('Predeclared single-fit configuration required')
    pairs,audit=old.preference_pairs(train_pools)
    if not pairs or not audit['success']: raise ValueError('No admitted TRAIN success supervision')
    groups={'corrective':[],'protective':[],'cost':[]}
    for k,i,j,weight,margin in pairs:
        success_difference=train_pools[k]['y'][i]['success']!=train_pools[k]['y'][j]['success']
        name=('protective' if i==train_pools[k]['baseline_id'] else 'corrective') if success_difference else 'cost'
        groups[name].append((k,i,j,margin))
    if not groups['corrective']: raise ValueError('No corrective TRAIN pair; no fit')
    x=np.array([p['x'] for p in train_pools]); scale=x.reshape(-1,x.shape[-1]).std(axis=0)
    scale[scale<1e-8]=1.; support=np.abs(x).max(axis=(0,1))
    model=old.ConditionalUtility(scale,np.zeros(x.shape[-1]),support)
    z=np.array([model.normalized(v) for v in x])
    base,refs,mask,official,margins,y,protect=_arrays(train_pools)
    def delta(w):
        raw=old.CAP/2*np.tanh(np.clip(z@w,-30.,30.))
        return raw-raw.mean(axis=1,keepdims=True)
    def safe(w):
        cid=fixed_selection(base,delta(w),refs,mask,official,margins)
        good=np.array([False if i<0 else y[k,i] for k,i in enumerate(cid)])
        return not np.any(protect & ~good)
    # V8 fallback is explicitly handled by the runner. Root pool fits must not
    # silently treat an unexecuted fallback as a failed training label.
    if any(p['baseline_id'] is None and p['fallback_y'] is not None and p['fallback_y']['success'] for p in train_pools):
        raise ValueError('Successful fallback requires an explicit guard representation')
    if not safe(model.weights): raise ValueError('Frozen zero identity has an existing harm; no fit')
    arrays={}
    for name,rows in groups.items():
        if rows:
            kk,ii,jj,mm=map(np.array,zip(*rows)); kk=kk.astype(int);ii=ii.astype(int);jj=jj.astype(int)
            arrays[name]=(z[kk,ii],z[kk,jj],base[kk,ii]-base[kk,jj],mm)
        else: arrays[name]=None
    def terms(w,name):
        if arrays[name] is None: return 0.,np.zeros_like(w),np.array([])
        zi,zj,b,m=arrays[name]; ti,tj=np.tanh(zi@w),np.tanh(zj@w)
        gap=b+old.CAP/2*(ti-tj)
        derivative=old.CAP/(2*m[:,None])*((1-ti*ti)[:,None]*zi-(1-tj*tj)[:,None]*zj)
        if name=='protective':
            violation=np.maximum(0.,(-m-gap)/m)
            return float(np.mean(violation**2)),np.mean(-2*violation[:,None]*derivative,axis=0),gap
        target=(REPAIR_MARGIN_MULTIPLIER if name=='corrective' else 1.)*m
        logits=(gap-target)/m; inv=1/(1+np.exp(np.clip(logits,-30,30)))
        return float(np.mean(np.logaddexp(0.,-logits))),np.mean(-inv[:,None]*derivative,axis=0),gap
    def repair_ready(w):
        _,_,gap=terms(w,'corrective')
        if not np.all(gap>=REPAIR_MARGIN_MULTIPLIER*arrays['corrective'][3]): return False
        cid=fixed_selection(base,delta(w),refs,mask,official,margins)
        return all(cid[k]>=0 and y[k,cid[k]] for k,_,_,_ in groups['corrective'])
    for b,m in zip(arrays['corrective'][2],arrays['corrective'][3]):
        if b+old.CAP<=REPAIR_MARGIN_MULTIPLIER*m: raise ValueError('Training reserve unreachable under frozen cap')
    w=model.weights.copy(); accepted=0; blocked=0; cost_steps=0; log=[]
    for step in range(steps):
        primary,pg,_=terms(w,'corrective'); guard,gg,_=terms(w,'protective')
        phase='success'
        if step>=SUCCESS_STEPS and repair_ready(w):
            loss,gradient,_=terms(w,'cost'); phase='cost'; cost_steps+=1
        else: loss,gradient=primary+guard,pg+gg
        gradient=gradient+l2*w; step_accepted=False
        for backtrack in range(BACKTRACK_STEPS):
            proposal=w-lr/(2**backtrack)*gradient
            if not safe(proposal) or (phase=='cost' and not repair_ready(proposal)): continue
            w=proposal; step_accepted=True;accepted+=1;break
        if not step_accepted: blocked+=1
        if step in (0,SUCCESS_STEPS-1,steps-1):
            _,_,gaps=terms(w,'corrective')
            log.append(dict(step=step+1,phase=phase,loss=float(loss+l2/2*(w@w)),
                corrective_gaps=gaps.tolist(),safe_success_preservation=safe(w)))
    return replace(model,weights=w),dict(**audit,fit_count=1,steps=steps,lr=lr,l2=l2,
        corrective_success_pairs=len(groups['corrective']),protective_success_pairs=len(groups['protective']),
        success_steps=SUCCESS_STEPS,cost_steps=cost_steps,accepted_updates=accepted,blocked_updates=blocked,
        repair_training_reserve=REPAIR_MARGIN_MULTIPLIER,guard='actual train selection against V8 and value successes',
        nonzero_weights=int(np.count_nonzero(abs(w)>1e-10)),checkpoints=log,
        validation_used=False,objective='cohort-normalized corrective margin; success protection constraint; conditional cost stage')
