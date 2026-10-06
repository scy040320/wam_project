"""Bounded development-only attribution/selection diagnostics.

Oracle and shuffled inputs are diagnostic controls, never deployment models.
Outcomes are separated from the outcome-free selection interface.
"""
from dataclasses import replace

import numpy as np

from .candidate_utility import select_with_utility
from .contracts import CandidateDecision
from .evidence_residual import fit_evidence_residual, select_evidence_residual
from .epistemic_backbone_policy import select_epistemic_backbone

GAINS = (0., .125, .25, .5, 1.)


def check_membership(rows):
    keys = [tuple(row['key']) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate pool identity')
    if any(row['split'] not in ('train', 'val') or row['key'][1] > 7 for row in rows):
        raise ValueError('Consumed acceptance or unsupported split entered diagnosis')
    train = {tuple(row['key'][:2]) for row in rows if row['split'] == 'train'}
    val = {tuple(row['key'][:2]) for row in rows if row['split'] == 'val'}
    if train & val:
        raise ValueError('task/state group leakage')
    for row in rows:
        if set(row['features']) != set(row['supervision']):
            raise ValueError('Candidate feature/outcome identity mismatch')
        for c in row['features'].values():
            if set(c) != {'backbone', 'cause', 'decision'}:
                raise ValueError('Unknown field or GT in feature envelope')


def feature_audit(rows):
    nonzero, discriminative, mixed, active_pairs = 0, 0, 0, 0
    for row in rows:
        ids = sorted(row['features'])
        z = np.stack([row['features'][i]['cause'] for i in ids])
        nonzero += bool(np.any(np.abs(z) > 1e-12))
        discriminative += bool(np.any(np.ptp(z, axis=0) > 1e-12))
        y = [row['supervision'][i]['success'] for i in ids]
        mixed += any(y) and not all(y)
        active_pairs += sum(y[a] and not y[b] and np.any(np.abs(z[a]-z[b]) > 1e-12)
                            for a in range(len(ids)) for b in range(len(ids)))
    return dict(pools=len(rows), nonzero_cause_pools=int(nonzero),
        within_pool_discriminative_pools=int(discriminative), mixed_pools=int(mixed),
        nonzero_success_preference_pairs=int(active_pairs))


def choose(features, backbone, residual, *, panel, attribution=None):
    """This deployable-shaped function has no supervision/outcome argument."""
    ds, xs, zs = [], {}, {}
    for cid, c in features.items():
        d = c['decision']
        if panel == 'ranking_only':
            d = replace(d, accepted=True, rejection_reasons=(), components={})
        elif panel != 'current_variant_gate':
            raise ValueError('Unsupported diagnostic panel')
        ds.append(d); xs[cid] = c['backbone']; zs[cid] = c['cause']
    selector = select_evidence_residual if panel == 'ranking_only' else select_epistemic_backbone
    args = dict(decisions=ds, backbone_features=xs, cause_features=zs,
                backbone=backbone, residual=residual)
    if panel != 'ranking_only': args['attribution'] = attribution
    selected, _ = selector(**args)
    return None if selected is None else selected.candidate_id


def evaluate(rows, backbone, residual, *, panel):
    entries = []
    for row in rows:
        features, labels = row['features'], row['supervision']
        ids = sorted(features)
        ds = [replace(features[i]['decision'], accepted=True, rejection_reasons=(),
                      components={}) for i in ids]
        base, _ = select_with_utility(ds, {i:features[i]['backbone'] for i in ids}, backbone)
        value = max(ids, key=lambda i:(features[i]['decision'].official_value, -i))
        selected = choose(features, backbone, residual, panel=panel, attribution=row.get('attribution'))
        selected_label = None if selected is None else labels[selected]
        base_label, value_label = labels[base.candidate_id], labels[value]
        entries.append(dict(key=row['key'], selected=selected,
            success=None if selected_label is None else bool(selected_label['success']),
            backbone_selected=base.candidate_id, backbone_success=bool(base_label['success']),
            value_selected=value, value_success=bool(value_label['success']),
            covered=any(y['success'] for y in labels.values()),
            steps_delta_vs_backbone=(selected_label['steps']-base_label['steps'])
                if selected_label and selected_label['success'] and base_label['success'] else None,
            calls_delta_vs_backbone=(selected_label['calls']-base_label['calls'])
                if selected_label and selected_label['success'] and base_label['success'] else None))
    summary = dict(pools=len(entries), success=sum(e['success'] is True for e in entries),
        backbone_success=sum(e['backbone_success'] for e in entries),
        value_success=sum(e['value_success'] for e in entries),
        candidate_coverage=sum(e['covered'] for e in entries),
        unobserved_fallback=sum(e['success'] is None for e in entries),
        selection_changes_vs_backbone=sum(e['selected'] != e['backbone_selected'] for e in entries))
    for control in ('backbone', 'value'):
        summary['gains_vs_'+control] = sum(e['success'] is True and not e[control+'_success'] for e in entries)
        summary['harms_vs_'+control] = sum(e['success'] is False and e[control+'_success'] for e in entries)
    for cost in ('steps', 'calls'):
        paired = [e[cost+'_delta_vs_backbone'] for e in entries if e[cost+'_delta_vs_backbone'] is not None]
        summary['paired_'+cost+'_delta_vs_backbone'] = dict(n=len(paired),
            mean=float(np.mean(paired)) if paired else None,
            median=float(np.median(paired)) if paired else None)
    return dict(summary=summary, entries=entries)


def fit_diagnostic(rows, backbone, *, panel, fitter=fit_evidence_residual):
    """Historical loss and fixed gain grid; calibration sees training only."""
    check_membership(rows)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    pairs, support = [], []
    for row in train:
        f, y = row['features'], row['supervision']
        support.extend(c['cause'] for c in f.values())
        for a in sorted(f):
            for b in sorted(f):
                if y[a]['success'] and not y[b]['success']:
                    pairs.append((f[a]['cause'], f[b]['cause'],
                        backbone.score(f[a]['backbone'])-backbone.score(f[b]['backbone'])))
    fitted = fitter(pairs, support)
    trials = [dict(gain=gain, **evaluate(train, backbone, replace(fitted, gain=gain), panel=panel)['summary'])
              for gain in GAINS]
    eligible = [r for r in trials if r['harms_vs_backbone'] == r['harms_vs_value'] == r['unobserved_fallback'] == 0]
    best = max(eligible, key=lambda r:(r['success'], -r['gain'])) if eligible else None
    chosen = replace(fitted, gain=0. if best is None else best['gain'])
    return dict(model=chosen, report=dict(
        panel=panel, train_gain_calibration=trials,
        train_calibration_feasible=best is not None, selected_gain=chosen.gain,
        train=evaluate(train, backbone, chosen, panel=panel),
        validation=evaluate(val, backbone, chosen, panel=panel),
        feature_audit_train=feature_audit(train), feature_audit_val=feature_audit(val),
        success_preference_pairs=len(pairs), gain_selection_used_validation=False))
