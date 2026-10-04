"""Shared training/evaluation contract for observable mismatch factors."""
from __future__ import annotations

import numpy as np


def supervised_factor_recall(probabilities, targets, masks, *, threshold=0.5):
    probabilities = np.asarray(probabilities)
    targets, masks = np.asarray(targets), np.asarray(masks)
    if probabilities.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError('factor probabilities, targets and supervision masks must align')
    recalls, support = [], []
    for column in range(targets.shape[1]):
        positive = (masks[:, column] > 0.5) & (targets[:, column] > 0.5)
        support.append(int(positive.sum()))
        recalls.append(float((probabilities[positive, column] >= threshold).mean()) if positive.any() else 0.0)
    return recalls, support


def contrastive_supervised_rows(target_available, masks):
    """Eligibility concerns label observability, not positive label values."""
    masks = np.asarray(masks)
    return (np.asarray(target_available) > 0.5) & (masks > 0.5).all(axis=1)


def paired_object_indices(parts, *, split='train'):
    """Match object shifts to clean controls only inside the same source/group.

    Index ordering follows the training combine() order. Legacy feature rows
    without moment metadata remain usable for supervised training but cannot
    provide a physical paired loss.
    """
    positives, negatives, offset = [], [], 0
    for source_number, part in enumerate(parts):
        chosen = np.flatnonzero(np.asarray(part['split']) == split)
        local_positions = {int(index): offset + i for i, index in enumerate(chosen)}
        if 'moments' in part:
            groups = {}
            for index in chosen:
                group = (source_number, int(part['tasks'][index]), int(part['states'][index]), int(part['moments'][index]))
                groups.setdefault(group, []).append(int(index))
            for indices in groups.values():
                clean = [i for i in indices if part['y'][i] == 0 and part['X'][i, -1] > .5]
                shifted = [i for i in indices if part['y'][i] == 2 and part['f'][i, 1] > .5 and part['f'][i, 5] < .5 and part['X'][i, -1] > .5]
                if shifted and len(clean) > 1:
                    raise ValueError('multiple supervised clean controls in one physical group')
                if len(clean) == 1:
                    for index in shifted:
                        positives.append(local_positions[index]); negatives.append(local_positions[clean[0]])
        offset += len(chosen)
    return np.asarray(positives, np.int64), np.asarray(negatives, np.int64)


def supported_normal(relation_evidence, normal_probabilities, factor_probabilities,
                     observation_reliable, action_unreliable, command_deviation,
                     *, threshold=0.5):
    """Use the trained normal head only when observations support a decision."""
    all_negative = (np.asarray(factor_probabilities) < threshold).all(axis=1)
    corroborated = np.asarray(relation_evidence, bool) | (
        np.asarray(observation_reliable, bool) & (np.asarray(normal_probabilities) >= threshold)
    )
    return corroborated & all_negative & ~np.asarray(action_unreliable, bool) & ~np.asarray(command_deviation, bool)


def task_development_gate(metrics_by_task):
    """Identical predeclared per-task thresholds for selection and reporting."""
    def recall(task, cause):
        return metrics_by_task[str(task)]['per_class'][cause]['recall']
    observable = (14, 16, 17, 34, 45, 64, 73)
    checks = {
        'task23_object_shift_recall_at_least_0_50': recall(23, 'object_shift') >= .5,
        'task58_evidence_insufficient_unknown_recall_at_least_0_60': recall(58, 'unknown') >= .6,
        'task58_execution_recall_at_least_0_50': recall(58, 'execution_contact_deviation') >= .5,
        'relation_tasks_normal_recall_each_at_least_0_50': min(recall(t, 'normal') for t in (14,34,64)) >= .5,
        'relation_tasks_object_shift_recall_each_at_least_0_50': min(recall(t, 'object_shift') for t in (14,34,64)) >= .5,
        'promoted_16_45_73_normal_each_at_least_0_50': min(recall(t, 'normal') for t in (16,45,73)) >= .5,
        'promoted_16_45_73_object_shift_each_at_least_0_50': min(recall(t, 'object_shift') for t in (16,45,73)) >= .5,
        'targeted_17_39_normal_each_at_least_0_50': min(recall(t, 'normal') for t in (17,39)) >= .5,
        'targeted_17_object_shift_recall_at_least_0_50': recall(17, 'object_shift') >= .5,
        'task39_evidence_insufficient_unknown_recall_at_least_0_60': recall(39, 'unknown') >= .6,
        'predeclared_observable_task_object_shift_recall_each_at_least_0_50': min(recall(t,'object_shift') for t in observable) >= .5,
    }
    return checks
