"""Separate actual recovery supervision from candidate prediction/features."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .contracts import PREDICATES, TriValue
from .recovery_contract import SCHEMA

SEQUENCE_FIELDS = ('primary_sequence', 'wrist_sequence', 'proprio_sequence',
                   'requested_actions', 'applied_actions')
LABEL_SOURCES = ('observed_image_support', 'measured_command_difference', 'proprio_support')


def audit_recovery_supervision(record, *, directory: Path, tracked=None):
    """Fail closed; episode success, simulator end state or predictions are not labels.

    Valid record paths are relative to the candidate folder. The first block's
    complete evidence is required, independently of a later episode outcome.
    """
    problems = []
    if not record:
        return dict(ready=False, missing=['recovery_execution_record'],
                    supervised_predicates=[], mask={p:False for p in PREDICATES})
    if record.get('schema') != SCHEMA:
        problems.append('recovery_schema')
    length = record.get('executed_length')
    if length != 16:
        problems.append('incomplete_or_unaligned_first_block')
    if record.get('step_indices') != list(range(17)):
        problems.append('step_alignment')
    for name in ('snapshot_sha256', 'query_observation_sha256', 'planned_actions_sha256'):
        value = record.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()):
            problems.append(name)
    arrays = {}
    directory = directory.resolve()
    def reference(name):
        raw = record.get(name)
        if not isinstance(raw, str) or not raw:
            problems.append(name); return None
        path = (directory/raw).resolve()
        if directory not in path.parents or not path.is_file() or path.stat().st_size == 0:
            problems.append(name); return None
        if tracked: tracked(path)
        return path
    for name in SEQUENCE_FIELDS:
        path = reference(name)
        if path is not None:
            array = np.load(path, allow_pickle=False)
            if not np.isfinite(array).all():
                problems.append(name+'.nonfinite')
            if name in ('requested_actions', 'applied_actions'):
                if array.shape != (16,7): problems.append(name+'.shape')
            elif name in ('primary_sequence','wrist_sequence'):
                if array.ndim != 4 or array.shape[0] != 17 or array.shape[-1] != 3:
                    problems.append(name+'.shape')
            elif array.ndim != 2 or array.shape[0] != 17:
                problems.append(name+'.shape')
            arrays[name] = array
    labels_path = reference('predicate_labels_path')
    supervised, mask = [], {p:False for p in PREDICATES}
    if labels_path is not None:
        labels = json.loads(labels_path.read_text())
        if labels.get('role') != 'supervision_only' or labels.get('schema') != SCHEMA:
            problems.append('predicate_label_role_or_schema')
        if labels.get('candidate_id') != record.get('candidate_id'):
            problems.append('predicate_label_candidate_identity')
        items = labels.get('predicates',{})
        if set(items) != set(PREDICATES):
            problems.append('predicate_label_coverage')
        for name, label in items.items():
            if name not in mask: continue
            if set(label) != {'before','after','mask','quality','source','evidence_steps','evidence_ids'}:
                problems.append(name+'.label_contract'); continue
            if label['before'] not in [v.value for v in TriValue] or label['after'] not in [v.value for v in TriValue]:
                problems.append(name+'.trivalue'); continue
            if not isinstance(label['mask'],bool):
                problems.append(name+'.mask_type'); continue
            if label['source'] not in LABEL_SOURCES:
                problems.append(name+'.unsupported_label_source'); continue
            if not isinstance(label['quality'],(float,int)) or not np.isfinite(label['quality']) or not 0 <= label['quality'] <= 1:
                problems.append(name+'.quality'); continue
            steps = label['evidence_steps']
            if not steps or any(not isinstance(i,int) or i not in range(17) for i in steps):
                problems.append(name+'.evidence_time'); continue
            ids = label['evidence_ids']
            if not ids or any(i not in SEQUENCE_FIELDS for i in ids):
                problems.append(name+'.evidence_reference'); continue
            if label['source'] == 'measured_command_difference' and name != 'execution_consistent':
                problems.append(name+'.command_does_not_certify_physics'); continue
            if label['mask'] and (label['before']=='unknown' or label['after']=='unknown' or label['quality']<.5):
                problems.append(name+'.unobservable_label_must_be_masked'); continue
            mask[name] = label['mask']
            if label['mask']: supervised.append(name)
    if not supervised: problems.append('no_observable_predicate_supervision')
    return dict(ready=not problems, missing=problems,
                supervised_predicates=supervised, mask=mask)
