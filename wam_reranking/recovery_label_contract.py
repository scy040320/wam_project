"""Auditable endpoint/availability silver labels, separate from deployment.

V2 inherits the V1 evidence separation, not its all-or-nothing masks. A hidden
before state can become observable after the chunk without inventing a physical
transition. Recorded endpoint proprio is optional; absent intermediate proprio
is never interpolated or required for image/command supervision. This revision
certifies visibility, recorded commands and a 2-D relation proxy only, NOT
physical grasp, lift, placement or contact success.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import PREDICATES

SCHEMA = 'candidate_recovery_labels_v2'
INHERITS = 'candidate_recovery_supervision_v1'
IDENTITY_FIELDS = ('dataset', 'suite', 'task', 'state', 'candidate_id', 'observed_block_id')
ARRAY_FIELDS = ('primary_sequence', 'wrist_sequence', 'requested_actions', 'applied_actions')
OPTIONAL_ARRAY_FIELDS = ('proprio_start', 'proprio_end', 'proprio_sequence')
TARGETS = tuple(PREDICATES) + ('command_consistent',)
OBSERVABILITY_TARGETS = ('target_evidence_available', 'receptacle_evidence_available')
VISIBILITY_TARGETS = ('target_visible', 'receptacle_visible')
LABEL_FIELDS = ('before', 'after', 'before_mask', 'after_mask', 'transition_mask',
                'before_quality', 'after_quality', 'source', 'evidence_steps', 'evidence_ids')
SOURCES = ('observed_image_support', 'measured_command_difference', 'proprio_support')
MIN_QUALITY = 0.5
ROLES = ('historical_selected_arm_auxiliary', 'paired_recovery_supervision')
# These already-consumed validation groups must not become V9 fitting data.
SEALED_32_GROUPS = frozenset((('libero90', 0, 35), ('libero90', 0, 36),
    ('libero90', 9, 14), ('libero90', 9, 15), ('libero90', 46, 14),
    ('libero90', 46, 15), ('libero90', 57, 14), ('libero90', 57, 15)))


def masked_label(source='observed_image_support'):
    """An explicit absence of supervision; not an invented negative label."""
    return dict(before='unknown', after='unknown', before_mask=False, after_mask=False,
                transition_mask=False, before_quality=0., after_quality=0.,
                source=source, evidence_steps=[], evidence_ids=[])


def _identity_problems(identity):
    if not isinstance(identity, Mapping) or set(identity) != set(IDENTITY_FIELDS):
        return ['source_scoped_identity_fields']
    problems = []
    for name in ('dataset', 'suite', 'observed_block_id'):
        if not isinstance(identity[name], str) or not identity[name].strip():
            problems.append('identity.'+name)
    for name in ('task', 'state', 'candidate_id'):
        if type(identity[name]) is not int or identity[name] < 0:
            problems.append('identity.'+name)
    return problems


def _physical_group(identity):
    # Source-scoped identity stays literal, while group isolation normalizes
    # harmless suite spelling aliases so they cannot hide the same task/state.
    suite = identity['suite'].lower().replace('_', '').replace('-', '')
    return (suite, identity['task'], identity['state'])


def audit_recovery_membership(records: Sequence[Mapping], *, excluded_groups=()):
    """Source identities are unique; physical task/state groups never cross split.

    Dataset names scope IDs but cannot excuse train/val overlap of the same
    suite/task/state. No requirement that the legacy 768 candidates be relabelled.
    ``excluded_groups`` lets the caller freeze additional reserved 128-scene IDs.
    """
    problems = []; seen = set(); groups = {'train': set(), 'val': set()}
    forbidden = SEALED_32_GROUPS | frozenset((str(g[0]).lower().replace('_','').replace('-',''),g[1],g[2])
                                             for g in excluded_groups)
    for index, record in enumerate(records):
        identity = record.get('identity')
        malformed = _identity_problems(identity)
        if malformed:
            problems.extend(f'record{index}.{p}' for p in malformed); continue
        key = tuple(identity[f] for f in IDENTITY_FIELDS)
        if key in seen: problems.append('duplicate_source_scoped_identity')
        seen.add(key)
        split = record.get('split')
        if split not in groups:
            problems.append('non_development_split'); continue
        if record.get('role') not in ROLES: problems.append('unsupported_data_role')
        group = _physical_group(identity)
        groups[split].add(group)
        if group in forbidden: problems.append('reserved_validation_group')
    overlap = groups['train'] & groups['val']
    if overlap: problems.append('group_leakage')
    return dict(ready=not problems, problems=problems, records=len(records),
                unique_identities=len(seen), group_leakage=bool(overlap),
                train_groups=sorted(groups['train']), val_groups=sorted(groups['val']))


def audit_recovery_labels(record: Mapping, *, directory: Path, tracked=None):
    """Validate actual evidence and independent V2 labels without deriving them."""
    problems = []; loaded = {}; supervision = {}
    identity = record.get('identity')
    problems.extend(_identity_problems(identity))
    if record.get('schema') != SCHEMA or record.get('inherits') != INHERITS:
        problems.append('schema_inheritance')
    if record.get('role') not in ROLES: problems.append('unsupported_data_role')
    if record.get('split') not in ('train', 'val'): problems.append('non_development_split')
    complete = type(record.get('executed_length')) is int and record['executed_length'] == 16
    if not complete: problems.append('short_block_retained_excluded_formal')
    if record.get('step_indices') != list(range(17)): problems.append('step_alignment')
    directory = directory.resolve()
    references = record.get('arrays', {})
    if not isinstance(references, Mapping):
        references = {}; problems.append('array_reference_mapping')
    if not set(ARRAY_FIELDS).issubset(references): problems.append('mandatory_actual_arrays')
    if set(references) - set(ARRAY_FIELDS + OPTIONAL_ARRAY_FIELDS):
        problems.append('unsupported_array_field')

    def reference(raw, name):
        if not isinstance(raw, str) or not raw:
            problems.append(name+'.missing_reference'); return None
        path = (directory/raw).resolve()
        if directory not in path.parents or not path.is_file() or path.stat().st_size == 0:
            problems.append(name+'.outside_or_missing_evidence'); return None
        if tracked is not None: tracked(path)
        return path

    for name, raw in references.items():
        path = reference(raw, name)
        if path is None: continue
        try:
            arr = np.load(path, allow_pickle=False)
            if not isinstance(arr, np.ndarray) or not np.isfinite(arr).all():
                problems.append(name+'.nonfinite_or_nonarray'); continue
            if name in ('primary_sequence', 'wrist_sequence'):
                if arr.ndim != 4 or arr.shape[0] != 17 or arr.shape[-1] != 3 or min(arr.shape[1:3]) < 1:
                    problems.append(name+'.shape')
            elif name in ('requested_actions', 'applied_actions'):
                if arr.shape != (16, 7): problems.append(name+'.shape')
            elif name == 'proprio_sequence':
                if arr.ndim != 2 or arr.shape[0] != 17: problems.append(name+'.shape')
            elif arr.ndim != 1 or arr.size == 0: problems.append(name+'.shape')
            loaded[name] = arr
        except (ValueError, OSError, TypeError):
            problems.append(name+'.unreadable_actual_array')
    if ('proprio_start' in loaded) != ('proprio_end' in loaded):
        problems.append('endpoint_proprio_must_be_paired')
    if 'proprio_start' in loaded and 'proprio_end' in loaded:
        if loaded['proprio_start'].shape != loaded['proprio_end'].shape:
            problems.append('endpoint_proprio_shape_mismatch')
    path = reference(record.get('labels_path'), 'labels')
    labels = None
    if path is not None:
        try: labels = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, OSError): problems.append('labels.unreadable')
    if isinstance(labels, Mapping):
        if labels.get('schema') != SCHEMA or labels.get('inherits') != INHERITS:
            problems.append('label_schema_inheritance')
        if labels.get('role') != 'supervision_only': problems.append('label_role')
        if labels.get('identity') != identity: problems.append('label_source_identity')
        predicates = labels.get('predicates', {})
        availability = labels.get('observability', {})
        if not isinstance(predicates, Mapping) or set(predicates) != set(TARGETS):
            problems.append('predicate_label_coverage'); predicates = {}
        if not isinstance(availability, Mapping) or set(availability) != set(OBSERVABILITY_TARGETS):
            problems.append('observability_label_coverage'); availability = {}
        items = [(n, x, False) for n, x in predicates.items()]
        items += [(n, x, False) for n, x in availability.items()]
        items += [('relation_support_2d', labels.get('relation_support_2d'), True)]
        for name, label, proxy in items:
            prefix = name+'.'; label_problems = []
            expected_fields = set(LABEL_FIELDS) | ({'units'} if proxy else set())
            if not isinstance(label, Mapping) or set(label) != expected_fields:
                problems.append(prefix+'label_fields'); continue
            if proxy and label['units'] != 'image_space_proxy':
                label_problems.append('proxy_not_physical_units')
            masks = {m: label[m] for m in ('before_mask', 'after_mask', 'transition_mask')}
            if any(type(v) is not bool for v in masks.values()):
                label_problems.append('mask_type')
            for time in ('before', 'after'):
                value, quality = label[time], label[time+'_quality']
                unknown = value is None if proxy else value == 'unknown'
                valid_value = (value is None or (type(value) in (int, float) and
                    np.isfinite(value) and 0 <= value <= 1)) if proxy else value in ('true', 'false', 'unknown')
                if not valid_value: label_problems.append(time+'.value')
                if type(quality) not in (int, float) or not np.isfinite(quality) or not 0 <= quality <= 1:
                    label_problems.append(time+'.quality')
                elif masks[time+'_mask'] and (unknown or quality < MIN_QUALITY):
                    label_problems.append(time+'.unobservable_must_be_masked')
            if masks['transition_mask'] and not (masks['before_mask'] and masks['after_mask']):
                label_problems.append('transition_requires_both_observed_endpoints')
            source = label['source']; steps = label['evidence_steps']; ids = label['evidence_ids']
            if source not in SOURCES: label_problems.append('unsupported_label_source')
            if not isinstance(steps, list) or any(type(s) is not int or s not in range(17) for s in steps):
                label_problems.append('evidence_time'); steps = []
            if not isinstance(ids, list) or any(i not in loaded for i in ids):
                label_problems.append('actual_evidence_reference'); ids = []
            unmasked = any(masks.values())
            if unmasked and (not steps or not ids): label_problems.append('unmasked_without_evidence')
            if masks['before_mask'] and 0 not in steps: label_problems.append('before_time_witness')
            if masks['after_mask'] and 16 not in steps: label_problems.append('after_time_witness')
            if source == 'measured_command_difference':
                if name != 'command_consistent': label_problems.append('command_does_not_certify_contact')
                if unmasked and not {'requested_actions', 'applied_actions'}.issubset(ids):
                    label_problems.append('paired_actual_commands_required')
            elif name == 'command_consistent' and unmasked:
                label_problems.append('command_label_requires_actual_commands')
            if source == 'observed_image_support' and unmasked:
                if not {'primary_sequence', 'wrist_sequence'}.intersection(ids):
                    label_problems.append('image_source_requires_actual_images')
            if source == 'proprio_support' and unmasked:
                if 'proprio_sequence' not in ids:
                    if masks['before_mask'] and 'proprio_start' not in ids: label_problems.append('proprio_before_endpoint')
                    if masks['after_mask'] and 'proprio_end' not in ids: label_problems.append('proprio_after_endpoint')
                    if any(s not in (0, 16) for s in steps): label_problems.append('absent_intermediate_proprio')
            if name in PREDICATES and name not in VISIBILITY_TARGETS and unmasked:
                label_problems.append('physical_predicate_not_certified_by_v2_proxy')
            if name in VISIBILITY_TARGETS + OBSERVABILITY_TARGETS and unmasked and source != 'observed_image_support':
                label_problems.append('visibility_requires_actual_images')
            if proxy and unmasked and source != 'observed_image_support':
                label_problems.append('relation_proxy_requires_actual_images')
            problems.extend(prefix+p for p in label_problems)
            supervision[name] = dict(masks) if not label_problems else dict.fromkeys(masks, False)
    elif labels is not None:
        problems.append('label_mapping')
    any_supervised = any(any(m.values()) for m in supervision.values())
    if not any_supervised: problems.append('no_observable_supervision')
    return dict(schema=SCHEMA, ready=not problems, formal_eligible=complete and not problems,
                problems=problems, supervision_masks=supervision,
                intermediate_proprio_present='proprio_sequence' in loaded,
                endpoint_proprio_present={'proprio_start', 'proprio_end'}.issubset(loaded),
                physical_recovery_certified=False,
                role=record.get('role'), identity=identity)
