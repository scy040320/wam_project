"""Contract tests for deployable subject–anchor temporal evidence."""

import numpy as np

from wam_reranking.relation_evidence import (
    instance_correspondence_maps,
    relation_temporal_features,
    typed_relation_features,
)


def blob(x: int, y: int) -> np.ndarray:
    out = np.zeros((64, 64), np.float32)
    out[y - 2 : y + 3, x - 2 : x + 3] = 1.0
    return out


def test_unchanged_relation_is_zero():
    subject, anchor = blob(20, 20), blob(40, 40)
    features = relation_temporal_features(subject, subject, anchor, anchor)
    assert features.shape == (8,)
    np.testing.assert_allclose(features[:6], 0, atol=1e-6)


def test_subject_and_anchor_changes_have_opposite_direction():
    subject, anchor = blob(20, 20), blob(40, 40)
    subject_moves = relation_temporal_features(subject, blob(24, 20), anchor, anchor)
    anchor_moves = relation_temporal_features(subject, subject, anchor, blob(44, 40))
    assert subject_moves[0] > 0
    assert anchor_moves[0] < 0


def test_joint_camera_translation_does_not_create_relation_change():
    features = relation_temporal_features(
        blob(20, 20), blob(23, 23), blob(40, 40), blob(43, 43)
    )
    np.testing.assert_allclose(features[:4], 0, atol=1e-6)


def test_missing_evidence_remains_finite():
    z = np.zeros((64, 64), np.float32)
    features = relation_temporal_features(z, z, z, z)
    assert np.isfinite(features).all()
    assert features[6] == 0 and features[7] == 0


def test_visual_prototype_selects_same_instance():
    maps = np.stack([blob(18, 20) + blob(46, 20), blob(18, 20) + blob(46, 20)])
    tokens = np.zeros((2, 64, 64, 2), np.float32)
    tokens[:, :, :, 0] = 1
    tokens[:, 18:23, 16:21] = np.asarray([0, 1], np.float32)
    matched = instance_correspondence_maps(maps, tokens, reference_index=0, prompt="left black bowl")
    assert matched[1, 20, 18] > matched[1, 20, 46]


def test_ordinal_prompt_disambiguates_identical_instances():
    maps = np.stack([blob(32, 18) + blob(32, 48)] * 2)
    tokens = np.ones((2, 64, 64, 2), np.float32)
    front = instance_correspondence_maps(
        maps, tokens, reference_index=0, prompt="front black bowl", ordinal_lock=True
    )
    back = instance_correspondence_maps(
        maps, tokens, reference_index=0, prompt="back black bowl", ordinal_lock=True
    )
    assert front[1, 48, 32] > front[1, 18, 32]
    assert back[1, 18, 32] > back[1, 48, 32]


def test_relation_types_produce_distinct_contracts():
    subject, shifted, anchor = blob(20, 20), blob(24, 20), blob(40, 40)
    on = typed_relation_features(subject, shifted, anchor, anchor, "on")
    inside = typed_relation_features(subject, shifted, anchor, anchor, "inside")
    articulated = typed_relation_features(subject, shifted, anchor, anchor, "articulated")
    assert on.shape == inside.shape == articulated.shape == (12,)
    assert not np.array_equal(on[:4], inside[:4])
    assert not np.array_equal(inside[:4], articulated[:4])
