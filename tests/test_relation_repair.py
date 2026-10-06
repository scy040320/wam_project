import numpy as np
import pytest

from wam_reranking.relation_repair import apply_relation_repair, relation_repair_features


def run(pred, fp, **kw):
    n = len(pred)
    options = dict(target_available=np.ones(n, bool), observation_reliable=np.ones(n, bool),
                   action_unreliable=np.zeros(n, bool), command_deviation=np.zeros(n, bool))
    options.update(kw)
    return apply_relation_repair(np.asarray(pred), np.asarray(fp, np.float32),
                                 np.full(n, .9, np.float32), **options)


def test_only_unresolved_negative_evidence_can_be_repaired():
    fp = np.array([[.1]*4, [.1]*4, [.8,.1,.1,.1], [.1,.8,.1,.1], [.1,.1,.8,.1], [.1,.1,.1,.8]])
    out, new, active = run([0,4,1,2,3,4], fp)
    assert out.tolist() == [2,2,1,2,3,4]
    assert active.tolist() == [True,True,False,False,False,False]
    np.testing.assert_array_equal(new[:,[0,2,3]], fp.astype(np.float32)[:,[0,2,3]])


@pytest.mark.parametrize('key,value', [('target_available',False), ('observation_reliable',False),
                                     ('action_unreliable',True), ('command_deviation',True)])
def test_hard_evidence_cannot_be_bypassed(key, value):
    out, _, active = run([0], [[.1]*4], **{key:np.array([value])})
    assert out[0] == 0 and not active[0]


def test_no_change_below_original_factor_threshold():
    out, fp, active = apply_relation_repair(np.array([0]), np.full((1,4),.1),np.array([.499]),
        target_available=np.array([True]),observation_reliable=np.array([True]),
        action_unreliable=np.array([False]),command_deviation=np.array([False]))
    assert out[0] == 0 and not active[0]


def test_feature_contract_excludes_global_context_and_preserves_inputs():
    x = np.zeros((2,2298),np.float32); copy=x.copy()
    out=relation_repair_features(x)
    x[:,:2138]=999
    np.testing.assert_array_equal(out,relation_repair_features(x))
    assert out.shape==(2,181) and np.isfinite(out).all()
    np.testing.assert_array_equal(copy,np.zeros_like(copy))


def test_nonfinite_features_are_rejected():
    x=np.zeros((1,2298),np.float32);x[0,2225]=np.nan
    with pytest.raises(ValueError): relation_repair_features(x)
