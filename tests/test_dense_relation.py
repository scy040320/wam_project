import numpy as np
import pytest
from PIL import Image
from wam_reranking.dense_relation import same_entity_family, dense_relation_patch_features


def test_family_is_language_not_task_id():
    assert same_entity_family('front black bowl','middle black bowl')=='black bowl'
    assert same_entity_family('middle black bowl','back black bowl')=='black bowl'
    assert same_entity_family('black bowl','plate') is None
    assert same_entity_family('black bowl','black bowl') is None


def test_spatial_change_is_not_collapsed_into_global_mean():
    a=np.zeros((64,64,3),np.uint8);a[16:24,16:24]=255
    b=np.zeros_like(a);b[40:48,40:48]=255
    images=[Image.fromarray(a),Image.fromarray(b),Image.fromarray(a),Image.fromarray(b),Image.fromarray(a),Image.fromarray(a)]
    maps=np.ones((6,64,64),np.float32)
    x=dense_relation_patch_features(images,maps)
    clean=dense_relation_patch_features([Image.fromarray(a)]*6,maps)
    assert x.shape==(1280,) and abs(a.mean()-b.mean())<1e-9
    assert not np.allclose(x,clean)


def test_missing_maps_do_not_hide_pixel_residual():
    a=Image.fromarray(np.zeros((64,64,3),np.uint8));b=Image.fromarray(np.ones((64,64,3),np.uint8)*255)
    x=dense_relation_patch_features([a,b,a,b,a,a],np.zeros((6,64,64),np.float32))
    assert x.max()>0 and np.isfinite(x).all()


def test_nonfinite_relevance_stops_extraction():
    maps=np.zeros((6,64,64));maps[1,0,0]=np.nan
    with pytest.raises(ValueError):dense_relation_patch_features([Image.new('RGB',(64,64))]*6,maps)
