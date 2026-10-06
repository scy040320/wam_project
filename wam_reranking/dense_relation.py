"""Spatial residual evidence for ambiguous same-category entity pairs.

Soft entity relevance masks qualify the region; they are not simulator masks
or proof of instance identity. Unlike scalar centroid/affinity features, patch
evidence retains displacement direction and multi-object configuration.
"""
import re

import numpy as np
from PIL import Image


def same_entity_family(subject: str, anchor: str) -> str | None:
    qualifiers = {'front','middle','back','left','right','top','bottom','stack','of','the'}
    def words(prompt):
        return tuple(w for w in re.findall(r'[a-z]+',prompt.lower()) if w not in qualifiers)
    a,b=words(subject),words(anchor)
    if not a or a!=b or subject.lower()==anchor.lower(): return None
    return ' '.join(a)


def dense_relation_patch_features(images, relevance, *, patch_side=8):
    if len(images)!=6 or len(relevance)!=6 or patch_side!=8:
        raise ValueError('expected pred/actual primary, pred/actual wrist, query primary/wrist; grid8')
    rgb=np.stack([np.asarray(im.convert('RGB').resize((64,64),Image.Resampling.BILINEAR),np.float32)/255 for im in images])
    maps=np.stack([np.asarray(Image.fromarray(np.asarray(m,np.float32),mode='F').resize((64,64),Image.Resampling.BILINEAR),np.float32) for m in relevance])
    if not np.isfinite(maps).all(): raise ValueError('nonfinite relevance')
    maps=np.clip(maps,0,1)
    def pool(arr):
        return arr.reshape(8,8,8,8,-1).mean((1,3)).reshape(-1)
    result=[]
    for predicted,actual,query in [(0,1,4),(2,3,5)]:
        union=np.maximum.reduce([maps[predicted],maps[actual],maps[query]])
        # A soft floor avoids treating a poorly segmented displaced entity as
        # zero residual. No unmasked full-frame embedding is added.
        weight=.1+.9*union
        delta=(rgb[actual]-rgb[predicted])*weight[...,None]
        gray=rgb.mean(-1)
        def edges(a):
            dy,dx=np.gradient(a);return np.hypot(dx,dy)
        edge=(edges(gray[actual])-edges(gray[predicted]))*weight
        evidence=np.concatenate([delta,np.abs(delta),edge[...,None],
            maps[predicted,...,None],maps[actual,...,None],maps[query,...,None]],axis=-1)
        result.append(pool(evidence))
    out=np.concatenate(result).astype(np.float32)
    if out.shape!=(1280,) or not np.isfinite(out).all():raise RuntimeError('dense relation contract violation')
    return out
