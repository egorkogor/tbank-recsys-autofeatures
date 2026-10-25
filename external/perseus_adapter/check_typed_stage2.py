"""Audit persisted H6 lookup indices, without reading pr or any labels."""
import json
from pathlib import Path
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[2]
cache=ROOT/'external/perseus_stage2/cache_typed_256'
q=pl.read_parquet(cache/'queries.parquet').sort('qid')
ix=np.load(cache/'history_indices.npy',mmap_mode='r');valid=ix>=0
safe=np.maximum(ix,0)
t=np.load(cache/'t.npy',mmap_mode='r')[safe]
kind=np.load(cache/'kind.npy',mmap_mode='r')[safe]
uid=np.load(cache/'uid.npy',mmap_mode='r')[safe]
cut=q['day'].to_numpy().astype(np.int64)*86400-6395
assert not ((t>=cut[:,None])&valid).any()
assert not ((uid!=q['uid'].to_numpy()[:,None])&valid).any()
for k,quota,horizon in [(0,176,30),(1,64,365),(2,16,90)]:
    mask=valid&(kind==k)
    assert (mask.sum(axis=1)<=quota).all()
    assert not ((t<cut[:,None]-horizon*86400)&mask).any()
assert not ((np.diff(ix,axis=1)<=0)&valid[:,1:]&valid[:,:-1]).any()
result={'success':True,'queries':len(q),'strict_cutoff':True,'correct_uid':True,'quotas_and_horizons':True,
        'stable_chronological_order':True,'labels_read':False}
(cache/'contract_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
