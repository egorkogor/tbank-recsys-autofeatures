"""Independent vectorized cache checks; no labels or torch initialization."""
import json
from pathlib import Path
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'external/perseus_stage2/cache_long'
q=pl.read_parquet(CACHE/'queries.parquet').sort('qid')
t=np.load(CACHE/'t.npy',mmap_mode='r');uid=np.load(CACHE/'uid.npy',mmap_mode='r')
start=q['start'].to_numpy();end=q['end'].to_numpy();u=q['uid'].to_numpy()
cut=q['day'].to_numpy().astype(np.int64)*86400-6395
ok=end>start
assert q['qid'].to_list()==list(range(len(q)))
assert q.select('uid','day').n_unique()==len(q)
assert q['day'].min()==231 and q['day'].max()==300
assert np.all((end-start>=0)&(end-start<=512))
assert np.all(t[end[ok]-1]<cut[ok])
assert np.all(uid[start[ok]]==u[ok]) and np.all(uid[end[ok]-1]==u[ok])
inside=end<len(t);same=np.zeros(len(q),dtype=bool);same[inside]=uid[end[inside]]==u[inside]
assert np.all(t[end[same]]>=cut[same])
assert 'y' not in pl.read_parquet_schema(CACHE/'rows_no_labels.parquet')
result={'success':True,'queries':len(q),'strict_cutoffs':True,'user_boundaries':True,'max_events':512,'day_range':[231,300],'labels_read':False}
(CACHE/'contract_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
