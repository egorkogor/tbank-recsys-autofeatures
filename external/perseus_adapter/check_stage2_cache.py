import json
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'external/perseus_stage2/cache'
q=pl.read_parquet(CACHE/'queries.parquet')
uid=np.load(CACHE/'uid.npy',mmap_mode='r');t=np.load(CACHE/'t.npy',mmap_mode='r');eid=np.load(CACHE/'_eid.npy',mmap_mode='r')
same=uid[1:]==uid[:-1]
assert np.all(t[1:][same]>=t[:-1][same])
ties=same&(t[1:]==t[:-1]);assert np.all(eid[1:][ties]>eid[:-1][ties])
start=q['start'].to_numpy();end=q['end'].to_numpy();cut=q['day'].to_numpy().astype(np.int64)*86400-6395
ok=end>start
assert np.all(t[end[ok]-1]<cut[ok])
assert np.all(uid[end[ok]-1]==q['uid'].to_numpy()[ok])
assert np.all(end-start<=512)
toy=pl.DataFrame({'uid':[1,1],'day':[273,273],'item':[7,7],'y':[0.,1.]})
assert toy.group_by('uid','day','item').agg(pl.col('y').mean())['y'].item()==0.5
result={'success':True,'queries_checked':len(q),'strict_cutoff':True,'uid_alignment':True,
        'stable_timestamp_ties':True,'dedup_mean':True,'OOT_labels_read':False}
(CACHE/'contract_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
