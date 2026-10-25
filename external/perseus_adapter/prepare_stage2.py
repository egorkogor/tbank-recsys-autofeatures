"""Read-only source preparation. No rows.y is projected here, including OOT."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import json
import time
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'external/perseus_stage2/cache'
DAY=86400
def cutoff(d): return d*DAY-6395

def main(first_day=259):
    started=time.perf_counter()
    CACHE.mkdir(parents=True,exist_ok=True)
    ev=pl.scan_parquet(ROOT/'data/processed/events.parquet').with_row_index('_eid')
    rows=pl.scan_parquet(ROOT/'artifacts/rows.parquet').select('row','uid','item','artist','album','day','val_half')
    rows.collect().write_parquet(CACHE/'rows_no_labels.parquet')
    early=(ev.filter(pl.col('day').is_between(first_day,272)&(pl.col('kind')==0)&(pl.col('org')==0))
           .select('uid','item','artist','album','day').collect())
    metadata=pl.concat([rows.select('item','artist','album').collect(),early.select('item','artist','album')]).unique('item').sort('item')
    lengths=pl.scan_parquet(ROOT/'data/processed/items.parquet').select('item',pl.col('track_len').cast(pl.Float32)).collect()
    metadata.join(lengths,on='item',how='left',maintain_order='left').write_parquet(CACHE/'catalog.parquet')
    queries=pl.concat([rows.select('uid','day').collect(),early.select('uid','day')]).unique().sort('uid','day').with_row_index('qid')
    del early,metadata
    print('Query count',len(queries),flush=True)
    cols=['uid','item','artist','album','t','kind','org','pr','_eid']
    eligible=ev.filter(pl.col('kind').is_in([0,1,2])&(pl.col('t')<cutoff(300)))
    past=(eligible.filter(pl.col('t')<cutoff(first_day)).select(cols).group_by('uid').tail(512).collect(engine='streaming'))
    recent=eligible.filter(pl.col('t')>=cutoff(first_day)).select(cols).collect(engine='streaming')
    history=pl.concat([past,recent]).sort('uid','t','_eid')
    del past,recent
    history=history.join(lengths,on='item',how='left',maintain_order='left')
    assert history['t'].max()<cutoff(300)
    for col in cols+['track_len']:
        np.save(CACHE/f'{col}.npy',history[col].to_numpy())
    spans=(history.select('uid').with_row_index('i').group_by('uid').agg(pl.col('i').min().alias('lo'),(pl.col('i').max()+1).alias('hi')))
    queries=queries.join(spans,on='uid',how='left',maintain_order='left')
    t=history['t'].to_numpy()
    starts=np.zeros(len(queries),dtype=np.int64)
    ends=np.zeros(len(queries),dtype=np.int64)
    for (uid,),group in queries.group_by('uid',maintain_order=True):
        ids=group['qid'].to_numpy()
        lo,hi=int(group['lo'][0]),int(group['hi'][0])
        cuts=group['day'].to_numpy().astype(np.int64)*DAY-6395
        end=lo+np.searchsorted(t[lo:hi],cuts,side='left')
        start=np.maximum(lo,end-512)
        starts[ids]=start
        ends[ids]=end
        ok=end>start
        assert np.all(t[end[ok]-1]<cuts[ok])
        inside=end<hi
        assert np.all(t[end[inside]]>=cuts[inside])
    queries=queries.drop('lo','hi').with_columns(pl.Series('start',starts),pl.Series('end',ends))
    queries.write_parquet(CACHE/'queries.parquet')
    stats={'success':True,'seconds':time.perf_counter()-started,'queries':len(queries),
           'history_events':len(history),'max_history_day':299,
           'rows_y_read':False,'cutoff_assertions_pass':True,
           'tie_break':['t','original_event_row_index'],
           'no_history_queries':int(np.sum(ends==starts))}
    (CACHE/'ready.json').write_text(json.dumps(stats,indent=2))
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--long',action='store_true');args=parser.parse_args()
    if args.long:CACHE=ROOT/'external/perseus_stage2/cache_long'
    main(231 if args.long else 259)
