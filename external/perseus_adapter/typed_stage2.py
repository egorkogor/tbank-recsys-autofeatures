"""Optional H6 storage: independent per-kind history quotas, stable day snapshots."""
import json
from pathlib import Path
import time
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'external/perseus_stage2'
def cutoff(day):return day*86400-6395

def prepare_typed(seq):
    started=time.perf_counter();cache=BASE/f'cache_typed_{seq}';cache.mkdir(parents=True,exist_ok=True)
    factor=seq//256
    quotas={0:176*factor,1:64*factor,2:16*factor}
    horizons={0:30,1:365,2:90}
    ev=pl.scan_parquet(ROOT/'data/processed/events.parquet').with_row_index('_eid').filter(pl.col('kind').is_in([0,1,2])&(pl.col('t')<cutoff(293)))
    cols=['uid','item','artist','album','t','kind','org','pr','_eid']
    past=[]
    for kind,quota in quotas.items():
        past.append(ev.filter((pl.col('t')<cutoff(273))&(pl.col('kind')==kind)).select(cols).group_by('uid').tail(quota).collect(engine='streaming'))
    recent=ev.filter(pl.col('t')>=cutoff(273)).select(cols).collect(engine='streaming')
    history=pl.concat(past+[recent]).sort('uid','t','_eid')
    lengths=pl.read_parquet(ROOT/'data/processed/items.parquet').select('item',pl.col('track_len').cast(pl.Float32))
    history=history.join(lengths,on='item',how='left',maintain_order='left')
    for col in cols+['track_len']:np.save(cache/f'{col}.npy',history[col].to_numpy())
    catalog=pl.read_parquet(BASE/'cache/catalog.parquet');catalog.write_parquet(cache/'catalog.parquet')
    queries=(pl.scan_parquet(BASE/'cache/rows_no_labels.parquet').filter(pl.col('day').is_between(273,293))
             .select('uid','day').unique().sort('uid','day').collect().with_row_index('qid'))
    spans=history.select('uid').with_row_index('i').group_by('uid').agg(pl.col('i').min().alias('lo'),(pl.col('i').max()+1).alias('hi'))
    queries=queries.join(spans,on='uid',how='left',maintain_order='left')
    matrix=np.lib.format.open_memmap(cache/'history_indices.npy',mode='w+',dtype=np.int32,shape=(len(queries),seq));matrix[:]=-1
    times=history['t'].to_numpy();kinds=history['kind'].to_numpy();counts=np.zeros(len(queries),dtype=np.int64)
    for (uid,),group in queries.group_by('uid',maintain_order=True):
        if group['lo'][0] is None:continue
        lo,hi=int(group['lo'][0]),int(group['hi'][0])
        by_kind={kind:np.flatnonzero(kinds[lo:hi]==kind)+lo for kind in quotas}
        for row in group.iter_rows(named=True):
            cut=cutoff(row['day']);parts=[]
            for kind,quota in quotas.items():
                indices=by_kind[kind];ts=times[indices]
                end=int(np.searchsorted(ts,cut,'left'));start=max(int(np.searchsorted(ts,cut-horizons[kind]*86400,'left')),end-quota)
                parts.append(indices[start:end])
            chosen=np.sort(np.concatenate(parts));n=len(chosen)
            if n:assert times[chosen[-1]]<cut;matrix[row['qid'],-n:]=chosen
            counts[row['qid']]=n
    matrix.flush()
    queries.drop('lo','hi').with_columns(pl.lit(0).cast(pl.Int64).alias('start'),pl.Series('end',counts)).write_parquet(cache/'queries.parquet')
    result={'success':True,'seconds':time.perf_counter()-started,'sequence':seq,'quotas':quotas,'horizons_days':horizons,
            'queries':len(queries),'no_history_queries':int(np.sum(counts==0)),'strict_cutoff_assertions':True,'labels_read':False}
    (cache/'ready.json').write_text(json.dumps(result,indent=2))
    return cache

def typed_raw_class(base_class):
    class TypedRaw(base_class):
        def __init__(self,seq):
            cache=BASE/f'cache_typed_{seq}'
            if not (cache/'ready.json').exists():prepare_typed(seq)
            super().__init__(cache)
            self.typed_indices=np.load(cache/'history_indices.npy',mmap_mode='r')
        def history_indices(self,qid,limit):
            ix=self.typed_indices[qid];return ix[ix>=0][-limit:]
        def observation_mask(self,qids,limit):
            mask=np.zeros(len(self.data['t']),dtype=bool)
            indices=self.typed_indices[qids].ravel();mask[indices[indices>=0]]=True
            return mask
    return TypedRaw
