import os
os.environ['POLARS_MAX_THREADS']='4'
import json
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
import psutil

root=Path(__file__).resolve().parents[2]
r=pl.scan_parquet(root/'artifacts/rows.parquet')
e=pl.scan_parquet(root/'data/processed/events.parquet')
out={'rows_schema':{k:str(v) for k,v in r.collect_schema().items()},
     'events_schema':{k:str(v) for k,v in e.collect_schema().items()},
     'rows_by_day':r.select('day','uid').group_by('day').agg(pl.len().alias('rows'),pl.col('uid').n_unique().alias('users')).sort('day').collect().to_dicts(),
     'history_users':e.select(pl.col('uid').n_unique()).collect().item(),
     'event_rows':pq.ParquetFile(root/'data/processed/events.parquet').metadata.num_rows,
     'affinity':psutil.Process().cpu_affinity(),'priority':str(psutil.Process().nice()),
     'ram_available':psutil.virtual_memory().available}
print(json.dumps(out,indent=2))
(root/'external/stage2_inspect.json').write_text(json.dumps(out,indent=2))
