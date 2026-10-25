"""Read back delivered R3 validation; only val-A labels are accessed."""
import json
from pathlib import Path
import types
import sys
import polars as pl
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'external/perseus_stage2';RUN=ROOT/'artifacts/runs/r3_perseus'
if '--typed' in sys.argv:RUN=ROOT/'artifacts/runs/r3_perseus_typed'
if '--long' in sys.argv:RUN=ROOT/'artifacts/runs/r3_perseus_long'
if '--authors' in sys.argv:RUN=ROOT/'artifacts/runs/r3_perseus_authors'
pred=pl.read_parquet(RUN/'pred_val.parquet')
keys=pl.scan_parquet(BASE/'cache/rows_no_labels.parquet').filter(pl.col('day').is_between(287,293)).select('row').collect()
assert pred.select('row').sort('row').equals(keys.sort('row'))
assert pred['row'].n_unique()==len(pred)
val=(pl.scan_parquet(ROOT/'artifacts/rows.parquet').filter(pl.col('day').is_between(287,293)&(pl.col('val_half')=='A'))
     .select('row','uid','day',pl.col('y').cast(pl.Int8)).collect())
scored=val.join(pred,on='row',how='left').with_columns(pl.col('score').fill_null(0.5))
module=types.ModuleType('canonical_metrics')
source=(ROOT/'src/tbrec/metrics.py').read_text(encoding='utf-8').replace('from .config import SEED','SEED=20261024')
exec(compile(source,str(ROOT/'src/tbrec/metrics.py'),'exec'),module.__dict__)
metrics=module.summary(scored,'score')
selection=json.loads((BASE/'selection.json').read_text(encoding='utf-8'))
if '--authors' in sys.argv:selection=json.loads((BASE/'r3_authors/result.json').read_text(encoding='utf-8'))
if '--long' in sys.argv:selection=json.loads((BASE/'r3_long/result.json').read_text(encoding='utf-8'))
metrics['selected_epoch_val_A_gauc']=selection['best_gauc']
metrics['delivered_minus_selection_gauc']=metrics['gauc']-selection['best_gauc']
if '--authors' in sys.argv or '--long' in sys.argv:assert abs(metrics['delivered_minus_selection_gauc'])<1e-6
(RUN/'validation.json').write_text(json.dumps(metrics,indent=2))
with (BASE/'label_access.jsonl').open('a',encoding='utf-8') as f:
    f.write(json.dumps({'start':287,'end':293,'val_A_only':True,'max_day':293,'rows':len(val),'purpose':'H6_readback' if '--typed' in sys.argv else 'R3_readback'})+'\n')
print(json.dumps(metrics,indent=2))
