"""Final stage-3 acceptance checks, with no label reads or GPU allocation."""
import hashlib
import json
from pathlib import Path
import re
import polars as pl
from safetensors import safe_open
import yaml
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'external/perseus_stage2';RUN=ROOT/'artifacts/runs/r3_perseus'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
s=read(BASE/'selection.json');a=read(BASE/'r3_authors/result.json');old=read(BASE/'stage3_initial_selection.json')
assert a['status']=='complete' and a['train_days']==[273,286] and a['completed_epochs']<=3
assert (a['sequence'],a['dim'],a['layers'],a['batch'],a['accumulation'])==(512,256,4,8,8)
assert a['affinity']==[11,12,13,14,15] and a['priority']=='16384'
assert int((ROOT/'artifacts/perseus_len.txt').read_text())==s['sequence']
assert hashlib.sha256((RUN/'selected_config.yaml').read_bytes()).hexdigest()==s['selected_config_sha256']
with safe_open(BASE/'r3_authors/checkpoint/backbone/weights.safetensors',framework='np') as weights:
    layers={int(m.group(1)) for k in weights.keys() if (m:=re.search(r'layers\.(\d+)\.',k))}
    assert layers=={0,1,2,3}
    assert weights.get_slice('history_aggregator.layers.0.attention.qkv_proj.weight').get_shape()==[768,256]
assert abs(read(ROOT/'artifacts/runs/r3_perseus_authors/validation.json')['gauc']-a['best_gauc'])<1e-9
if a['best_gauc']<=old['best_gauc']:
    assert s['model']==old['model'] and s['selected_config_sha256']==old['selected_config_sha256']
else:assert s['model']=='r3_authors'
assert read(BASE/'verification.json')['success']
long=read(BASE/'r3_long/result.json')
assert long['status']=='complete' and long['train_days']==[231,286] and long['cache']=='cache_long'
assert read(BASE/'cache_long/contract_checks.json')['success']
assert abs(read(ROOT/'artifacts/runs/r3_perseus_long/validation.json')['gauc']-long['best_gauc'])<1e-9
def normalize(path):
    c=yaml.safe_load(path.read_text(encoding='utf-8'))
    c['training'].pop('num_epochs');c['training'].pop('early_stopping')
    return c
assert normalize(BASE/s['model']/'config.yaml')==normalize(BASE/'r3_long/config.yaml')
rows=pl.read_parquet(BASE/'cache/rows_no_labels.parquet').filter(pl.col('day').is_between(287,293))
checks={}
for name in ['r3_perseus_authors','r3_perseus_long']:
    pred=pl.read_parquet(ROOT/'artifacts/runs'/name/'pred_val.parquet')
    assert pred.columns==['row','score'] and pred.schema['row']==pl.UInt32
    assert len(pred)==len(rows)==pred['row'].n_unique()
    assert pred.select('row').sort('row').equals(rows.select('row').sort('row'))
    assert pred['score'].drop_nulls().is_finite().all() and pred['score'].drop_nulls().is_between(0,1).all()
    assert rows.join(pred,on='row').group_by('uid','day','item').agg(pl.col('score').n_unique())['score'].max()==1
    checks[name]={'rows':len(pred),'null':pred['score'].null_count(),'exact_keys':True,'duplicate_scores_equal':True}
accesses=[json.loads(line) for line in (BASE/'label_access.jsonl').read_text(encoding='utf-8').splitlines()]
assert all(x['end']<=293 and x['max_day']<=293 for x in accesses)
result={'success':True,'selected':s['model'],'selected_length':s['sequence'],
        'authors_actual_backbone_layers':sorted(layers),'authors_val_A':a['best_gauc'],
        'long_val_A':long['best_gauc'],'no_OOT_label_access':True,'outputs':checks}
(BASE/'stage3_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
