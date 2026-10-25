"""Independent output/key/window checks; the only labels read are val-A 287..293."""
import json
from pathlib import Path
import types
import polars as pl
import yaml
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'external/perseus_stage2'
RUN=ROOT/'artifacts/runs/r3_perseus'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    selection=read(BASE/'selection.json')
    rows=pl.read_parquet(BASE/'cache/rows_no_labels.parquet')
    tests={}
    outputs=[('r3_val',287,293,RUN/'pred_val.parquet','score'),
             ('r4_val',273,293,ROOT/'artifacts/features/perseus_val.parquet','perseus_score'),
             ('r4_oot',280,300,ROOT/'artifacts/features/perseus_oot.parquet','perseus_score')]
    if (RUN/'pred_oot.parquet').exists():outputs.append(('r3_oot',294,300,RUN/'pred_oot.parquet','score'))
    typed_path=ROOT/'artifacts/runs/r3_perseus_typed/pred_val.parquet'
    if typed_path.exists():outputs.append(('h6_val',287,293,typed_path,'score'))
    for name,start,end,path,score in outputs:
        expected=rows.filter(pl.col('day').is_between(start,end))
        out=pl.read_parquet(path)
        assert out.columns==['row',score],(name,out.columns)
        assert out.schema['row']==pl.UInt32
        assert out['row'].n_unique()==len(out)==len(expected)
        assert out.select('row').sort('row').equals(expected.select('row').sort('row'))
        assert out[score].drop_nulls().is_finite().all()
        assert out[score].drop_nulls().is_between(0,1).all()
        combined=expected.join(out,on='row')
        assert combined.group_by('uid','day','item').agg(pl.col(score).n_unique().alias('n'))['n'].max()==1
        tests[name]={'rows':len(out),'null':out[score].null_count(),'null_fraction':out[score].null_count()/len(out),
                     'exact_row_keys':True,'duplicates_same_score':True,'finite_probability_scores':True}
    accesses=[json.loads(s) for s in (BASE/'label_access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(x['end']<=293 and x['max_day']<=293 for x in accesses)
    assert all(not x['val_A_only'] or (x['start'],x['end'])==(287,293) for x in accesses)
    configs=[]
    models=selection.get('models',{'r4_val':'r4_val','r4_oot':'r4_oot','r3_oot':'r3_oot'})
    for model,start,end in [(selection['model'],273,286),(models['r4_val'],259,272),(models['r4_oot'],266,279),(models['r3_oot'],280,293)]:
        result=read(BASE/model/'result.json')
        assert result['status']=='complete' and result['train_days']==[start,end]
        assert result['affinity']==[11,12,13,14,15]
        assert result['sequence']==selection['sequence'] and result['dim']==selection['dim']
        if model!=selection['model']:assert result['completed_epochs']==selection['chosen_epochs']
        config=yaml.safe_load((BASE/model/'config.yaml').read_text())
        config['training'].pop('num_epochs',None);config['training'].pop('early_stopping',None)
        configs.append(config)
        assert (BASE/model/'checkpoint/backbone/weights.safetensors').exists()
        for relative in ['backbone/weights.safetensors','head/layer/weights.safetensors']:
            with safe_open(BASE/model/'checkpoint'/relative,framework='pt',device='cpu') as weights:
                assert len(list(weights.keys()))>0
    assert all(c==configs[0] for c in configs)
    if (RUN/'pred_oot.parquet').exists():
        assert read(BASE/models['r3_oot']/'r3_oot_score.json')['freeze_gate_present_at_start'] is True
    # Recompute GAUC from the delivered R3 val file, using only A and days <=293.
    val=(pl.scan_parquet(ROOT/'artifacts/rows.parquet')
         .filter(pl.col('day').is_between(287,293)&(pl.col('val_half')=='A'))
         .select('row','uid','day',pl.col('y').cast(pl.Int8)).collect())
    with (BASE/'label_access.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'start':287,'end':293,'val_A_only':True,'max_day':293,'rows':len(val),'purpose':'delivered_R3_validation'})+'\n')
    scored=val.join(pl.read_parquet(RUN/'pred_val.parquet'),on='row',how='left').with_columns(pl.col('score').fill_null(0.5))
    module=types.ModuleType('canonical_metrics')
    source=(ROOT/'src/tbrec/metrics.py').read_text(encoding='utf-8').replace('from .config import SEED','SEED=20261024')
    exec(compile(source,str(ROOT/'src/tbrec/metrics.py'),'exec'),module.__dict__)
    metrics=module.summary(scored,'score')
    metrics['selected_epoch_val_A_gauc']=selection['best_gauc']
    metrics['delivered_minus_selection_gauc']=metrics['gauc']-selection['best_gauc']
    assert abs(metrics['delivered_minus_selection_gauc'])<0.002
    (RUN/'validation.json').write_text(json.dumps(metrics,indent=2))
    result={'success':True,'outputs':tests,'frozen_configs_match':True,'no_OOT_labels_in_access_log':True,
            'val_A':metrics,'r3_oot_scored':(RUN/'pred_oot.parquet').exists()}
    (BASE/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
