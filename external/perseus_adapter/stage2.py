"""Day-cutoff ranking adapter. Native Perseus model, loss, optimizer and fit loop.

Only the sample loader and validation callback are adapted: compact chronological
memory maps replace EventHub cross-joins, and validation uses the task's val-A GAUC.
No OOT y is projected, derived, logged or evaluated.
"""
import os
for key in ['POLARS_MAX_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'4')
os.environ['ACCELERATE_MIXED_PRECISION']='bf16'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import argparse
import contextlib
import importlib
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
EXTERNAL=ROOT/'external'
for cache_key,cache_name in [('CUDA_CACHE_PATH','.cuda-cache'),('TRITON_CACHE_DIR','.triton-cache'),('TORCHINDUCTOR_CACHE_DIR','.torchinductor-cache')]:
    os.environ.setdefault(cache_key,str(EXTERNAL/cache_name))
sys.path.insert(0,str(EXTERNAL))
from probe_runtime import configure_tempdirs
configure_tempdirs()
import numpy as np
import polars as pl
import psutil
import torch
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
from perseus import api,core,utils
from perseus.core.dataloaders.samples import _CollatedSamples

# Import the canonical metrics without executing tbrec.__init__ (which sets affinity).
metrics_spec=importlib.util.spec_from_file_location('tbrec_metrics_readonly',ROOT/'src/tbrec/metrics.py')
metrics_source=(ROOT/'src/tbrec/metrics.py').read_text(encoding='utf-8').replace('from .config import SEED','SEED = 20261024')
metrics_module=type(sys)('tbrec_metrics_readonly')
exec(compile(metrics_source,str(ROOT/'src/tbrec/metrics.py'),'exec'),metrics_module.__dict__)

CACHE=EXTERNAL/'perseus_stage2/cache'
BASE=EXTERNAL/'perseus_stage2'
RUN=ROOT/'artifacts/runs/r3_perseus'
FEATURE_SOURCE={'item_id':'item','artist_id':'artist','album_id':'album','is_organic':'org',
                'event':'kind','played_ratio_pct':'pr','track_length_seconds':'track_len'}
ID_FEATURES=['item_id','artist_id','album_id','is_organic','event']
NUM_FEATURES=['played_ratio_pct','track_length_seconds']
def cutoff(d): return d*86400-6395
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def log(msg): print(msg,flush=True)

def label_rows(start,end,val_a=False):
    if not (231<=start<=end<=293): raise ValueError('Forbidden label window')
    # Predicate is applied before projection/collection of y. No full-y materialization.
    if start>=273:
        scan=pl.scan_parquet(ROOT/'artifacts/rows.parquet').filter(pl.col('day').is_between(start,end))
        if val_a: scan=scan.filter(pl.col('val_half')=='A')
        result=scan.select('row','uid','day','item',pl.col('y').cast(pl.Float32)).collect()
    else:
        if end>=273:
            return pl.concat([label_rows(start,272),label_rows(273,end)],how='diagonal_relaxed')
        result=(pl.scan_parquet(ROOT/'data/processed/events.parquet')
                .filter(pl.col('day').is_between(start,end)&(pl.col('kind')==0)&(pl.col('org')==0))
                .select('uid','day','item',(pl.col('pr')>=50).cast(pl.Float32).alias('y')).collect())
    assert result['day'].min()>=start and result['day'].max()<=end
    with (BASE/'label_access.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps({'start':start,'end':end,'val_A_only':val_a,'rows':len(result),'max_day':result['day'].max()})+'\n')
    return result

class Raw:
    def __init__(self,cache=CACHE):
        self.queries=pl.read_parquet(cache/'queries.parquet').sort('qid')
        self.data={key:np.load(cache/f'{key}.npy',mmap_mode='r') for key in set(FEATURE_SOURCE.values())|{'t'}}
        self.starts=self.queries['start'].to_numpy()
        self.ends=self.queries['end'].to_numpy()
        self.days=self.queries['day'].to_numpy()
        self.catalog=(pl.read_parquet(cache/'catalog.parquet').rename({'artist':'artist_id','album':'album_id','track_len':'track_length_seconds'}).sort('item'))
    def history_indices(self,qid,limit):
        end=self.ends[qid]
        return np.arange(max(self.starts[qid],end-limit),end,dtype=np.int64)
    def observation_mask(self,qids,limit):
        diff=np.zeros(len(self.data['t'])+1,dtype=np.int16)
        ends=self.ends[qids];starts=np.maximum(self.starts[qids],ends-limit)
        np.add.at(diff,starts,1);np.add.at(diff,ends,-1)
        return np.cumsum(diff,dtype=np.int32)[:-1]>0
    def links(self,rows,labels=False):
        x=rows.group_by('uid','day','item').agg(pl.col('y').mean() if labels else pl.lit(0.0).alias('y'))
        x=x.join(self.queries.select('uid','day','qid','start','end'),on=['uid','day'],how='left')
        assert not x['qid'].is_null().any()
        return x.filter(pl.col('end')>pl.col('start')).drop('start','end').sort('qid','item')

def make_config(seq,dim,epochs,select,layers=2,batch=None):
    encoders={'item':{'type':'id','preprocessor':{'top_k':250000}},
              'artist':{'type':'id','preprocessor':{'top_k':100000}},
              'album':{'type':'id','preprocessor':{'top_k':100000}},
              'organic':{'type':'id'},'ratio':{'type':'ple','preprocessor':{'max_bins':32}},
              'length':{'type':'ple','preprocessor':{'max_bins':32}}}
    features={}
    for feature,encoder in [('item_id','item'),('artist_id','artist'),('album_id','album'),('is_organic','organic'),
                            ('played_ratio_pct','ratio'),('track_length_seconds','length')]:
        features[feature]={'located_in':{'event':True,'artifacts':feature in ['artist_id','album_id','track_length_seconds']},'encoder':encoder}
    features['item']={'located_in':{'artifacts':True},'encoder':'item'}
    cfg={'task':{'type':'ranking','metrics':{},'head':{'allow_unknown_items':True,'label_to_weight':{'listen_plus':1.0}}},
         'events':{name:{'attributes':{f:None for f in features if f!='item'},'priority':0} for name in ['listen','like','dislike']},
         'max_events_per_sequence':seq,'features':features,'encoders':encoders,
         'backbone':{'dim':dim,'history_aggregator':{'type':'modern_bert','params':{'num_layers':layers,'num_heads':4,'dropout':0.1}},'event_aggregator':{'type':'sum'}},
         'training':{'num_epochs':epochs,'log_every_n_train_steps':100,
                     'dataloader':{'batch_size':16 if dim==128 else 8,'num_workers':0,'shuffle':True,'pin_memory':True},
                     'test_dataloader':{'batch_size':16 if dim==128 else 8,'num_workers':0,'pin_memory':True},
                     'optimizer':{'params':{'lr':0.0003}},
                     'early_stopping':{'metric':'gauc','mode':'max','patience':1} if select else None},
         'inference':{'backbone_dataloader':{'batch_size':16 if dim==128 else 8,'num_workers':0},'head_dataloader':{'batch_size':16 if dim==128 else 8,'num_workers':0}}}
    if batch is not None:
        for section in ['dataloader','test_dataloader']:cfg['training'][section]['batch_size']=batch
        for section in ['backbone_dataloader','head_dataloader']:cfg['inference'][section]['batch_size']=batch
    return core.Config.model_validate(cfg)

def basis(links):
    return (links.group_by('uid','day','qid',maintain_order=True)
            .agg(target=pl.struct('item',labels=pl.struct(pl.col('y').alias('listen_plus'))))
            .with_columns(pl.col('uid').cast(pl.String).alias('client_id'),
                          pl.from_epoch(pl.col('day').cast(pl.Int64)*86400-6395,time_unit='s').cast(pl.Datetime('ns')).alias('timestamp')))

def prepare_model(raw,train,config,directory):
    started=time.perf_counter()
    links=raw.links(train,labels=True)
    train_basis=basis(links)
    observers={name:core.encoders.registry[desc.type].observer() for name,desc in config.encoders.items()}
    qids=links['qid'].unique().to_numpy()
    mask=raw.observation_mask(qids,config.max_events_per_sequence)
    assert raw.data['t'][mask].max()<cutoff(int(train['day'].max())+1)
    for feature,encoder in config.feature_to_encoder(event=True).items():
        values=raw.data[FEATURE_SOURCE[feature]][mask]
        if feature=='played_ratio_pct': values=values[values>=0].astype(np.float32)
        if feature=='track_length_seconds': values=values[np.isfinite(values)]
        observers[encoder].observe(pl.Series(values))
        log(f'observed {feature}: {len(values)}')
    del mask
    train_art=core.tasks.ranking.Artifacts(raw.catalog.clone())
    test_art=core.tasks.ranking.Artifacts(raw.catalog.clone())
    artifact_observers={feature:observers[encoder] for feature,encoder in config.feature_to_encoder(artifacts=True).items()}
    task_pre=core.tasks.ranking.Preprocessor.fit(train_basis['target'],train_art,artifact_observers)
    preprocessors={name:core.encoders.registry[desc.type].preprocessor.fit(observers[name],**desc.preprocessor) for name,desc in config.encoders.items()}
    del observers
    prepared=core.checkpoint.Prepared(config,preprocessors)
    prepared.save(directory/'prepared')
    code=np.lib.format.open_memmap(directory/'event_codes.npy',mode='w+',dtype=np.int32,shape=(len(raw.data['t']),len(ID_FEATURES)))
    for i,feature in enumerate(ID_FEATURES):
        pre=preprocessors[config.feature_to_encoder(event=True)[feature]]
        code[:,i]=pre.static_transform(pl.Series(raw.data[FEATURE_SOURCE[feature]])).to_numpy()
        code.flush()
        log('encoded '+feature)
    art_pre={feature:preprocessors[encoder] for feature,encoder in config.feature_to_encoder(artifacts=True).items()}
    train_art.static_transform(art_pre);test_art.static_transform(art_pre)
    links=links.join(task_pre.items.select('item',pl.col('index').alias('item_index')),on='item',how='left',maintain_order='left')
    assert not links['item_index'].is_null().any()
    dataset=core.Dataset(train_basis,train_art,train_basis.head(0),test_art,None,task_pre)
    seconds=time.perf_counter()-started
    return prepared,dataset,links,code,seconds

class FastSamples(Dataset):
    def __init__(self,raw,links,codes,config,preprocessors):
        self.raw,self.links,self.codes,self.config,self.preprocessors=raw,links.sort('qid','item'),codes,config,preprocessors
        self.groups=(self.links.select('qid').with_row_index('i').group_by('qid',maintain_order=True).agg(pl.col('i').min().alias('offset'),pl.len().alias('n')))
        self.qids=self.groups['qid'].to_numpy();self.offsets=self.groups['offset'].to_numpy();self.counts=self.groups['n'].to_numpy()
        self.indices=self.links['item_index'].to_numpy().astype(np.int32)
        self.labels=self.links['y'].to_numpy().astype(np.float32)
    def __len__(self): return len(self.groups)
    def __getitem__(self,i): return i
    def collate(self,indices):
        qids=self.qids[indices]
        histories=[self.raw.history_indices(int(qid),self.config.max_events_per_sequence) for qid in qids]
        lengths=np.asarray([len(ix) for ix in histories]);size=int(lengths.max());b=len(indices)
        ids=np.zeros((b,size,len(ID_FEATURES)),dtype=np.int32)
        nums=np.full((b,size,2),np.nan,dtype=np.float32)
        positions=np.zeros((b,size),dtype=np.int32);times=np.zeros((b,size),dtype=np.float32)
        row_indices=[];item_indices=[];labels=[]
        for j,(i,ix,n) in enumerate(zip(indices,histories,lengths)):
            assert self.raw.data['t'][ix[-1]]<cutoff(int(self.raw.days[qids[j]]))
            ids[j,-n:]=self.codes[ix]
            pr=self.raw.data['pr'][ix].astype(np.float32);pr[pr<0]=np.nan
            nums[j,-n:,0]=pr;nums[j,-n:,1]=self.raw.data['track_len'][ix]
            positions[j,-n:]=np.arange(n,0,-1,dtype=np.int32)
            times[j,-n:]=self.raw.data['t'][ix]
            offset,count=int(self.offsets[i]),int(self.counts[i])
            row_indices.extend([j]*count);item_indices.extend(self.indices[offset:offset+count]);labels.extend(self.labels[offset:offset+count])
        features={name:torch.from_numpy(ids[:,:,k].copy()) for k,name in enumerate(ID_FEATURES)}
        for k,name in enumerate(NUM_FEATURES):
            pre=self.preprocessors[self.config.feature_to_encoder(event=True)[name]]
            features[name]=pre.dynamic_transform(pl.Series(nums[:,:,k].ravel())).reshape(b,size,-1)
        target=(row_indices,torch.tensor(np.asarray(item_indices),dtype=torch.int32),torch.tensor(np.asarray(labels).reshape(-1,1),dtype=torch.float32))
        return _CollatedSamples(features,torch.from_numpy(positions),torch.from_numpy(times),
            torch.tensor(self.raw.days[qids].astype(np.int64)*86400-6395,dtype=torch.float32),None,{'qid':qids.tolist()},target)

def loader(raw,links,codes,checkpoint,train=False):
    ds=FastSamples(raw,links,codes,checkpoint.config,checkpoint.encoder_to_preprocessor)
    dl=DataLoader(ds,batch_size=checkpoint.config.training.dataloader.batch_size,shuffle=train,
                  num_workers=0,pin_memory=True,collate_fn=ds.collate)
    return dl

def inference_links(raw,rows):
    return raw.links(rows).join(raw.catalog.select('item').with_row_index('item_index'),on='item',how='left',maintain_order='left')

@torch.no_grad()
def predict_loader(dl,raw,checkpoint,artifacts,transformed,embedders,backbone,layer,accelerator):
    for embedder in embedders.values():embedder.eval()
    backbone.eval();layer.eval()
    feature_embed={feature:embedders[encoder] for feature,encoder in checkpoint.config.feature_to_encoder(event=True).items()}
    artifact_embed={feature:embedders[encoder] for feature,encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()}
    item_ids=artifacts.items['_item'].to_numpy()
    parts=[]
    for batch in dl:
        with accelerator.autocast():
            u=backbone(feature_embed,batch.events_features,batch.events_positions,batch.events_timestamps,batch.sample_timestamp,None)
            embedded=artifacts.embed(transformed,artifact_embed,batch.target)
            ri,ii,_=batch.target
            logits=layer(u[ri],embedded)
            # Accelerate unwraps saved layers. Promote logits before sigmoid so
            # inference keeps the validation callback's FP32 probability precision.
            score=logits.float().sigmoid().cpu().numpy().ravel()
            if not parts and isinstance(checkpoint,core.checkpoint.Trained):
                flat_items=item_ids[ii.cpu().numpy()]
                extra={'items':[[int(flat_items[k]) for k,r in enumerate(ri) if r==j] for j in range(len(u))]}
                native=checkpoint.head.predict(u,embedded,extra)
                native_values=[]
                for j in range(len(u)):
                    by_item={int(row['item']):row['score'] for row in native[j].to_list()}
                    native_values.extend(by_item[item] for item in extra['items'][j])
                error=float(np.max(np.abs(np.asarray(native_values)-score)))
                assert error<0.002,('Native ranking Head parity failed',error)
                checkpoint._stage2_native_head_error=error
        assert np.isfinite(score).all()
        qids=np.asarray(batch.extras['qid'])[np.asarray(ri)]
        candidates=item_ids[ii.cpu().numpy()]
        parts.append(pl.DataFrame({'qid':qids.astype(np.uint32),'item':candidates,'score':score}))
        if len(parts)%500==0:log(f'inference batches completed: {len(parts)} / {len(dl)}')
    if not parts:return pl.DataFrame(schema={'qid':pl.UInt32,'item':pl.UInt32,'score':pl.Float32})
    return pl.concat(parts).join(raw.queries.select('qid','uid','day'),on='qid',how='left')

def attach(rows,pred):
    if 'uid' not in pred.columns:
        return rows.with_columns(score=pl.lit(None,dtype=pl.Float32))
    out=rows.join(pred.select('uid','day','item','score'),on=['uid','day','item'],how='left',maintain_order='left')
    assert len(out)==len(rows)
    return out

def train(args):
    os.environ['ACCELERATE_GRADIENT_ACCUMULATION_STEPS']=str(args.accumulation)
    directory=BASE/args.model;directory.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261024)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    log(f'affinity={psutil.Process().cpu_affinity()} priority={psutil.Process().nice()}')
    if args.typed:
        from typed_stage2 import typed_raw_class
        raw=typed_raw_class(Raw)(args.seq)
    else:raw=Raw(BASE/args.cache)
    labels=label_rows(args.start,args.end)
    config=make_config(args.seq,args.dim,args.epochs,args.select,args.layers,args.batch)
    config.save(directory/'config.yaml')
    prepared,dataset,train_links,codes,prepare_seconds=prepare_model(raw,labels,config,directory)
    val=label_rows(287,293,val_a=True) if args.select else None
    val_links=inference_links(raw,val) if val is not None else None
    native=importlib.import_module('perseus.api.fit_model')
    state={'model':args.model,'train_days':[args.start,args.end],'sequence':args.seq,'dim':args.dim,
           'layers':args.layers,'batch':config.training.dataloader.batch_size,'accumulation':args.accumulation,'cache':args.cache,
           'prepare_seconds':prepare_seconds,'train_rows':len(labels),'train_pairs':len(train_links),
           'train_samples':len(dataset.train_samples),'epochs':[], 'status':'training','typed_history':args.typed}
    dump(directory/'progress.json',state)
    stop_ref={}
    original_stop=native._create_early_stopping
    def create_stop(checkpoint):
        result=original_stop(checkpoint);stop_ref['object']=result;return result
    def create_loaders(acc,checkpoint,_dataset):
        assert acc.gradient_accumulation_steps==args.accumulation
        log(f'gradient_accumulation_steps={acc.gradient_accumulation_steps}')
        tr=acc.prepare_data_loader(loader(raw,train_links,codes,checkpoint,True))
        va=acc.prepare_data_loader(loader(raw,val_links,codes,checkpoint)) if val_links is not None else []
        return tr,va
    original_train=native._train_epoch
    def epoch_train(*a,**kw):
        torch.cuda.synchronize();t=time.perf_counter()
        result=original_train(*a,**kw)
        torch.cuda.synchronize()
        state['epochs'].append({'epoch':a[0],'train_seconds':time.perf_counter()-t})
        dump(directory/'progress.json',state)
        return result
    def epoch_test(epoch,cache_dir,acc,embedders,backbone,layer,head,evaluator,dl,transformed,checkpoint,ds,tracker):
        if val is None:return {}
        t=time.perf_counter()
        pred=predict_loader(dl,raw,checkpoint,ds.test_artifacts,transformed,embedders,backbone,layer,acc)
        scored=attach(val,pred)
        missing=scored['score'].null_count()
        # Fixed 0.5 only for metric calculation; exported missing scores remain null.
        metrics=metrics_module.summary(scored.with_columns(pl.col('score').fill_null(0.5),pl.col('y').cast(pl.Int8)),'score')
        result={'gauc':{'overall':metrics['gauc']}}
        state['epochs'][-1].update(val_seconds=time.perf_counter()-t,val_A_gauc=metrics['gauc'],val_A_missing=missing)
        if metrics['gauc']>state.get('best_gauc',float('-inf')):
            state.update(best_gauc=metrics['gauc'],best_epoch=epoch)
        dump(directory/'progress.json',state)
        log('VAL_A '+json.dumps(state['epochs'][-1]))
        return result
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    with patch.object(native,'_create_dataloaders',create_loaders),patch.object(native,'_train_epoch',epoch_train),patch.object(native,'_test_epoch',epoch_test),patch.object(native,'_create_early_stopping',create_stop):
        trained=api.fit_model(prepared,dataset)
    trained.save(directory/'checkpoint')
    state.update(status='complete',fit_save_seconds=time.perf_counter()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 checkpoint_bytes=sum(p.stat().st_size for p in (directory/'checkpoint').rglob('*') if p.is_file()),
                 completed_epochs=len(state['epochs']),chosen_epochs=state.get('best_epoch',args.epochs),
                 affinity=psutil.Process().cpu_affinity(),priority=str(psutil.Process().nice()))
    dump(directory/'result.json',state)
    log(json.dumps(state,indent=2))

def score(args):
    allowed={'r3_val':(287,293),'r4_val':(273,293),'r4_oot':(280,300),'r3_oot':(294,300)}
    if (args.start,args.end)!=allowed[args.kind]:raise ValueError('Score window does not match the authorized task')
    freeze_gate_present=(ROOT/'artifacts/freeze.json').is_file()
    if args.kind=='r3_oot' and not freeze_gate_present:
        log('R3 OOT NOT SCORED: artifacts/freeze.json absent');return
    if args.kind=='r3_oot':
        freeze_manifest=json.loads((ROOT/'artifacts/freeze.json').read_text(encoding='utf-8'))
        assert not freeze_manifest.get('publication_only',False), 'publication-only manifest cannot authorize OOT'
    directory=BASE/args.model
    torch.set_num_threads(4)
    checkpoint=core.checkpoint.Trained.load(directory/'checkpoint')
    torch.cuda.reset_peak_memory_stats()
    checkpoint.head.allow_unknown_items=True
    if args.typed:
        from typed_stage2 import typed_raw_class
        raw=typed_raw_class(Raw)(checkpoint.config.max_events_per_sequence)
    else:raw=Raw(BASE/args.cache)
    rows=pl.read_parquet(BASE/args.cache/'rows_no_labels.parquet').filter(pl.col('day').is_between(args.start,args.end))
    links=inference_links(raw,rows)
    codes=np.load(directory/'event_codes.npy',mmap_mode='r')
    art=core.tasks.ranking.Artifacts(raw.catalog.clone())
    pre={feature:checkpoint.encoder_to_preprocessor[encoder] for feature,encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()}
    art.static_transform(pre);transformed=art.dynamic_transform(pre)
    acc=Accelerator()
    for embedder in checkpoint.encoder_to_embedder.values():embedder.to(utils.torch.device)
    checkpoint.backbone.to(utils.torch.device);checkpoint.head.layer.to(utils.torch.device)
    started=time.perf_counter()
    dl=acc.prepare_data_loader(loader(raw,links,codes,checkpoint))
    pred=predict_loader(dl,raw,checkpoint,art,transformed,checkpoint.encoder_to_embedder,checkpoint.backbone,checkpoint.head.layer,acc)
    out=attach(rows,pred).select('row','score').sort('row')
    if args.kind.startswith('r4'):
        path=ROOT/f'artifacts/features/perseus_{"val" if args.kind=="r4_val" else "oot"}.parquet'
        out=out.rename({'score':'perseus_score'})
    else:
        target_run=ROOT/'artifacts/runs/r3_perseus_typed' if args.typed else RUN
        if args.output_run:
            if not args.output_run.startswith('r3_perseus') or '/' in args.output_run or '\\' in args.output_run:raise ValueError('Output outside authorized runs')
            target_run=ROOT/'artifacts/runs'/args.output_run
        path=target_run/('pred_val.parquet' if args.kind=='r3_val' else 'pred_oot.parquet')
    path.parent.mkdir(parents=True,exist_ok=True);out.write_parquet(path)
    seconds=time.perf_counter()-started
    score_name='perseus_score' if args.kind.startswith('r4') else 'score'
    known=checkpoint.head.known_items['_item']
    result={'kind':args.kind,'model':args.model,'days':[args.start,args.end],'rows':len(out),
            'missing':out[score_name].null_count(),'missing_fraction':out[score_name].null_count()/len(out),
            'outside_train_item_dictionary_fraction':float((~rows['item'].is_in(known.implode())).mean()),
            'item_encoder_unk_row_fraction':float((~rows['item'].is_in(checkpoint.encoder_to_preprocessor['item'].values['value'].implode())).mean()),
            'inference_seconds':seconds,'inference_seconds_per_1000_rows':seconds*1000/len(out),
            'path':str(path),'metrics_computed':False,
            'freeze_gate_present_at_start':freeze_gate_present if args.kind=='r3_oot' else None,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'native_head_parity_max_error':getattr(checkpoint,'_stage2_native_head_error',None)}
    dump(directory/f'{args.kind}_score.json',result)
    log(json.dumps(result,indent=2))

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--model',required=True);tr.add_argument('--start',type=int,required=True);tr.add_argument('--end',type=int,required=True)
    tr.add_argument('--seq',type=int,default=256);tr.add_argument('--dim',type=int,default=128);tr.add_argument('--epochs',type=int,default=3);tr.add_argument('--select',action='store_true')
    tr.add_argument('--typed',action='store_true')
    tr.add_argument('--layers',type=int,default=2);tr.add_argument('--batch',type=int);tr.add_argument('--accumulation',type=int,default=1)
    tr.add_argument('--cache',default='cache',choices=['cache','cache_long'])
    sc=sub.add_parser('score');sc.add_argument('--model',required=True);sc.add_argument('--start',type=int,required=True);sc.add_argument('--end',type=int,required=True);sc.add_argument('--kind',choices=['r3_val','r4_val','r4_oot','r3_oot'],required=True)
    sc.add_argument('--typed',action='store_true')
    sc.add_argument('--cache',default='cache',choices=['cache','cache_long']);sc.add_argument('--output-run')
    args=parser.parse_args()
    if args.command=='train':train(args)
    else:score(args)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    main()
