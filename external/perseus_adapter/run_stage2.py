"""Sequential driver; no affinity/priority changes, no OOT label reads."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
(ROOT/'logs/experiment').mkdir(parents=True, exist_ok=True)
BASE=ROOT/'external/perseus_stage2'
RUN=ROOT/'artifacts/runs/r3_perseus'
REPORT=ROOT/'logs/experiment/perseus_stage2_report.md'
ENGINE=ROOT/'external/perseus_adapter/stage2.py'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def append(message):
    with REPORT.open('a',encoding='utf-8') as f:f.write('\n'+message+'\n')
    print(message,flush=True)
def call(args,name,optional=False):
    path=ROOT/f'logs/experiment/perseus_stage2_{name}.log'
    with path.open('w',encoding='utf-8') as out:
        result=subprocess.run([sys.executable,str(ENGINE)]+args,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
    if result.returncode:
        append(f'**Ошибка {name}**: exit={result.returncode}, лог `{path.relative_to(ROOT)}`.')
        if not optional:raise RuntimeError(name)
        return False
    return True
def model_args(model,start,end,selection,epochs=3):
    return ['train','--model',model,'--start',str(start),'--end',str(end),'--seq',str(selection['sequence']),
            '--dim',str(selection['dim']),'--epochs',str(epochs)]
def score(model,start,end,kind):
    call(['score','--model',model,'--start',str(start),'--end',str(end),'--kind',kind],kind+'_score')
    result=read(BASE/model/f'{kind}_score.json')
    append(f"{kind}: сохранено {result['rows']} строк, null={result['missing']} ({result['missing_fraction']:.4%}), "
           f"инференс {result['inference_seconds_per_1000_rows']:.3f} с / 1000 строк; метрики при этом не вычислялись.")
    return result
def costs():
    data={'global_prepare':read(BASE/'cache/ready.json'),'models':{},'inference':{},'selection':read(BASE/'selection.json'),
          'failed_attempts':read(BASE/'failed_attempts.json') if (BASE/'failed_attempts.json').exists() else []}
    for model in ['r3_small','r3_large','r4_val','r4_oot','r3_oot']:
        if (BASE/model/'result.json').exists():data['models'][model]=read(BASE/model/'result.json')
        for kind in ['r3_val','r4_val','r4_oot','r3_oot']:
            path=BASE/model/f'{kind}_score.json'
            if path.exists():data['inference'][kind]=read(path)
    write(RUN/'cost.json',data)
def main():
    RUN.mkdir(parents=True,exist_ok=True)
    small=read(BASE/'r3_small/result.json')
    append(f"R3 val 256/128: GAUC val-A={small['best_gauc']:.6f}, лучшая эпоха {small['chosen_epochs']}; обучение завершено.")
    large_ok=call(model_args('r3_large',273,286,{'sequence':512,'dim':256})+['--select'],'r3_large',optional=True)
    candidates=[small]
    if large_ok:
        large=read(BASE/'r3_large/result.json');candidates.append(large)
        append(f"R3 val 512/256: GAUC val-A={large['best_gauc']:.6f}, лучшая эпоха {large['chosen_epochs']}; обучение завершено.")
    chosen=max(candidates,key=lambda x:x['best_gauc'])
    selection={k:chosen[k] for k in ['model','sequence','dim','chosen_epochs','best_gauc']}
    selection.update(selection_days=[287,293],selection_half='A',frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                     candidates=[{k:x[k] for k in ['model','sequence','dim','chosen_epochs','best_gauc']} for x in candidates])
    selected_config=BASE/chosen['model']/'config.yaml'
    shutil.copyfile(selected_config,RUN/'selected_config.yaml')
    selection['selected_config_sha256']=hashlib.sha256(selected_config.read_bytes()).hexdigest()
    write(BASE/'selection.json',selection)
    append(f"**Конфигурация зафиксирована**: history={selection['sequence']}, dim={selection['dim']}, 2 слоя, bf16, "
           f"эпох={selection['chosen_epochs']}, GAUC val-A={selection['best_gauc']:.6f}. Веса score: listen_plus=1.0; allow_unknown_items=true.")
    score(chosen['model'],287,293,'r3_val');costs()
    for model,start,end,score_start,score_end in [('r4_val',259,272,273,293),('r4_oot',266,279,280,300),('r3_oot',280,293,294,300)]:
        append(f'Запущен {model}: метки {start}–{end}, фиксированные конфигурация и эпохи.')
        call(model_args(model,start,end,selection,selection['chosen_epochs']),model)
        result=read(BASE/model/'result.json')
        append(f"{model}: checkpoint готов, обучение+сохранение {result['fit_save_seconds']:.1f} с; "
               f"peak torch reserved={result['peak_reserved_bytes']/2**20:.1f} MiB, checkpoint={result['checkpoint_bytes']/2**20:.1f} MiB.")
        if model!='r3_oot' or (ROOT/'artifacts/freeze.json').is_file():score(model,score_start,score_end,model)
        else:
            append('R3 OOT **не проскорен**: `artifacts/freeze.json` отсутствует. Checkpoint: `external/perseus_stage2/r3_oot/checkpoint/`.')
        costs()
    append('Команда разрешённого R3 OOT скоринга (внутри повторно проверяется freeze.json):\n'
           '```powershell\n. external/probe_env.ps1\n& $ProbePython external/perseus_adapter/stage2.py score --model r3_oot --start 294 --end 300 --kind r3_oot\n```')
    append('Пункты 1–4 завершены с условием freeze для R3 OOT. Статус дополнительного H6 будет указан отдельно.')
    write(BASE/'complete.json',{'success':True,'selection':selection,'r3_oot_scored':(RUN/'pred_oot.parquet').is_file()})

if __name__=='__main__':main()
