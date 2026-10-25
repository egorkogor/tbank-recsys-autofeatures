"""Authors architecture comparison, conditional fixed refits, optional long labels.

No network, affinity or priority mutation; OOT labels never loaded.
Run after the initial r3_authors trial completes.
"""
import hashlib
import json
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
SCRIPTS=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def call(script,args,name):
    path=ROOT/f'logs/experiment/perseus_stage3_{name}.log'
    print(f'START {name}',flush=True)
    with path.open('w',encoding='utf-8') as out:
        result=subprocess.run([sys.executable,str(SCRIPTS/script)]+args,stdout=out,stderr=subprocess.STDOUT,env=os.environ.copy())
    if result.returncode:raise RuntimeError(f'{name}: exit {result.returncode}; {path}')
    print(f'DONE {name}',flush=True)
def score(model,start,end,kind,extra=()):
    call('stage2.py',['score','--model',model,'--start',str(start),'--end',str(end),'--kind',kind]+list(extra),model+'_'+kind)
def train(model,start,end,s,select=False,cache='cache'):
    args=['train','--model',model,'--start',str(start),'--end',str(end),'--seq',str(s['sequence']),
          '--dim',str(s['dim']),'--layers',str(s.get('layers',2)),'--batch',str(s.get('batch',16)),
          '--accumulation',str(s.get('accumulation',1)),'--epochs',str(3 if select else s['chosen_epochs']),'--cache',cache]
    if select:args+=['--select']
    call('stage2.py',args,model)

def report(state):
    s=read(BASE/'selection.json')
    cost=read(RUN/'cost.json')
    for p in BASE.glob('*/result.json'):
        if p.parent.name.startswith(('r3_','r4_')):cost['models'][p.parent.name]=read(p)
    cost['trial_inference']={}
    for model in ['r3_authors','r3_long']:
        for p in (BASE/model).glob('*_score.json'):
            cost['trial_inference'].setdefault(model,{})[p.stem]=read(p)
    cost['selection']=s
    for kind,model in {'r3_val':s['model'],**s.get('models',{'r4_val':'r4_val','r4_oot':'r4_oot','r3_oot':'r3_oot'})}.items():
        p=BASE/model/f'{kind}_score.json'
        if p.exists():cost['inference'][kind]=read(p)
    cost['stage3']=state
    if (BASE/'cache_long/ready.json').exists():cost['long_global_prepare']=read(BASE/'cache_long/ready.json')
    write(RUN/'cost.json',cost)
    lines=['# Perseus — этапы 2–3',f"Обновлено {time.strftime('%Y-%m-%d %H:%M:%S')}. Статус: **{state['status']}**.",
      f"Основной выбор R3/R4: **{s['sequence']} событий, dim {s['dim']}, {s.get('layers',2)} слоя, {s['chosen_epochs']} эпохи; val-A GAUC {s['best_gauc']:.8f}**. "
      +state.get('decision','Сравнение конфигурации авторов выполняется.'),
      'Выбор только по половине A дней 287–293. GAUC: AUC внутри (uid, day), вес — число показов, canonical src/tbrec/metrics.py:summary. Val-B и OOT метрики не вычислялись.',
      '| Конфиг | Окно меток | История / dim / слои | Batch × накопление | GAUC val-A | Лучшая / выполнено эпох | Fit+save, с | VRAM allocated / reserved, MiB |',
      '|---|---|---|---|---:|---:|---:|---:|']
    for model,title in [('r3_small','R3 small'),('r3_large','R3 large'),('r3_typed','H6: лимиты типов и горизонты'),('r3_authors','R3 авторы'),('r3_long','R3 длинное окно')]:
        p=BASE/model/'result.json'
        if not p.exists():continue
        x=read(p);epoch=x.get('chosen_epochs',2)
        g=x.get('best_gauc')
        if model=='r3_typed':
            v=ROOT/'artifacts/runs/r3_perseus_typed/validation.json'
            if v.exists():g=read(v)['gauc']
        if g is None:continue
        a,b=x['train_days']
        lines.append(f"| {title} | {a}–{b} | {x['sequence']} / {x['dim']} / {x.get('layers',2)} | {x.get('batch',16 if x['dim']==128 else 8)} × {x.get('accumulation',1)} | {g:.8f} | {epoch} / {x['completed_epochs']} | {x['fit_save_seconds']:.1f} | {x['peak_allocated_bytes']/2**20:.0f} / {x['peak_reserved_bytes']/2**20:.0f} |")
    lines += ['', 'Архитектура авторов сверена в `external/perseus-exps/configs/baseline_comparison/perseus/yambda-50m/config.yaml` и `external/perseus-exps/configs/extra_features/perseus/yambda-50m-all-w-features/config.yaml`: ModernBERT 4 слоя, dim256, 4 головы, dropout0.1, 512 событий, sum, AdamW lr3e-4. '
      'Низкоуровневый `external/perseus/perseus/core/backbone/history/modern_bert.py:Aggregator.__init__` имеет fallback 1 слой/1 голову; взяты явные параметры опубликованного yambda-эксперимента (4/4). '
      'Исходный batch64 уменьшен до физического 8 с накоплением8 (effective64); поддержка Accelerator.accumulate и фактическое число шагов накопления проверены. Архитектура не уменьшалась. '
      'Исходная retrieval-задача адаптирована в ranking Listen+; данные, cutoff, признаки, словари и PLE такие же, как в остальных R3. В отличие от исходных 30 эпох/patience5 — максимум3, early stopping val-A/patience1 по заданию.',
      'bf16; num_workers=0; унаследованы CPU 11–15 и below-normal. Приоритет, affinity и чужие процессы не менялись. Torch VRAM показывает память нашего аллокатора; на GPU также занято около3ГБ чужими процессами.',
      '| Выход | Модель / метки | Строк / null | Сек / 1000 строк |','|---|---|---:|---:|']
    mapping=s.get('models',{'r4_val':'r4_val','r4_oot':'r4_oot','r3_oot':'r3_oot'})
    for kind,model in {'r3_val':s['model'],**mapping}.items():
        p=BASE/model/f'{kind}_score.json'
        x=read(BASE/model/'result.json') if (BASE/model/'result.json').exists() else None
        if p.exists():
            z=read(p);lines.append(f"| `{z['path'].replace(str(ROOT)+os.sep,'')}` | {model}; {x['train_days']} | {z['rows']} / {z['missing']} ({z['missing_fraction']:.4%}) | {z['inference_seconds_per_1000_rows']:.3f} |")
        else:lines.append(f'| {kind} | {model}; checkpoint '+('готов' if x else 'ожидается')+' | скоринга пока нет | — |')
    lines += ['', 'Для каждой строки история строго t < day*86400−6395. Все listen (включая skips), like/dislike; played_ratio, organic, artist/album, длина. Дневные повторы train: mean(y), при инференсе одинаковый скор каждому row. '
      'Listen+ = pr>=50, до273 метки из алгоритмических listen в events; с273 из rows. Метки294–300 не читались и не выводились из pr. На инференсе rows_no_labels. '
      'Пустые истории возвращаются с null; для GAUC только null заменяется фиксированным0.5. allow_unknown_items=true; словари/PLE fit только на train, item cap250k, artist/album100k. '
      'Score=sigmoid FP32 нативного ranking logit, единственная метка listen_plus с весом1. Priority0, независимый last-N для каждого cutoff; порядок t/исходный индекс, ловушки объединения срезов нет.',
      'Длина выбранной основной истории: `artifacts/perseus_len.txt`. Конфиг: `artifacts/runs/r3_perseus/selected_config.yaml`; выбор и model mapping: `external/perseus_stage2/selection.json`. '
      'Полные затраты (подготовка/модель, fit, чекпойнт, инференс, VRAM), включая прежний технический перезапуск: `artifacts/runs/r3_perseus/cost.json`. '
      'H6 остаётся отдельным экспериментом и не подменяет основной выбор. Длинное окно также сохраняется отдельно, без перезаписи R3/R4.',
      f"R3 OOT checkpoint: `external/perseus_stage2/{mapping['r3_oot']}/checkpoint/`. Freeze сейчас: **{'есть' if (ROOT/'artifacts/freeze.json').is_file() else 'отсутствует; R3 OOT не скорится'}**. Команда с повторной проверкой:",
      '```powershell','. external/probe_env.ps1',f"& $ProbePython external/perseus_adapter/stage2.py score --model {mapping['r3_oot']} --start 294 --end 300 --kind r3_oot",'```',
      'Повтор этапа3 (драйвер читает готовый результат автора и сохраняет контрольные точки отдельно):',
      '```powershell','. external/probe_env.ps1',
      '& $ProbePython external/perseus_adapter/stage2.py train --model r3_authors --start 273 --end 286 --seq 512 --dim 256 --layers 4 --batch 8 --accumulation 8 --epochs 3 --select',
      '& $ProbePython external/perseus_adapter/run_stage3.py','```',
      'Для длинного окна используется `prepare_stage2.py --long` и `stage2.py train --model r3_long --start 231 --end 286 --cache cache_long --select` с выбранными seq/dim/layers/batch/accumulation. '
      'Логи: `logs/experiment/perseus_stage3_*.log`; скрипты — `external/perseus_adapter/`.',state.get('note','')]
    REPORT.write_text('\n\n'.join(lines[:4])+'\n\n'+'\n'.join(lines[4:])+'\n',encoding='utf-8')
    write(BASE/'stage3_state.json',state)

def main():
    initial=read(BASE/'selection.json')
    author=read(BASE/'r3_authors/result.json')
    assert author['status']=='complete' and author['layers']==4 and author['sequence']==512 and author['dim']==256
    state={'status':'авторы обучены; выбор и обновление артефактов','authors':author,
      'authors_command':'stage2.py train --model r3_authors --start 273 --end 286 --seq 512 --dim 256 --layers 4 --batch 8 --accumulation 8 --epochs 3 --select'}
    if not (BASE/'stage3_initial_selection.json').exists():write(BASE/'stage3_initial_selection.json',initial)
    old=read(BASE/'stage3_initial_selection.json')
    wins=author['best_gauc']>old['best_gauc']
    state['decision']=f"Авторы {'лучше' if wins else 'хуже'} прежнего выбора: {author['best_gauc']:.8f} против {old['best_gauc']:.8f}; "+('выбран конфиг авторов.' if wins else 'основные R3/R4 сохранены.')
    if wins:
        s={k:author[k] for k in ['model','sequence','dim','layers','batch','accumulation','chosen_epochs','best_gauc']}
        s.update(selection_days=[287,293],selection_half='A',frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
          candidates=old['candidates']+[s.copy()],models={k:k+'_authors' for k in ['r4_val','r4_oot','r3_oot']})
        config=BASE/author['model']/'config.yaml';shutil.copyfile(config,RUN/'selected_config.yaml')
        s['selected_config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest();write(BASE/'selection.json',s)
    else:
        s=old.copy()
        s.update(layers=2,batch=16 if s['dim']==128 else 8,accumulation=1)
        s['candidates']=old['candidates']+[{k:author[k] for k in ['model','sequence','dim','layers','chosen_epochs','best_gauc']}]
        write(BASE/'selection.json',s)
    (ROOT/'artifacts/perseus_len.txt').write_text(str(s['sequence']),encoding='ascii')
    report(state)
    score('r3_authors',287,293,'r3_val',['--output-run','r3_perseus_authors'])
    call('check_r3_val.py',['--authors'],'authors_readback')
    if wins:
        shutil.copyfile(ROOT/'artifacts/runs/r3_perseus_authors/pred_val.parquet',RUN/'pred_val.parquet')
        for kind,start,end,ss,se in [('r4_val',259,272,273,293),('r4_oot',266,279,280,300),('r3_oot',280,293,294,300)]:
            model=s['models'][kind]
            if not (BASE/model/'result.json').exists():train(model,start,end,s)
            if kind!='r3_oot' or (ROOT/'artifacts/freeze.json').is_file():score(model,ss,se,kind)
            state['status']=f'{kind} обновлён';report(state)
    elif (ROOT/'artifacts/freeze.json').is_file() and not (RUN/'pred_oot.parquet').exists():score('r3_oot',294,300,'r3_oot')
    call('verify_stage2.py',[],'verify_main')
    state['status']='пункты 1–4 этапа3 завершены; длинное окно запущено';report(state)
    if not (BASE/'cache_long/ready.json').exists():call('prepare_stage2.py',['--long'],'prepare_long')
    if not (BASE/'r3_long/result.json').exists():train('r3_long',231,286,s,select=True,cache='cache_long')
    score('r3_long',287,293,'r3_val',['--cache','cache_long','--output-run','r3_perseus_long'])
    call('check_r3_val.py',['--long'],'long_readback')
    if (ROOT/'artifacts/freeze.json').is_file() and not (RUN/'pred_oot.parquet').exists():
        score(s.get('models',{}).get('r3_oot','r3_oot'),294,300,'r3_oot')
        call('verify_stage2.py',[],'verify_after_freeze')
    state['status']='этап3: пункты 1–5 завершены'
    state['long']=read(BASE/'r3_long/result.json')
    state['note']='Длинное окно: отдельный кэш с дня231 и отдельные словари; прежний кэш/выравнивание event_codes не менялись. GAUC проверен чтением сохранённых pred_val.'
    report(state)

if __name__=='__main__':
    if '--wait-authors' in sys.argv:
        deadline=time.monotonic()+2400
        while not (BASE/'r3_authors/result.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Authors result absent after 40 min; inspect authors log')
            time.sleep(5)
    main()
