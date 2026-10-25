import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
(ROOT/'logs/experiment').mkdir(parents=True, exist_ok=True)
BASE=ROOT/'external/perseus_stage2'
ENGINE=ROOT/'external/perseus_adapter/stage2.py'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def call(arguments,name):
    with (ROOT/f'logs/experiment/perseus_stage2_{name}.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,str(ENGINE)]+arguments,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(name)
def main():
    if not (BASE/'complete.json').exists():raise RuntimeError('Finish points 1-4 first')
    s=read(BASE/'selection.json')
    call(['train','--model','r3_typed','--start','273','--end','286','--seq',str(s['sequence']),
          '--dim',str(s['dim']),'--epochs',str(s['chosen_epochs']),'--typed'],'h6_train')
    call(['score','--model','r3_typed','--start','287','--end','293','--kind','r3_val','--typed'],'h6_score')
    subprocess.run([sys.executable,str(ROOT/'external/perseus_adapter/check_r3_val.py'),'--typed'],check=True)
    model=read(BASE/'r3_typed/result.json');infer=read(BASE/'r3_typed/r3_val_score.json')
    metrics=read(ROOT/'artifacts/runs/r3_perseus_typed/validation.json')
    result={'model':model,'inference':infer,'val_A':metrics,'typed_history':read(BASE/f'cache_typed_{s["sequence"]}/ready.json'),
            'main_selection_changed':False}
    (ROOT/'artifacts/runs/r3_perseus_typed/cost.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    cost=read(ROOT/'artifacts/runs/r3_perseus/cost.json');cost['h6']=result
    (ROOT/'artifacts/runs/r3_perseus/cost.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2))
    with (ROOT/'logs/experiment/perseus_stage2_report.md').open('a',encoding='utf-8') as report:
        report.write(f"\nH6 готов: val-A GAUC={metrics['gauc']:.6f}, delta к основной ветке={metrics['delivered_minus_selection_gauc']:+.6f}; "
                     f"null={infer['missing_fraction']:.4%}. Файл `artifacts/runs/r3_perseus_typed/pred_val.parquet`. "
                     'Основная конфигурация/модели R3/R4 после этого не менялись.\n')
    (BASE/'complete_typed.json').write_text(json.dumps({'success':True,'gauc':metrics['gauc']}))
    print(json.dumps({'success':True,'gauc':metrics['gauc']},indent=2))

if __name__=='__main__':main()
