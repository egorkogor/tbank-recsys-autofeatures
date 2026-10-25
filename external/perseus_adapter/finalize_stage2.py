"""Final freeze-gate check and report rendering. Never computes OOT metrics."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
(ROOT/'logs/experiment').mkdir(parents=True, exist_ok=True)
BASE=ROOT/'external/perseus_stage2';RUN=ROOT/'artifacts/runs/r3_perseus'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    assert read(BASE/'complete.json')['success']
    if (ROOT/'artifacts/freeze.json').is_file() and not (RUN/'pred_oot.parquet').is_file():
        with (ROOT/'logs/experiment/perseus_stage2_r3_oot_score.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,str(ROOT/'external/perseus_adapter/stage2.py'),'score','--model','r3_oot',
                            '--start','294','--end','300','--kind','r3_oot'],stdout=log,stderr=subprocess.STDOUT,check=True)
    if (RUN/'pred_oot.parquet').is_file():
        stat=read(BASE/'r3_oot/r3_oot_score.json')
        assert stat['freeze_gate_present_at_start'] is True
        cost=read(RUN/'cost.json');cost['inference']['r3_oot']=stat
        (RUN/'cost.json').write_text(json.dumps(cost,ensure_ascii=False,indent=2))
        subprocess.run([sys.executable,str(ROOT/'external/perseus_adapter/verify_stage2.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'external/perseus_adapter/write_stage2_report.py')],check=True)
    result={'success':True,'local_time':time.strftime('%Y-%m-%d %H:%M:%S'),
            'r3_oot_scored':(RUN/'pred_oot.parquet').is_file(),'h6_complete':(BASE/'complete_typed.json').is_file()}
    (BASE/'finalized.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
