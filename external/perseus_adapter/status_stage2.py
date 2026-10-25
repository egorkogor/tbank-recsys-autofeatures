import json
from pathlib import Path
import time
import psutil
out=[]
for p in psutil.process_iter(['pid','cmdline']):
    args=p.info['cmdline'] or []
    names={'stage2.py','run_stage2.py'}
    if not any(Path(x).name in names for x in args):continue
    c=p.cpu_times()
    out.append({'pid':p.pid,'script':next(Path(x).name for x in args if Path(x).name in names),
                'wall_s':round(time.time()-p.create_time(),1),'cpu_s':round(c.user+c.system,1),
                'rss_mib':round(p.memory_info().rss/2**20,1),'status':p.status(),'threads':p.num_threads()})
print(json.dumps(out,indent=2))
print('RAM available MiB',round(psutil.virtual_memory().available/2**20))
