"""tbrec: automated feature construction vs hand-made features for ranking (Yambda-50M).

Importing the package caps the process so the machine stays usable: 5 worker threads, pinned to the
last 5 logical CPUs, below-normal priority. Import tbrec before polars. Override with TBREC_CORES
(e.g. "0-15") or disable with TBREC_NO_LIMIT=1.
"""
import os

if not os.environ.get("TBREC_NO_LIMIT"):
    _lo, _hi = map(int, os.environ.get("TBREC_CORES", "11-15").split("-"))
    _n = str(_hi - _lo + 1)
    for _v in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(_v, _n)
    try:
        import psutil

        _p = psutil.Process()
        _p.cpu_affinity(list(range(_lo, _hi + 1)))
        _p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # limits are a courtesy, never a failure
        pass

THREADS = int(os.environ.get("POLARS_MAX_THREADS", "5"))
