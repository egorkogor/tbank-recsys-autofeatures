"""Prepare KION in our contract: python scripts/prepare_kion.py (see src/tbrec/kion_prep.py)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TBREC_DATASET", "kion")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401

from tbrec.kion_prep import prepare  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    prepare(log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True))
