"""Prepare X5 RetailHero in our contract: python scripts/prepare_x5.py (see src/tbrec/x5.py)."""
import os
import sys
import time
from pathlib import Path

os.environ["TBREC_DATASET"] = "x5"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401

from tbrec.x5 import prepare  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    prepare(log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True))
