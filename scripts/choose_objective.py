"""Choose the LightGBM objective once, on B0 (default params, sel users of the training window, early
stopping and decision on val half A). The choice is then fixed for every branch and for the R1 selection
model: artifacts/objective.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.model import OBJECTIVE_FILE, fit, gauc, load, predict, split  # noqa: E402

if __name__ == "__main__":
    df, cols = load({"b0": None})
    p = split(df, "val")
    tr = p["train"].filter(pl.col("sel"))
    res = {}
    for obj in ["binary", "lambdarank", "rank_xendcg"]:
        t = time.time()
        b = fit(tr, cols, {"objective": obj}, va=p["val_A"])
        res[obj] = {"gauc_val_A": gauc(p["val_A"], predict(b, p["val_A"], cols)), "rounds": b.best_iteration,
                    "seconds": time.time() - t}
        print(obj, res[obj], flush=True)
    best = max(res, key=lambda k: res[k]["gauc_val_A"])
    OBJECTIVE_FILE.write_text(json.dumps({"objective": best, "candidates": res}, indent=1))
    print("chosen", best)
