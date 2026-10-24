"""Cost and scalability (Yambda): feature build time vs users, and quality vs training users.

  python scripts/scaling.py build    # one day (290): B0, R1 (96 names), whole decay grammar; 10% / 33% / 100% of users
  python scripts/scaling.py learn    # B0 and R1 trained on 10% / 33% / 100% of the training users, val half B
"""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402
import psutil  # noqa: E402

from tbrec.branches import resolve  # noqa: E402
from tbrec.config import ART, day_start  # noqa: E402
from tbrec.data import load_events  # noqa: E402
from tbrec.features import DECAY_GRID, grammar, snapshot  # noqa: E402
from tbrec.metrics import summary  # noqa: E402
from tbrec.model import fit, load, predict, split  # noqa: E402
from tbrec.specs import B0, BASE  # noqa: E402

FRACS = [0.1, 0.33, 1.0]
DAY_ = 290


def frac_users(uids, f):
    return [u for u in uids if int.from_bytes(hashlib.sha256(f"scale|{u}".encode()).digest()[:8], "big") / 2**64 < f]


def build():
    ev_all = load_events()
    rows_all = pl.read_parquet(ART / "rows.parquet").filter(pl.col("day") == DAY_).drop("val_half", "sel")
    r1 = json.loads((ART / "r1_decay_selected.json").read_text())["names"]
    sets = {"B0": BASE + B0, "R1 (96)": r1, "grammar (1127)": grammar(DECAY_GRID)}
    uids = ev_all["uid"].unique().to_list()
    res = []
    for f in FRACS:
        us = frac_users(uids, f)
        ev = ev_all.filter(pl.col("uid").is_in(us))
        rows = rows_all.filter(pl.col("uid").is_in(us))
        for name, names in sets.items():
            p = psutil.Process()
            rss0, t0 = p.memory_info().rss, time.time()
            snapshot(rows, ev, names, day_start(DAY_))
            res.append({"users": len(us), "events": ev.height, "rows": rows.height, "set": name, "features": len(names),
                        "seconds": time.time() - t0, "rss_gb_after": p.memory_info().rss / 1e9, "rss_gb_before": rss0 / 1e9})
            print(res[-1], flush=True)
    (ART / "scaling_build.json").write_text(json.dumps(res, indent=1))


def learn():
    res = []
    for branch in ["b0", "r1_decay"]:
        params = json.loads((ART / "runs" / branch / "params.json").read_text())
        df, cols = load(resolve(branch, "val"))
        parts = split(df, "val")
        vb = parts["val_B"]
        uids = parts["train"]["uid"].unique().to_list()
        for f in FRACS:
            tr = parts["train"].filter(pl.col("uid").is_in(frac_users(uids, f)))
            t0 = time.time()
            b = fit(tr, cols, params["params"], rounds=params["rounds"])
            s = summary(vb.with_columns(pl.Series("s", predict(b, vb, cols))), "s")
            res.append({"branch": branch, "train_users_frac": f, "train_rows": tr.height, "fit_seconds": time.time() - t0,
                        "gauc_val_B": s["gauc"], "ndcg10_val_B": s["ndcg10"]})
            print(res[-1], flush=True)
    (ART / "scaling_learn.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": build, "learn": learn}[sys.argv[1]]()
