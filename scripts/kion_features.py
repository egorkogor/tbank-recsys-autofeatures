"""KION feature tables with the schema-driven engine (run with TBREC_DATASET=kion).

  python scripts/kion_features.py b0                      # STATIC + BASE + B0 on all rows
  python scripts/kion_features.py grammar --grid decay    # whole grammar on the selection rows
  python scripts/kion_features.py names --names-file artifacts/kion/r1_decay_selected.json --out r1_decay
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TBREC_DATASET", "kion")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec import generic, kion  # noqa: E402
from tbrec.config import ART, DATASET, STAGES  # noqa: E402

assert DATASET == "kion"
FEAT = ART / "features"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["b0", "grammar", "names"])
    ap.add_argument("--grid", choices=["decay", "window"], default="decay")
    ap.add_argument("--names-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sch = kion.schema()
    ev = kion.load_events()
    rows = kion.enrich(pl.read_parquet(ART / "rows.parquet"))
    if a.what == "b0":
        names, out = kion.BASE + kion.B0, "b0"
    elif a.what == "grammar":
        names = sch.grammar(generic.DECAY_GRID_DAYS if a.grid == "decay" else generic.WINDOW_GRID_DAYS)
        out = f"grammar_{a.grid}"
        # selection rows, as for Yambda: training weeks of the 30% "sel" users + val week half A
        rows = rows.filter((pl.col("day").is_in(STAGES["val"]["train"]) & pl.col("sel"))
                           | (pl.col("day").is_in(STAGES["val"]["eval"]) & (pl.col("val_half") == "A")))
    else:
        names, out = json.loads(Path(a.names_file).read_text())["names"], a.out
    print(f"{out}: {len(names)} features, {rows.height} rows", flush=True)
    keep = ["row"] + (kion.STATIC if a.what == "b0" else [])
    x = generic.build(rows, ev, names, sch, log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True))
    x = x.select(keep + names).sort("row")
    x.write_parquet(FEAT / f"{out}.parquet")
    (FEAT / f"{out}.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "rows": x.height, "features": len(names)}))
    print("done", x.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
