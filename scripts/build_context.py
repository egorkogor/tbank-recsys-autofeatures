"""Request-time context features for Yambda (see src/tbrec/context.py).

  python scripts/build_context.py b0       # hand context + raw request time, all rows -> features/ctx_b0.parquet
  python scripts/build_context.py grammar  # every context-key grammar name on the selection rows -> features/grammar_ctx.parquet
  python scripts/build_context.py search   # context names added to R1 (lambdarank importance, top-K curve on val half A)
  python scripts/build_context.py names    # chosen context names, all rows -> features/ctx_r1.parquet
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec import context, generic  # noqa: E402
from tbrec.branches import BASE_COLS  # noqa: E402
from tbrec.config import ART, STAGES  # noqa: E402
from tbrec.data import load_events  # noqa: E402
from tbrec.model import FEAT, ROWS, fit, gauc, predict  # noqa: E402

KS, TOL = [4, 8, 16, 32, 64], 0.0005
PARAMS = {"learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def rows_all():
    return pl.read_parquet(ROWS).select("row", "uid", "item", "artist", "album", "day", "t", "y", "val_half", "sel")


def build(rows, names, out):
    t0 = time.time()
    ev = context.add(load_events())
    x = generic.build(context.add(rows.drop("val_half", "sel")), ev, names, context.schema(),
                      log=lambda s: log(f"{s} {time.time() - t0:.0f}s"))
    x = x.select(["row"] + names).join(context.raw(rows), on="row").sort("row")
    x.write_parquet(FEAT / f"{out}.parquet")
    (FEAT / f"{out}.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "rows": x.height, "features": len(names)}))
    log(f"done {out} {x.shape}")


def search():
    t0 = time.time()
    r1 = json.loads((ART / "r1_decay_selected.json").read_text())["names"]
    g = pl.read_parquet(FEAT / "grammar_ctx.parquet")
    cand = [c for c in g.columns if c not in ("row",) + tuple(context.RAW)]
    base = pl.read_parquet(FEAT / "grammar_decay.parquet", columns=["row"] + r1)
    b0 = pl.read_parquet(FEAT / "b0.parquet", columns=["row"] + [c for c in BASE_COLS if c not in base.columns])
    df = (g.join(rows_all().select("row", "uid", "day", "y", "val_half", "sel"), on="row").join(base, on="row")
          .join(b0, on="row").sort("day", "uid", maintain_order=True))
    tr = df.filter(pl.col("day").is_in(STAGES["val"]["train"]) & pl.col("sel"))
    va = df.filter(pl.col("day").is_in(STAGES["val"]["eval"]) & (pl.col("val_half") == "A"))
    cur = BASE_COLS + r1 + context.RAW

    def score(cols, **kw):
        b = fit(tr, cols, {**PARAMS, **kw}, va=va)
        return gauc(va, predict(b, va, cols)), b

    g0, _ = score(BASE_COLS + r1)
    g_raw, _ = score(cur)
    _, b = score(cur + cand, objective="lambdarank")
    imp = sorted(zip(cur + cand, b.feature_importance("gain")), key=lambda x: -x[1])
    ranked = [n for n, _ in imp if n in set(cand)]
    curve = {}
    for k in KS:
        curve[k], _ = score(cur + ranked[:k])
        log(f"  +ctx top-{k}: {curve[k]:.4f}")
    best = max(curve.values())
    k = min(kk for kk, v in curve.items() if v >= best - TOL)
    res = {"k": k, "ctx_names": ranked[:k], "names": r1 + ranked[:k], "curve": curve, "gauc_r1": g0, "gauc_r1_raw_ctx": g_raw,
           "importance_top30": imp[:30], "seconds": time.time() - t0}
    (ART / "r1_ctx_selected.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"R1 {g0:.4f}, + raw request time {g_raw:.4f}, + context top-{k} {curve[k]:.4f}: {ranked[:k]}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "b0":
        build(rows_all(), context.B0_CTX, "ctx_b0")
    elif what == "grammar":
        r = rows_all()
        r = r.filter((pl.col("day").is_in(STAGES["val"]["train"]) & pl.col("sel"))
                     | (pl.col("day").is_in(STAGES["val"]["eval"]) & (pl.col("val_half") == "A")))
        r = r.join(pl.read_parquet(FEAT / "grammar_decay.parquet", columns=["row"]), on="row", how="semi")
        build(r, context.grammar(generic.DECAY_GRID_DAYS[:1] + ["h6h"] + generic.DECAY_GRID_DAYS[1:]), "grammar_ctx")
    elif what == "search":
        search()
    else:
        build(rows_all(), json.loads((ART / "r1_ctx_selected.json").read_text())["ctx_names"], "ctx_r1")
