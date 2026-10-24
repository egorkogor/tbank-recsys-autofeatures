"""R1: automatic feature selection from the grammar.

Input: artifacts/features/grammar_<grid>.parquet - every grammar feature on the selection rows
(training days of the 30% "sel" users + val days; only val half A is used for decisions).

1. one LightGBM (lambdarank, groups = user-days) on base + all candidates -> gain importance;
2. top-K curve, K in KS: GAUC on val half A; choose the smallest K within TOL of the best;
3. write artifacts/r1_<grid>_selected.json (names, curve, chosen weightings, cost).

  python scripts/search_r1.py --grid decay
  python scripts/search_r1.py --grid window --k 32     # force K (equal count for H4)
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.branches import BASE_COLS  # noqa: E402
from tbrec.config import ART, STAGES  # noqa: E402
from tbrec.features import parse  # noqa: E402
from tbrec.model import FEAT, ROWS, fit, gauc, predict  # noqa: E402

KS = [8, 16, 24, 32, 48, 64, 96, 128]
TOL = 0.0005
PARAMS = {"learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 200}


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def selection_data(grid: str):
    g = pl.read_parquet(FEAT / f"grammar_{grid}.parquet")
    # lifetimes (age) are in the grammar and in the shared base block: keep one copy, not a candidate
    base = pl.read_parquet(FEAT / "b0.parquet", columns=["row"] + [c for c in BASE_COLS if c not in g.columns])
    rows = pl.read_parquet(ROWS).select("row", "uid", "day", "y", "val_half", "sel")
    df = g.join(rows, on="row", how="inner").join(base, on="row", how="left").sort("day", "uid", maintain_order=True)
    cand = [c for c in g.columns if c != "row" and c not in BASE_COLS]
    del g
    tr = df.filter(pl.col("day").is_in(STAGES["val"]["train"]) & pl.col("sel"))
    va = df.filter(pl.col("day").is_in(STAGES["val"]["eval"]) & (pl.col("val_half") == "A"))
    del df
    return tr, va, cand


def score(tr, va, cols, **extra):
    b = fit(tr, cols, {**PARAMS, **extra}, va=va)
    return gauc(va, predict(b, va, cols)), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["decay", "window"], default="decay")
    ap.add_argument("--k", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    tr, va, cand = selection_data(a.grid)
    log(f"{a.grid}: {len(cand)} candidates, selection train {tr.height}, val_A {va.height}")
    g_base, _ = score(tr, va, BASE_COLS)
    g_all, _ = score(tr, va, BASE_COLS + cand)
    # Importance from a ranking objective (groups = user-days): with the binary objective the gain goes to
    # user-level features that explain differences between users, which within-user-day GAUC cannot see.
    _, b = score(tr, va, BASE_COLS + cand, objective="lambdarank")
    imp = sorted(zip(BASE_COLS + cand, b.feature_importance("gain")), key=lambda x: -x[1])
    ranked = [n for n, _ in imp if n in set(cand)]
    log(f"base only {g_base:.4f}, all {len(cand)} candidates {g_all:.4f}")
    curve = {}
    for k in ([a.k] if a.k else KS):
        curve[k], _ = score(tr, va, BASE_COLS + ranked[:k])
        log(f"  top-{k}: {curve[k]:.4f}")
    best = max(curve.values())
    k = a.k or min(kk for kk, v in curve.items() if v >= best - TOL)
    names = ranked[:k]
    res = {"grid": a.grid, "k": k, "names": names, "curve": curve, "gauc_base_only": g_base, "gauc_all_candidates": g_all,
           "weights": Counter(parse(n)[3] for n in names), "keys": Counter(parse(n)[0] for n in names),
           "aggs": Counter(parse(n)[2] for n in names), "importance_top50": imp[:50],
           "search_seconds": time.time() - t0,
           "grammar_build_seconds": json.loads((FEAT / f"grammar_{a.grid}.cost.json").read_text())["seconds"]}
    suffix = f"_k{a.k}" if a.k else ""
    (ART / f"r1_{a.grid}{suffix}_selected.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"chosen K={k}: weights {dict(res['weights'])}, keys {dict(res['keys'])}")


if __name__ == "__main__":
    main()
