"""R2 first-stage model: implicit ALS, scores as booster features.

Item factors are fitted only on events of days < first_stage_before of the stage (before the booster's
first training row), so no training row's label is inside the model. User vectors are folded in daily
from the history before the row's day, item factors fixed.

  python scripts/first_stage_ials.py --stage val   # -> artifacts/features/ials_val.parquet
  python scripts/first_stage_ials.py --stage oot   # -> artifacts/features/ials_oot.parquet
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from tbrec import THREADS  # noqa: E402
from tbrec.config import ART, STAGES, day_start  # noqa: E402
from tbrec.data import load_events  # noqa: E402
from tbrec.features import history  # noqa: E402

FACTORS, ITERS, REG, ALPHA = 64, 15, 0.05, 2.0


def interactions(h: pl.DataFrame) -> pl.DataFrame:
    """Confidence per (uid, item): log1p(Listen+ plays) + 2 * liked - 2 * disliked, floored at 0."""
    x = h.filter(pl.col("kind").is_in([0, 1, 2])).group_by("uid", "item").agg(
        ((pl.col("kind") == 0) & (pl.col("pr") >= 50)).sum().alias("lp"),
        (pl.col("kind") == 1).sum().alias("like"), (pl.col("kind") == 2).sum().alias("dis"))
    return x.with_columns((pl.col("lp").cast(pl.Float32).log1p() + 2 * pl.col("like").clip(upper_bound=1)
                           - 2 * pl.col("dis").clip(upper_bound=1)).clip(lower_bound=0).alias("v")).filter(pl.col("v") > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "oot"], required=True)
    a = ap.parse_args()
    from implicit.als import AlternatingLeastSquares

    st = STAGES[a.stage]
    t0 = time.time()
    ev = load_events()
    cut = day_start(st["first_stage_before"])
    x = interactions(history(ev, cut))
    users = x["uid"].unique().sort()
    items = x["item"].unique().sort()
    umap = dict(zip(users.to_list(), range(len(users))))
    imap = pl.DataFrame({"item": items, "ii": np.arange(len(items), dtype=np.int32)})
    x = x.join(imap, on="item").with_columns(pl.col("uid").replace_strict(umap, return_dtype=pl.Int32).alias("ui"))
    m = sp.csr_matrix((x["v"].to_numpy() * ALPHA, (x["ui"].to_numpy(), x["ii"].to_numpy())), shape=(len(users), len(items)))
    model = AlternatingLeastSquares(factors=FACTORS, iterations=ITERS, regularization=REG, num_threads=THREADS, random_state=0)
    model.fit(m, show_progress=False)
    t_fit = time.time() - t0
    V = np.asarray(model.item_factors)
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "item", "day")
    days = sorted(set(st["train"] + st["eval"]))
    rows = rows.filter(pl.col("day").is_in(days)).join(imap, on="item", how="left")
    parts = []
    for d in days:
        r = rows.filter(pl.col("day") == d)
        h = history(ev, day_start(d)).join(r.select("uid").unique(), on="uid", how="semi")
        xi = interactions(h).join(imap, on="item")
        du = r["uid"].unique().sort()
        dmap = dict(zip(du.to_list(), range(len(du))))
        xi = xi.filter(pl.col("uid").is_in(du.to_list())).with_columns(pl.col("uid").replace_strict(dmap, return_dtype=pl.Int32).alias("ui"))
        um = sp.csr_matrix((xi["v"].to_numpy() * ALPHA, (xi["ui"].to_numpy(), xi["ii"].to_numpy())), shape=(len(du), len(items)))
        U = np.asarray(model.recalculate_user(np.arange(len(du)), um))
        r = r.with_columns(pl.col("uid").replace_strict(dmap, return_dtype=pl.Int32).alias("ui"))
        ok = r["ii"].is_not_null().to_numpy()
        s = np.full(r.height, np.nan, dtype=np.float32)
        ui, ii = r["ui"].to_numpy()[ok], r["ii"].to_numpy()[ok].astype(np.int64)
        s[ok] = np.einsum("ij,ij->i", U[ui], V[ii])
        parts.append(r.select("row").with_columns(pl.Series("ials_score", s)))
        print(f"day {d}: {r.height} rows, scored {ok.mean():.3f}, {time.time() - t0:.0f}s", flush=True)
    out = pl.concat(parts).sort("row")
    out.write_parquet(ART / "features" / f"ials_{a.stage}.parquet")
    (ART / "features" / f"ials_{a.stage}.cost.json").write_text(json.dumps(
        {"fit_seconds": t_fit, "total_seconds": time.time() - t0, "trained_before_day": st["first_stage_before"],
         "users": len(users), "items": len(items), "nnz": int(m.nnz)}))


if __name__ == "__main__":
    main()
