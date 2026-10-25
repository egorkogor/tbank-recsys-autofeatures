"""Two data checks behind README statements (Yambda; days before OOT only, nothing is trained or scored anew).

  python scripts/label_checks.py

1. Label noise: when the feed plays the same track to a user twice in a day, how often do both plays have the
   same Listen+ outcome (task rows, days 273-293; consecutive impressions of one (user, day, track), the later
   strictly later in time). Chance agreement is computed from the outcome shares of the pairs themselves.
2. Session branch vs double logging: GAUC of b0 and b0_sess on val-B, on all rows and without rows whose track
   was already played to the user earlier that day (anywhere earlier / at the user's previous listening moment,
   i.e. the previous occupied 5-second bin, which may be minutes or hours earlier). If the session gain came from
   the same play logged twice, it would vanish without those rows. This tests only that narrow hypothesis:
   misaligned features are the job of the invariance tests (tests/test_generic.py).

Writes artifacts/label_checks.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import polars as pl  # noqa: E402

from tbrec.config import ART, PROC  # noqa: E402
from tbrec.metrics import paired_bootstrap  # noqa: E402

VAL_DAYS = (287, 293)
LAST_DAY_BEFORE_OOT = 293


def repeat_agreement(rows: pl.DataFrame) -> dict:
    r = (rows.filter(pl.col("day") <= LAST_DAY_BEFORE_OOT).sort("uid", "day", "item", "t")
         .with_columns(pl.col("y").shift(1).over("uid", "day", "item").alias("y_prev"),
                       pl.col("t").shift(1).over("uid", "day", "item").alias("t_prev")))
    p = r.filter(pl.col("t_prev").is_not_null() & (pl.col("t_prev") < pl.col("t")))
    a, b = float(p["y_prev"].mean()), float(p["y"].mean())
    return {"pairs": p.height, "same_outcome": float((p["y"] == p["y_prev"]).mean()),
            "chance_same_outcome": a * b + (1 - a) * (1 - b), "listen_plus_share_rows": float(r["y"].mean())}


def session_check(rows: pl.DataFrame) -> dict:
    v = rows.filter(pl.col("day").is_between(*VAL_DAYS) & (pl.col("val_half") == "B"))
    keys = v.select("uid", "day").unique()
    lis = (pl.scan_parquet(PROC / "events.parquet").filter((pl.col("kind") == 0) & pl.col("day").is_between(*VAL_DAYS))
           .select("uid", "day", "t", "item").collect().join(keys, on=["uid", "day"], how="semi").unique())
    first = lis.group_by("uid", "day", "item").agg(pl.col("t").min().alias("t_first"))
    bins = (lis.select("uid", "day", "t").unique().sort("uid", "day", "t")
            .with_columns(pl.col("t").shift(1).over("uid", "day").alias("t_prev")))
    prev_same = (v.join(bins, on=["uid", "day", "t"], how="left")
                 .join(lis.rename({"t": "t_prev"}).with_columns(pl.lit(True).alias("prev_same")),
                       on=["uid", "day", "t_prev", "item"], how="left")
                 .select("row", pl.col("prev_same").fill_null(False)))
    v = (v.join(first, on=["uid", "day", "item"], how="left")
         .with_columns((pl.col("t_first") < pl.col("t")).alias("seen_today"))
         .join(prev_same, on="row", how="left"))
    for b in ("b0", "b0_sess"):
        pred = pl.read_parquet(ART / "runs" / b / "pred_val.parquet")
        assert pred["row"].is_unique().all(), b
        v = v.join(pred.rename({"score": b}), on="row", how="left")
    assert v.height == v["row"].n_unique() and v.select(pl.col("b0", "b0_sess").null_count()).sum_horizontal().item() == 0
    out = {"rows": v.height, "share_seen_today": float(v["seen_today"].mean()),
           "share_prev_bin_same_track": float(v["prev_same"].mean())}
    for name, sub in {"all": v, "without_seen_today": v.filter(~pl.col("seen_today")),
                      "without_prev_bin_same_track": v.filter(~pl.col("prev_same"))}.items():
        res = paired_bootstrap(sub, ["b0", "b0_sess"], "b0")
        out[name] = {"rows": sub.height, "b0_gauc": res["b0"]["gauc"], "sess_gauc": res["b0_sess"]["gauc"],
                     "d_gauc": res["b0_sess"]["d_gauc"], "d_gauc_ci": res["b0_sess"]["d_gauc_ci"]}
    return out


def main():
    rows = pl.read_parquet(ART / "rows.parquet")
    res = {"repeat_agreement": repeat_agreement(rows), "session_vs_double_logging": session_check(rows)}
    (ART / "label_checks.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
