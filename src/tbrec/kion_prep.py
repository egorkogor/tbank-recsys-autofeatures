"""KION -> our contract: events, user / item attributes, candidates and labels per target week.

Sample: 30k users (fixed hash) with their whole history; popularity is computed on the sample, as in
Yambda-50M. Weeks: c5 = last 7 days (OOT), c4 = val, c1..c3 = training weeks (OOT stage: c2..c4).
Candidates for a user and week c: top-50 popular titles of the 7 days before c plus top-50 iALS
(trained on events before c), minus titles the user already watched before c. Label: viewed in [c, c+7).
"""
import hashlib
import json
from datetime import date

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import THREADS
from .config import DAY, ROOT

SRC = ROOT / "data" / "raw" / "kion" / "extracted" / "data_en"
PROC = ROOT / "data" / "processed" / "kion"
ART = ROOT / "artifacts" / "kion"
N_USERS, TOP, START = 30_000, 50, date(2021, 3, 13)


def _h(s: str) -> float:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") / 2**64


def _codes(s: pl.Series) -> tuple[pl.Series, dict]:
    vals = sorted(v for v in s.drop_nulls().unique().to_list() if str(v).strip() != "")
    m = {v: i + 1 for i, v in enumerate(vals)}  # 0 = unknown
    return s.replace_strict(m, default=0, return_dtype=pl.Int32), {str(k): v for k, v in m.items()}


def prepare(log=print) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    it = pl.read_csv(SRC / "interactions.csv", try_parse_dates=True)
    users_all = it["user_id"].unique().to_list()
    keep = sorted(users_all, key=lambda u: _h(f"kion-20261024|{u}"))[:N_USERS]
    it = it.filter(pl.col("user_id").is_in(keep))
    umap = pl.DataFrame({"user_id": sorted(keep)}).with_row_index("uid").with_columns(pl.col("uid").cast(pl.UInt32))
    imap = it.select("item_id").unique().sort("item_id").with_row_index("item").with_columns(pl.col("item").cast(pl.UInt32))
    ev = (it.join(umap, on="user_id").join(imap, on="item_id")
          .with_columns(((pl.col("last_watch_dt").cast(pl.Date) - pl.lit(START)).dt.total_days()).cast(pl.Int16).alias("day"))
          .with_columns((pl.col("day").cast(pl.Int32) * DAY).alias("t"), pl.lit(0, pl.UInt8).alias("kind"),
                        pl.col("watched_pct").fill_null(0).round(0).cast(pl.Int16).alias("pr"),
                        pl.col("total_dur").cast(pl.Int32).alias("dur"), pl.lit(0, pl.UInt8).alias("org"))
          .select("uid", "item", "t", "day", "kind", "org", "pr", "dur").sort("t", "uid", "item"))
    ev.write_parquet(PROC / "events.parquet")
    log(f"events {ev.height}, users {umap.height}, items {imap.height}, days {ev['day'].min()}..{ev['day'].max()}")

    dicts = {}
    us = pl.read_csv(SRC / "users_en.csv").drop("", strict=False).join(umap, on="user_id", how="right")
    cols = {}
    for src, dst in [("age", "age"), ("income", "income"), ("sex", "sex"), ("kids_flg", "kids")]:
        cols[dst], dicts[dst] = _codes(us[src].cast(pl.Utf8))
    pl.DataFrame({"uid": us["uid"], **cols}).write_parquet(PROC / "users.parquet")

    items = pl.read_csv(SRC / "items_en.csv", infer_schema_length=0).drop("", strict=False)
    items = items.with_columns(pl.col("item_id").cast(pl.Int64)).join(imap, on="item_id", how="inner")
    cols = {}
    decade = (items["release_year"].cast(pl.Float64, strict=False) // 10 * 10).cast(pl.Int32, strict=False).cast(pl.Utf8)
    for name, s in [("content_type", items["content_type"]), ("release_decade", decade),
                    ("age_rating", items["age_rating"]), ("for_kids", items["for_kids"])]:
        cols[name], dicts[name] = _codes(s.cast(pl.Utf8))
    year = items["release_year"].cast(pl.Float64, strict=False).cast(pl.Float32).alias("release_year")
    pl.DataFrame({"item": items["item"], **cols}).with_columns(year).write_parquet(PROC / "items.parquet")
    for attr, src in [("genre", "genres"), ("country", "countries"), ("director", "directors_translated"),
                      ("actor", "actors_translated"), ("studio", "studios")]:
        x = (items.select("item", pl.col(src).fill_null("").str.split(",").alias("v")).explode("v")
             .with_columns(pl.col("v").str.strip_chars().str.to_lowercase()).filter(pl.col("v") != ""))
        if attr == "actor":
            freq = x.group_by("v").len().filter(pl.col("len") >= 5).select("v")
            x = x.join(freq, on="v", how="semi")
        x = x.unique(["item", "v"])
        codes, dicts[attr] = _codes(x["v"])
        pl.DataFrame({"item": x["item"], attr: codes}).write_parquet(PROC / f"item_{attr}.parquet")
        log(f"  {attr}: {x['v'].n_unique()} values, items covered {x['item'].n_unique() / imap.height:.1%}")
    (PROC / "codes.json").write_text(json.dumps(dicts, ensure_ascii=False), encoding="utf-8")

    d_end = int(ev["day"].max())
    c5 = d_end - 6
    c4 = c5 - 7
    cuts = [c4 - 21, c4 - 14, c4 - 7, c4, c5]
    stages = {"val": {"train": cuts[:3], "eval": [c4], "first_stage_before": cuts[0]},
              "oot": {"train": cuts[1:4], "eval": [c5], "first_stage_before": cuts[1]}, "day_end": d_end}
    (ART / "stages.json").write_text(json.dumps(stages))
    parts, report = [], {}
    for c in cuts:
        parts.append(_candidates(ev, c, imap.height, log))
        if c != c5:  # OOT week: no metrics before the freeze
            r = parts[-1]
            tgt = ev.filter((pl.col("day") >= c) & (pl.col("day") < c + 7)).join(r.select("uid").unique(), on="uid", how="semi")
            tgt = tgt.join(ev.filter(pl.col("day") < c).select("uid", "item"), on=["uid", "item"], how="anti").unique(["uid", "item"])
            report[c] = {"rows": r.height, "users": r["uid"].n_unique(), "pos_rate": float(r["y"].mean()),
                         "recall_new_targets": float(r["y"].sum() / max(tgt.height, 1))}
            log(f"  cutoff {c}: {report[c]}")
    rows = pl.concat(parts).sort("day", "uid", "item").with_row_index("row").with_columns(pl.col("row").cast(pl.UInt32))
    users = rows["uid"].unique().to_list()
    b = pl.DataFrame({"uid": users,
                      "val_half": ["A" if _h(f"half-20261024|{u}") < 0.5 else "B" for u in users],
                      "sel": [_h(f"sel-20261024|{u}") < 0.3 for u in users]}).cast({"uid": pl.UInt32})
    rows = rows.join(b, on="uid", how="left").sort("day", "uid", "item")
    rows.write_parquet(ART / "rows.parquet")
    (ART / "prep_report.json").write_text(json.dumps({"cutoffs": cuts, "train_val_weeks": report}, indent=1))
    log(f"rows {rows.height}")


def _candidates(ev: pl.DataFrame, c: int, n_items: int, log) -> pl.DataFrame:
    from implicit.als import AlternatingLeastSquares

    hist = ev.filter(pl.col("day") < c)
    target = ev.filter((pl.col("day") >= c) & (pl.col("day") < c + 7))
    users = target.select("uid").unique().join(hist.select("uid").unique(), on="uid", how="semi").sort("uid")
    pop = (hist.filter(pl.col("day") >= c - 7).group_by("item").len().sort(["len", "item"], descending=[True, False])
           .head(TOP).with_row_index("cand_pop_rank", offset=1).select("item", pl.col("cand_pop_rank").cast(pl.Float32)))
    hu = hist.group_by("uid", "item").agg((1 + pl.col("pr").max() / 100).cast(pl.Float32).alias("w"))
    n_u = int(ev["uid"].max()) + 1
    m = sp.csr_matrix((hu["w"].to_numpy(), (hu["uid"].to_numpy(), hu["item"].to_numpy())), shape=(n_u, n_items))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, num_threads=THREADS, random_state=0)
    als.fit(m, show_progress=False)
    uids = users["uid"].to_numpy()
    ids, scores = als.recommend(uids, m[uids], N=TOP, filter_already_liked_items=True)
    ials = pl.DataFrame({"uid": np.repeat(uids, TOP).astype(np.uint32), "item": ids.reshape(-1).astype(np.uint32),
                         "cand_ials_rank": np.tile(np.arange(1, TOP + 1, dtype=np.float32), len(uids)),
                         "cand_ials_score": scores.reshape(-1).astype(np.float32)}).filter(pl.col("item") < n_items)
    popc = users.join(pop, how="cross")
    seen = hist.select("uid", "item").unique()
    cand = (popc.join(ials, on=["uid", "item"], how="full", coalesce=True)
            .join(seen, on=["uid", "item"], how="anti"))
    # iALS score for popular-only candidates too (same model), so the feature exists for every row
    miss = cand.filter(pl.col("cand_ials_score").is_null())
    U, V = np.asarray(als.user_factors), np.asarray(als.item_factors)
    s = np.einsum("ij,ij->i", U[miss["uid"].to_numpy()], V[miss["item"].to_numpy()]).astype(np.float32)
    cand = pl.concat([cand.filter(pl.col("cand_ials_score").is_not_null()),
                      miss.with_columns(pl.Series("cand_ials_score", s))], how="vertical_relaxed")
    y = target.select("uid", "item").unique().with_columns(pl.lit(1, pl.Int8).alias("y"))
    cand = cand.join(y, on=["uid", "item"], how="left").with_columns(pl.col("y").fill_null(0),
                                                                    pl.lit(c, pl.Int16).alias("day"),
                                                                    pl.lit(c * DAY, pl.Int32).alias("t"))
    log(f"  cutoff {c}: users {users.height}, candidates {cand.height}")
    return cand.select("uid", "item", "day", "t", "y", "cand_pop_rank", "cand_ials_rank", "cand_ials_score")
