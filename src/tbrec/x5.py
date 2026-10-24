"""X5 RetailHero (grocery retail, hashed ids) in the generic grammar.

Rows: (client, product) for a target week; label = bought in [c, c+7). Candidates mix the client's
frequent past products (repeat purchases, like repeats in music) with popular and iALS products (new
to the client, like in cinema). All attributes are hashed ids: the grammar does not need their meaning.
"""
import hashlib
import json
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

from . import THREADS
from .config import DAY, PROC, ROOT
from .generic import Schema

SRC = ROOT / "data" / "raw" / "x5"
ART = ROOT / "artifacts" / "x5"
N_USERS, TOP_HIST, TOP_POP, TOP_IALS = 10_000, 40, 30, 30
START = datetime(2018, 11, 21)
USER_ATTRS = ["age", "gender"]
ITEM_ATTRS = ["level_1", "level_2", "level_3", "level_4", "segment", "brand", "vendor", "own", "alcohol", "netto"]

FILTERS = {"buy": pl.col("kind") == 0, "multi": pl.col("pr") >= 20, "points": pl.col("pts") == 1}


def schema() -> Schema:
    keys = {"U": ["uid"], "I": ["item"], "UI": ["uid", "item"],
            "L2": ["level_2"], "L3": ["level_3"], "L4": ["level_4"], "SEG": ["segment"], "BR": ["brand"],
            "UL2": ["uid", "level_2"], "UL3": ["uid", "level_3"], "UL4": ["uid", "level_4"], "USEG": ["uid", "segment"],
            "UBR": ["uid", "brand"], "UOWN": ["uid", "own"], "UALC": ["uid", "alcohol"],
            "AGE_I": ["age", "item"], "GEN_I": ["gender", "item"], "AGE_L3": ["age", "level_3"]}
    return Schema(keys=keys, filters=FILTERS, base="buy", intensity="pr", intensity_filters=("buy",),
                  unique_keys=("U", "UL3", "UBR", "L3", "BR"))


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    users = pl.read_parquet(PROC / "users.parquet")
    items = pl.read_parquet(PROC / "items.parquet")
    df = df.drop([c for c in USER_ATTRS + ITEM_ATTRS if c in df.columns])
    return (df.join(users, on="uid", how="left").join(items, on="item", how="left")
            .with_columns([pl.col(c).fill_null(0) for c in USER_ATTRS + ITEM_ATTRS]))


def load_events() -> pl.DataFrame:
    return enrich(pl.read_parquet(PROC / "events.parquet")).sort("t", maintain_order=True)


BASE = ["I__buy__age__all", "U__buy__age__all"]
STATIC = ["cand_hist_rank", "cand_pop_rank", "cand_ials_rank", "cand_ials_score", "age", "gender",
          "level_1", "level_2", "own", "alcohol", "netto"]

# Hand-made X5 baseline: frequency, recency, repeat history, category and brand affinity, popularity.
B0 = [
    "U__buy__cnt__w7d", "U__buy__cnt__w30d", "U__buy__cnt__all", "U__buy__nuniq__w30d", "U__buy__rec__all",
    "U__points__rate__all",
    "I__buy__cnt__w7d", "I__buy__cnt__w30d", "I__buy__cnt__all", "I__multi__rate__all", "I__buy__rec__all",
    "UI__buy__cnt__w7d", "UI__buy__cnt__w30d", "UI__buy__cnt__all", "UI__buy__rec__all", "UI__buy__mpr__all",
    "UL3__buy__cnt__w30d", "UL3__buy__cnt__all", "UL4__buy__cnt__all", "UBR__buy__cnt__all", "UOWN__buy__cnt__all",
    "L3__buy__cnt__w7d", "BR__buy__cnt__w30d",
    "AGE_I__buy__cnt__w30d", "GEN_I__buy__cnt__w30d",
]


def _h(s: str) -> float:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big") / 2**64


def _codes(s: pl.Series) -> pl.Series:
    vals = sorted(v for v in s.drop_nulls().unique().to_list() if str(v).strip() != "")
    return s.replace_strict({v: i + 1 for i, v in enumerate(vals)}, default=0, return_dtype=pl.Int32)


def prepare(log=print) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    pu = pl.scan_parquet(SRC / "purchases.parquet")
    clients = pu.select("client_id").unique().collect()["client_id"].to_list()
    keep = sorted(clients, key=lambda c: _h(f"x5-20261024|{c}"))[:N_USERS]
    pu = pu.filter(pl.col("client_id").is_in(keep)).collect()
    umap = pl.DataFrame({"client_id": sorted(keep)}).with_row_index("uid").with_columns(pl.col("uid").cast(pl.UInt32))
    imap = pu.select("product_id").unique().sort("product_id").with_row_index("item").with_columns(pl.col("item").cast(pl.UInt32))
    ts = pl.col("transaction_datetime").str.to_datetime("%Y-%m-%d %H:%M:%S")
    secs = (ts - pl.lit(START)).dt.total_seconds()
    pts = ((pl.col("regular_points_spent").cast(pl.Float64) < 0) | (pl.col("express_points_spent").cast(pl.Float64) < 0))
    ev = (pu.join(umap, on="client_id").join(imap, on="product_id")
          .with_columns(secs.cast(pl.Int32).alias("t"), (secs // DAY).cast(pl.Int16).alias("day"),
                        pl.lit(0, pl.UInt8).alias("kind"), pl.lit(0, pl.UInt8).alias("org"),
                        (pl.col("product_quantity").cast(pl.Float64) * 10).round(0).clip(0, 1000).cast(pl.Int16).alias("pr"),
                        pl.col("trn_sum_from_iss").cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int32).alias("dur"),
                        pts.cast(pl.UInt8).alias("pts"))
          .select("uid", "item", "t", "day", "kind", "org", "pr", "dur", "pts").sort("t", "uid", "item"))
    ev.write_parquet(PROC / "events.parquet")
    log(f"events {ev.height}, users {umap.height}, items {imap.height}, days {ev['day'].min()}..{ev['day'].max()}")
    cl = pl.read_parquet(SRC / "clients.parquet").join(umap, on="client_id", how="right")
    age = cl["age"].cast(pl.Float64, strict=False)
    age_b = pl.Series(np.where((age >= 14) & (age <= 90), (age.fill_null(0) // 10).to_numpy(), -1)).cast(pl.Utf8)
    pl.DataFrame({"uid": cl["uid"], "age": _codes(age_b.replace("-1", None)), "gender": _codes(cl["gender"])}).write_parquet(PROC / "users.parquet")
    pr = pl.read_parquet(SRC / "products.parquet").join(imap, on="product_id", how="inner")
    netto_b = (pr["netto"].cast(pl.Float64, strict=False).fill_null(-1).log1p() * 4).round(0).cast(pl.Utf8)
    cols = {c: _codes(pr[c]) for c in ["level_1", "level_2", "level_3", "level_4", "brand_id", "vendor_id"]}
    pl.DataFrame({"item": pr["item"], "level_1": cols["level_1"], "level_2": cols["level_2"], "level_3": cols["level_3"],
                  "level_4": cols["level_4"], "segment": _codes(pr["segment_id"]), "brand": cols["brand_id"],
                  "vendor": cols["vendor_id"], "own": _codes(pr["is_own_trademark"]), "alcohol": _codes(pr["is_alcohol"]),
                  "netto": _codes(netto_b)}).write_parquet(PROC / "items.parquet")
    d_end = int(ev["day"].max())
    c5 = d_end - 6
    c4 = c5 - 7
    cuts = [c4 - 21, c4 - 14, c4 - 7, c4, c5]
    stages = {"val": {"train": cuts[:3], "eval": [c4], "first_stage_before": cuts[0]},
              "oot": {"train": cuts[1:4], "eval": [c5], "first_stage_before": cuts[1]}, "day_end": d_end}
    (ART / "stages.json").write_text(json.dumps(stages))
    parts, report = [], {}
    for c in cuts:
        r = _candidates(ev, c, imap.height)
        parts.append(r)
        if c != c5:  # OOT week: no metrics before the freeze
            tgt = ev.filter((pl.col("day") >= c) & (pl.col("day") < c + 7)).join(r.select("uid").unique(), on="uid", how="semi").select("uid", "item").unique()
            report[c] = {"rows": r.height, "users": r["uid"].n_unique(), "pos_rate": float(r["y"].mean()),
                         "recall": float(r["y"].sum() / max(tgt.height, 1))}
        log(f"  cutoff {c}: {r.height} rows, {r['uid'].n_unique()} users" + (f", {report[c]}" if c in report else ""))
    rows = pl.concat(parts).sort("day", "uid", "item").with_row_index("row").with_columns(pl.col("row").cast(pl.UInt32))
    users = rows["uid"].unique().to_list()
    b = pl.DataFrame({"uid": users, "val_half": ["A" if _h(f"half-20261024|{u}") < 0.5 else "B" for u in users],
                      "sel": [_h(f"sel-20261024|{u}") < 0.3 for u in users]}).cast({"uid": pl.UInt32})
    rows.join(b, on="uid", how="left").sort("day", "uid", "item").write_parquet(ART / "rows.parquet")
    (ART / "prep_report.json").write_text(json.dumps({"cutoffs": cuts, "train_val_weeks": report}, indent=1))
    log(f"rows {rows.height}")


def _candidates(ev: pl.DataFrame, c: int, n_items: int) -> pl.DataFrame:
    from implicit.als import AlternatingLeastSquares

    hist = ev.filter(pl.col("day") < c)
    target = ev.filter((pl.col("day") >= c) & (pl.col("day") < c + 7))
    users = target.select("uid").unique().join(hist.select("uid").unique(), on="uid", how="semi").sort("uid")
    own = (hist.join(users, on="uid", how="semi").group_by("uid", "item").agg(pl.len().alias("n"), pl.col("t").max().alias("last"))
           .sort(["uid", "n", "last"], descending=[False, True, True]).with_columns(pl.int_range(1, pl.len() + 1).over("uid").alias("r"))
           .filter(pl.col("r") <= TOP_HIST).select("uid", "item", pl.col("r").cast(pl.Float32).alias("cand_hist_rank")))
    pop = (hist.filter(pl.col("day") >= c - 7).group_by("item").agg(pl.col("uid").n_unique().alias("n"))
           .sort(["n", "item"], descending=[True, False]).head(TOP_POP).with_row_index("cand_pop_rank", offset=1)
           .select("item", pl.col("cand_pop_rank").cast(pl.Float32)))
    hu = hist.group_by("uid", "item").agg(pl.len().cast(pl.Float32).log1p().alias("w"))
    n_u = int(ev["uid"].max()) + 1
    m = sp.csr_matrix((hu["w"].to_numpy() + 1, (hu["uid"].to_numpy(), hu["item"].to_numpy())), shape=(n_u, n_items))
    als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15, num_threads=THREADS, random_state=0)
    als.fit(m, show_progress=False)
    uids = users["uid"].to_numpy()
    ids, scores = als.recommend(uids, m[uids], N=TOP_IALS, filter_already_liked_items=True)
    ials = pl.DataFrame({"uid": np.repeat(uids, TOP_IALS).astype(np.uint32), "item": ids.reshape(-1).astype(np.uint32),
                         "cand_ials_rank": np.tile(np.arange(1, TOP_IALS + 1, dtype=np.float32), len(uids))}).filter(pl.col("item") < n_items)
    cand = (own.join(users.join(pop, how="cross"), on=["uid", "item"], how="full", coalesce=True)
            .join(ials, on=["uid", "item"], how="full", coalesce=True))
    U, V = np.asarray(als.user_factors), np.asarray(als.item_factors)
    cand = cand.with_columns(pl.Series("cand_ials_score", np.einsum("ij,ij->i", U[cand["uid"].to_numpy()], V[cand["item"].to_numpy()]).astype(np.float32)))
    y = target.select("uid", "item").unique().with_columns(pl.lit(1, pl.Int8).alias("y"))
    return (cand.join(y, on=["uid", "item"], how="left")
            .with_columns(pl.col("y").fill_null(0), pl.lit(c, pl.Int16).alias("day"), pl.lit(c * DAY, pl.Int32).alias("t"))
            .select("uid", "item", "day", "t", "y", "cand_hist_rank", "cand_pop_rank", "cand_ials_rank", "cand_ials_score"))
