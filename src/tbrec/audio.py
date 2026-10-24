"""Audio content features: how the track sounds compared to what the user finished or skipped before.

Yambda ships a 128-d normalised audio embedding per track. The user's taste profile on day d is the
(decayed) sum of embeddings of their Listen+ plays before day_start(d); a second profile is built from
skips (< 10%). Features of a row: cosine of the track to each profile and their difference. Same snapshot
rule as the aggregates: only events with t < cutoff are read.
"""
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import scipy.sparse as sp

from .config import DAY, PROC, RAW
from .features import history

EMB_IDS, EMB = PROC / "item_emb_ids.npy", PROC / "item_emb.npy"
HALF_LIFE = 30 * DAY
NAMES = ["audio_cos_lp_h30d", "audio_cos_lp_all", "audio_cos_skip_h30d", "audio_lp_minus_skip_h30d"]


def prepare(items: np.ndarray) -> None:
    """Keep the embeddings of our tracks only (row group by row group, float32)."""
    f = pq.ParquetFile(RAW / "embeddings.parquet")
    want = pl.Series(np.unique(items).astype(np.uint32))
    ids, vecs = [], []
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["item_id", "normalized_embed"])
        x = pl.from_arrow(t).filter(pl.col("item_id").is_in(want.implode()))
        if x.height:
            ids.append(x["item_id"].to_numpy())
            vecs.append(np.asarray(x["normalized_embed"].to_list(), dtype=np.float32))
    np.save(EMB_IDS, np.concatenate(ids))
    np.save(EMB, np.concatenate(vecs))


def load() -> tuple[pl.DataFrame, np.ndarray]:
    ids = np.load(EMB_IDS)
    return pl.DataFrame({"item": ids.astype(np.uint32), "ei": np.arange(len(ids), dtype=np.int32)}), np.load(EMB)


def snapshot(rows: pl.DataFrame, ev: pl.DataFrame, cutoff: int, idx: pl.DataFrame, E: np.ndarray) -> pl.DataFrame:
    users = rows.select("uid").unique().sort("uid")
    umap = users.with_row_index("ui")
    h = (history(ev, cutoff).filter(pl.col("kind") == 0).join(users, on="uid", how="semi")
         .with_columns((2.0 ** (-(cutoff - pl.col("t")).cast(pl.Float64) / HALF_LIFE)).alias("w"),
                       (pl.col("pr") >= 50).cast(pl.Float64).alias("lp"), (pl.col("pr") < 10).cast(pl.Float64).alias("sk"))
         .group_by("uid", "item").agg((pl.col("w") * pl.col("lp")).sum().alias("lp_w"), pl.col("lp").sum().alias("lp_n"),
                                      (pl.col("w") * pl.col("sk")).sum().alias("sk_w"))
         .join(idx, on="item", how="inner").join(umap, on="uid"))
    shape = (users.height, E.shape[0])
    prof = {}
    for col in ["lp_w", "lp_n", "sk_w"]:
        m = sp.csr_matrix((h[col].to_numpy(), (h["ui"].to_numpy(), h["ei"].to_numpy())), shape=shape)
        p = np.asarray(m @ E, dtype=np.float32)
        prof[col] = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)
    r = rows.select("row", "uid", "item").join(umap, on="uid").join(idx, on="item", how="left")
    ok = r["ei"].is_not_null().to_numpy()
    ui, ei = r["ui"].to_numpy(), r["ei"].fill_null(0).to_numpy()
    out = {}
    for name, col in [("audio_cos_lp_h30d", "lp_w"), ("audio_cos_lp_all", "lp_n"), ("audio_cos_skip_h30d", "sk_w")]:
        v = np.einsum("ij,ij->i", prof[col][ui], E[ei]).astype(np.float32)
        v[~ok] = np.nan
        out[name] = v
    out["audio_lp_minus_skip_h30d"] = out["audio_cos_lp_h30d"] - out["audio_cos_skip_h30d"]
    return r.select("row").with_columns([pl.Series(k, v) for k, v in out.items()])
