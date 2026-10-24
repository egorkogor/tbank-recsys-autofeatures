"""Listener state proxies (Yambda): what this person finishes at this time of day, and how it sounds.

There are no emotion labels in the log. The audio embedding encodes timbre / energy, i.e. partly the mood of a
track, so state is approximated by sound x time:
  - taste profile per part of day: cosine of the track to the user's Listen+ (and skip) profile in the
    request's part of day ("energetic in the morning, calm in the evening");
  - yesterday's mood: cosine to the profile of the last 24 h before the cutoff;
  - unnamed "moods": 32 k-means clusters of the embeddings, used as a track attribute in the grammar
    (user x cluster, user x cluster x part of day).
Only history before the cutoff and the request's own time are used; clustering uses content only.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp

from .audio import load as load_emb
from .config import DAY, PROC
from .context import add as add_context
from .features import history

CLUSTERS = PROC / "item_audio_cluster.parquet"
N_CLUSTERS = 32
MOOD_KEYS = {"UACL": ["uid", "acl"], "UACLDP": ["uid", "acl", "dp"], "ACLDP": ["acl", "dp"]}
HAND = ["UACL__lp__rate__all", "UACL__skip__rate__all", "UACLDP__lp__rate__all", "ACLDP__lp__rate__h30d"]
COS = ["mood_cos_lp_dp", "mood_cos_skip_dp", "mood_cos_lp_1d", "mood_shift_1d_vs_30d"]


def build_clusters(seed: int = 0) -> None:
    from sklearn.cluster import MiniBatchKMeans

    idx, E = load_emb()
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=seed, batch_size=8192, n_init=3).fit(E)
    idx.with_columns(pl.Series("acl", km.labels_.astype(np.int16) + 1)).select("item", "acl").write_parquet(CLUSTERS)


def add_cluster(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(pl.read_parquet(CLUSTERS), on="item", how="left").with_columns(pl.col("acl").fill_null(0))


def _profiles(h: pl.DataFrame, group: list[str], w: pl.Expr, idx, E, n_groups_map: pl.DataFrame) -> np.ndarray:
    g = (h.with_columns(w.alias("w")).group_by(group + ["item"]).agg(pl.col("w").sum())
         .join(idx, on="item", how="inner").join(n_groups_map, on=group))
    m = sp.csr_matrix((g["w"].to_numpy(), (g["gi"].to_numpy(), g["ei"].to_numpy())), shape=(n_groups_map.height, E.shape[0]))
    p = np.asarray(m @ E, dtype=np.float32)
    return p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-9)


def snapshot(rows: pl.DataFrame, ev: pl.DataFrame, cutoff: int, idx, E) -> pl.DataFrame:
    """Cosine features; rows need uid, item, t, day (the request's part of day comes from t)."""
    rows = add_context(rows)
    h = add_context(history(ev, cutoff).filter(pl.col("kind") == 0).join(rows.select("uid").unique(), on="uid", how="semi"))
    age = (cutoff - pl.col("t")).cast(pl.Float64)
    lp, sk = (pl.col("pr") >= 50).cast(pl.Float64), (pl.col("pr") < 10).cast(pl.Float64)
    ud = rows.select("uid", "dp").unique().sort("uid", "dp").with_row_index("gi")
    uu = rows.select("uid").unique().sort("uid").with_row_index("gi")
    p_lp_dp = _profiles(h, ["uid", "dp"], lp * 2.0 ** (-age / (30 * DAY)), idx, E, ud)
    p_sk_dp = _profiles(h, ["uid", "dp"], sk * 2.0 ** (-age / (30 * DAY)), idx, E, ud)
    p_lp_1d = _profiles(h.filter(pl.col("t") >= cutoff - DAY), ["uid"], lp, idx, E, uu)
    p_lp_30 = _profiles(h, ["uid"], lp * 2.0 ** (-age / (30 * DAY)), idx, E, uu)
    r = (rows.select("row", "uid", "item", "dp").join(ud.rename({"gi": "gd"}), on=["uid", "dp"])
         .join(uu.rename({"gi": "gu"}), on="uid").join(idx, on="item", how="left"))
    ok = r["ei"].is_not_null().to_numpy()
    ei, gd, gu = r["ei"].fill_null(0).to_numpy(), r["gd"].to_numpy(), r["gu"].to_numpy()
    out = {}
    for name, P, gi in [("mood_cos_lp_dp", p_lp_dp, gd), ("mood_cos_skip_dp", p_sk_dp, gd), ("mood_cos_lp_1d", p_lp_1d, gu)]:
        v = np.einsum("ij,ij->i", P[gi], E[ei]).astype(np.float32)
        v[~ok] = np.nan
        out[name] = v
    shift = (1 - np.einsum("ij,ij->i", p_lp_1d, p_lp_30)).astype(np.float32)  # how far yesterday drifted from the usual taste
    out["mood_shift_1d_vs_30d"] = shift[gu]
    return r.select("row").with_columns([pl.Series(k, v) for k, v in out.items()])
