"""Ranking metrics inside a user-day and a paired bootstrap over users.

GAUC: AUC inside each (uid, day) group with both labels present, weighted by group size.
NDCG@10: binary relevance inside each (uid, day) group with at least one positive and one negative;
the plain mean over such groups. Both are blind to anything constant within a group (user features).
"""
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from .config import SEED


def per_group(df: pl.DataFrame, score: str, k: int = 10) -> pl.DataFrame:
    """One row per (uid, day): auc, weight, ndcg.

    Ties: AUC uses average ranks; for NDCG positions inside a tie are ordered by a fixed hash of the row id,
    so the result depends neither on the row order of the frame nor on the label (reviewer finding 24.10).
    """
    g = ["uid", "day"]
    tb = (pl.col("row").hash(seed=SEED) if "row" in df.columns else pl.int_range(pl.len()).hash(seed=SEED)).alias("tb")
    x = (df.select(g + ["y", pl.col(score).alias("s"), tb])
         .sort(g + ["s", "tb"], descending=[False, False, True, False])
         .with_columns(pl.col("s").rank("average").over(g).alias("r"),
                       pl.int_range(1, pl.len() + 1).over(g).alias("pos")))
    x = x.with_columns(pl.when(pl.col("pos") <= k).then(pl.col("y") / (pl.col("pos") + 1).log(2)).otherwise(0.0).alias("dcg"))
    agg = x.group_by(g).agg(pl.len().alias("n"), pl.col("y").sum().alias("npos"),
                            pl.col("r").filter(pl.col("y") == 1).sum().alias("rpos"),
                            pl.col("dcg").sum().alias("dcg"))
    agg = agg.filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("n")))
    idcg = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
    ideal = idcg[np.minimum(agg["npos"].to_numpy(), k)]
    agg = agg.with_columns(
        ((pl.col("rpos") - pl.col("npos") * (pl.col("npos") + 1) / 2) / (pl.col("npos") * (pl.col("n") - pl.col("npos")))).alias("auc"),
        (pl.col("dcg") / pl.Series(ideal)).alias("ndcg"))
    return agg.select(g + ["n", "auc", "ndcg"])


def summary(df: pl.DataFrame, score: str) -> dict:
    pg = per_group(df, score)
    y, s = df["y"].to_numpy(), df[score].to_numpy()
    out = {"gauc": float((pg["auc"] * pg["n"]).sum() / pg["n"].sum()), "ndcg10": float(pg["ndcg"].mean()),
           "roc_auc": float(roc_auc_score(y, s)), "groups": pg.height, "rows": df.height}
    if s.min() >= 0 and s.max() <= 1:
        out["logloss"] = float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6)))
    return out


def _user_sums(pg: pl.DataFrame, users: np.ndarray) -> np.ndarray:
    """Per user: [sum n*auc, sum n, sum ndcg, groups], aligned to `users`."""
    u = (pg.group_by("uid").agg((pl.col("n") * pl.col("auc")).sum().alias("a"), pl.col("n").sum().alias("w"),
                                pl.col("ndcg").sum().alias("g"), pl.len().alias("c")))
    u = pl.DataFrame({"uid": users}).join(u, on="uid", how="left").fill_null(0)
    return u.select("a", "w", "g", "c").to_numpy().astype(np.float64)


def paired_bootstrap(df: pl.DataFrame, scores: list[str], base: str, n: int = 1000, seed: int = SEED) -> dict:
    """Users are resampled with all their days; the same draws are used for every score."""
    pgs = {s: per_group(df, s) for s in scores}
    users = np.sort(pgs[base]["uid"].unique().to_numpy())
    sums = {s: _user_sums(pg.filter(pl.col("uid").is_in(users)), users) for s, pg in pgs.items()}
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(users), np.full(len(users), 1 / len(users)), size=n).astype(np.float64)
    res = {}
    stats = {}
    for s, m in sums.items():
        t = draws @ m  # n x 4
        stats[s] = np.stack([t[:, 0] / t[:, 1], t[:, 2] / t[:, 3]], axis=1)
        point = np.array([m[:, 0].sum() / m[:, 1].sum(), m[:, 2].sum() / m[:, 3].sum()])
        res[s] = {"gauc": point[0], "ndcg10": point[1],
                  "gauc_ci": np.percentile(stats[s][:, 0], [2.5, 97.5]).tolist(),
                  "ndcg10_ci": np.percentile(stats[s][:, 1], [2.5, 97.5]).tolist()}
    for s in scores:
        if s == base:
            continue
        d = stats[s] - stats[base]
        res[s]["d_gauc"] = res[s]["gauc"] - res[base]["gauc"]
        res[s]["d_gauc_ci"] = np.percentile(d[:, 0], [2.5, 97.5]).tolist()
        res[s]["d_ndcg10"] = res[s]["ndcg10"] - res[base]["ndcg10"]
        res[s]["d_ndcg10_ci"] = np.percentile(d[:, 1], [2.5, 97.5]).tolist()
        for j, m in enumerate(["gauc", "ndcg10"]):  # two-sided bootstrap p-value of delta != 0
            res[s][f"p_{m}"] = float(min(1.0, 2 * (min((d[:, j] <= 0).sum(), (d[:, j] >= 0).sum()) + 1) / (n + 1)))
    return res


def verdict_holm(delta: float, ci: list[float], p_holm: float, metric: str, alpha: float = 0.05) -> str:
    """Decision word after the Holm correction: "better" / "worse" also need p_holm < alpha."""
    v = verdict(delta, ci, metric)
    if v in ("лучше", "хуже") and p_holm >= alpha:
        return "не определено (после Холма)"
    return v


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (family = the hypotheses fixed in the freeze)."""
    order = sorted(pvals, key=pvals.get)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pvals[k]))
        out[k] = run
    return out


# Decision words, fixed before the OOT run (plan section 6).
MARGIN = {"gauc": 0.002, "ndcg10": 0.003}


def verdict(delta: float, ci: list[float], metric: str) -> str:
    if ci[0] > 0:
        return "лучше"
    if ci[1] < 0:
        return "хуже"
    return "сопоставимо" if abs(delta) < MARGIN[metric] else "не определено"
