"""Post-freeze exploration, outside the frozen hypotheses: B0 vs R1 on a graded target (Yambda, val stage only).

  python scripts/graded_label.py

The main task labels a play 1 if at least half of the track was played (Listen+). Here the label is a ladder:
0 = skipped (< 50%), 1 = played at least half but not to the end, 2 = played to the end. B0 and R1 (r1_decay)
are retrained with lambdarank on this ladder: the same features, the same frozen hyperparameters (no new tuning),
early stopping on graded NDCG@10 of val half A, report on val half B only. OOT is not touched.

Reported on val-B for the two binary models (frozen val predictions) and the two ladder models:
  - graded NDCG@10 (gain 2^grade - 1) and graded pair accuracy (pairs with different grades) inside a user-day;
  - Listen+ GAUC (the main metric of the task);
  - like GAUC: label = the user liked the track that day (likes are rare and barely tied to playing it through).
Paired bootstrap over users, as everywhere in the project. Writes artifacts/graded_label.json and
artifacts/runs/{b0,r1_decay}_graded/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from tbrec.branches import resolve  # noqa: E402
from tbrec.config import ART, DATASET, PROC, SEED, STAGES  # noqa: E402
from tbrec.data import impressions  # noqa: E402
from tbrec.metrics import _user_sums, paired_bootstrap, summary  # noqa: E402
from tbrec.metrics import per_group as per_group_binary  # noqa: E402
from tbrec.model import BASE_PARAMS, EARLY, MAX_ROUNDS, groups, load, matrix, safe_names, split  # noqa: E402

RUNS = ART / "runs"
BRANCHES = ["b0", "r1_decay"]
K = 10
VAL_DAYS = STAGES["val"]["train"] + STAGES["val"]["eval"]


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def grades() -> pl.DataFrame:
    """row -> grade (0/1/2) and like, rebuilt exactly as the task rows (scripts/build_features.py make_rows)."""
    rows = (pl.scan_parquet(ART / "rows.parquet").filter(pl.col("day").is_in(VAL_DAYS))
            .select("row", "uid", "item", "day", "t", "y").collect())
    days = sorted(rows["day"].unique().to_list())
    ev = pl.scan_parquet(PROC / "events.parquet").filter(pl.col("day").is_in(days)).collect()
    imp = (ev.filter((pl.col("kind") == 0) & (pl.col("org") == 0) & pl.col("day").is_in(days))
           .select("uid", "item", "day", "t", "pr"))
    key = ["day", "uid", "t", "item"]
    # duplicates of one key are matched in label order, so pr >= 50 always agrees with y
    imp = imp.sort(key + ["pr"], descending=[False] * 4 + [True]).with_columns(pl.int_range(pl.len()).over(key).alias("k"))
    r = rows.sort(key + ["y"], descending=[False] * 4 + [True]).with_columns(pl.int_range(pl.len()).over(key).alias("k"))
    r = r.join(imp, on=key + ["k"], how="left")
    assert r["pr"].null_count() == 0 and ((r["pr"] >= 50).cast(pl.Int8) == r["y"]).all()
    likes = ev.filter((pl.col("kind") == 1) & pl.col("day").is_in(days)).select("uid", "item", "day").unique()
    r = r.join(likes.with_columns(pl.lit(1, pl.Int8).alias("like")), on=["uid", "item", "day"], how="left")
    return r.select("row", pl.when(pl.col("pr") < 50).then(0).when(pl.col("pr") < 100).then(1).otherwise(2)
                    .cast(pl.Int8).alias("g"), pl.col("like").fill_null(0))


def train(branch: str, g: pl.DataFrame) -> pl.DataFrame:
    df, cols = load(resolve(branch, "val"))
    df = df.filter(pl.col("day").is_in(VAL_DAYS)).join(g, on="row", how="left")
    parts = split(df, "val")
    tr, va = parts["train"], parts["val_A"]
    frozen = json.loads((RUNS / branch / "params.json").read_text())["params"]
    p = {**BASE_PARAMS, **frozen, "objective": "lambdarank", "metric": "ndcg", "eval_at": [K]}
    names = safe_names(cols)
    dtr = lgb.Dataset(matrix(tr, cols), tr["g"].to_numpy(), group=groups(tr), feature_name=names)
    dva = lgb.Dataset(matrix(va, cols), va["g"].to_numpy(), group=groups(va), feature_name=names, reference=dtr)
    t0 = time.time()
    b = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
    log(f"{branch}: {len(cols)} features, {b.best_iteration} trees, {time.time() - t0:.0f} s")
    ev = parts["eval"]
    pred = ev.select("row", pl.Series("score", b.predict(matrix(ev, cols), num_iteration=b.best_iteration)))
    out = RUNS / f"{branch}_graded"
    out.mkdir(parents=True, exist_ok=True)
    pred.write_parquet(out / "pred_val.parquet")
    (out / "params.json").write_text(json.dumps({"params": frozen, "objective": "lambdarank", "label": "0/1/2 ladder",
                                                 "rounds": b.best_iteration, "seconds": round(time.time() - t0)}, indent=1))
    return pred


def per_group(df: pl.DataFrame, s: str) -> pl.DataFrame:
    """One row per (uid, day): n, graded pair accuracy (auc), graded NDCG@K (ndcg)."""
    gk = ["uid", "day"]
    x = df.select(gk + ["row", "g", pl.col(s).alias("s")])
    parts = []
    for lo, hi in [(0, 1), (0, 2), (1, 2)]:
        y = (x.filter(pl.col("g").is_in([lo, hi])).with_columns((pl.col("g") == hi).alias("pos"))
             .with_columns(pl.col("s").rank("average").over(gk).alias("r")))
        a = (y.group_by(gk).agg(pl.col("pos").sum().alias("npos"), pl.len().alias("m"),
                                pl.col("r").filter(pl.col("pos")).sum().alias("rpos"))
             .filter((pl.col("npos") > 0) & (pl.col("npos") < pl.col("m")))
             .select(gk + [(pl.col("npos") * (pl.col("m") - pl.col("npos"))).alias("w"),
                           (pl.col("rpos") - pl.col("npos") * (pl.col("npos") + 1) / 2).alias("hit")]))
        parts.append(a)
    pairs = pl.concat(parts).group_by(gk).agg(pl.col("w").sum(), pl.col("hit").sum())
    tb = pl.col("row").hash(seed=SEED).alias("tb")  # ties ordered by a fixed hash, as in tbrec.metrics
    z = (x.with_columns(tb).sort(gk + ["s", "tb"], descending=[False, False, True, False])
         .with_columns(pl.int_range(1, pl.len() + 1).over(gk).alias("pos"),
                       (2.0 ** pl.col("g") - 1).alias("gain")))
    dcg = z.filter(pl.col("pos") <= K).group_by(gk).agg((pl.col("gain") / (pl.col("pos") + 1).log(2)).sum().alias("dcg"))
    ideal = (z.sort(gk + ["gain"], descending=[False, False, True]).with_columns(pl.int_range(1, pl.len() + 1).over(gk).alias("ip"))
             .filter(pl.col("ip") <= K).group_by(gk).agg((pl.col("gain") / (pl.col("ip") + 1).log(2)).sum().alias("idcg")))
    n = x.group_by(gk).agg(pl.len().alias("n"))
    return (pairs.join(n, on=gk).join(dcg, on=gk).join(ideal, on=gk)
            .select(gk + ["n", (pl.col("hit") / pl.col("w")).alias("auc"), (pl.col("dcg") / pl.col("idcg")).alias("ndcg")]))


def bootstrap(pgs: dict[str, pl.DataFrame], pairs: list[tuple[str, str]], n: int = 1000) -> dict:
    users = np.sort(pgs[next(iter(pgs))]["uid"].unique().to_numpy())
    sums = {s: _user_sums(pg, users) for s, pg in pgs.items()}
    draws = np.random.default_rng(SEED).multinomial(len(users), np.full(len(users), 1 / len(users)), size=n).astype(np.float64)
    stats = {s: np.stack([(draws @ m)[:, 0] / (draws @ m)[:, 1], (draws @ m)[:, 2] / (draws @ m)[:, 3]], axis=1) for s, m in sums.items()}
    point = {s: [m[:, 0].sum() / m[:, 1].sum(), m[:, 2].sum() / m[:, 3].sum()] for s, m in sums.items()}
    res = {s: {"graded_pair_acc": point[s][0], "graded_ndcg10": point[s][1]} for s in pgs}
    for a, b in pairs:
        d = stats[a] - stats[b]
        res[f"{a} - {b}"] = {m: {"delta": point[a][j] - point[b][j], "ci": np.percentile(d[:, j], [2.5, 97.5]).tolist()}
                             for j, m in enumerate(["graded_pair_acc", "graded_ndcg10"])}
    return res


def main():
    assert DATASET == "yambda", "This exploration is defined for Yambda only"
    g = grades()
    log(f"grades: {g.group_by('g').len().sort('g').to_dicts()}, likes {int(g['like'].sum())}")
    preds = {b: train(b, g) for b in BRANCHES}
    rows = (pl.scan_parquet(ART / "rows.parquet").filter(pl.col("day").is_in(VAL_DAYS))
            .select("row", "uid", "day", "y", "val_half").collect().join(g, on="row"))
    v = rows.filter(pl.col("val_half") == "B")
    names = {}
    for b in BRANCHES:
        for kind, p in [("binary", pl.read_parquet(RUNS / b / "pred_val.parquet")), ("graded", preds[b])]:
            name = f"{b}_{kind}"
            v = v.join(p.rename({"score": name}), on="row", how="inner")
            names[name] = name
    assert v.select([pl.col(c).null_count() for c in names]).sum_horizontal().item() == 0
    pgs = {s: per_group(v, s) for s in names}
    out = {"rows_val_B": v.height,
           "grade_shares_val_B": {str(k): float((v["g"] == k).mean()) for k in (0, 1, 2)},
           "bootstrap": bootstrap(pgs, [("r1_decay_graded", "b0_graded"), ("r1_decay_binary", "b0_binary"),
                                        ("b0_graded", "b0_binary"), ("r1_decay_graded", "r1_decay_binary")]),
           "listen_plus_gauc": {s: summary(v, s)["gauc"] for s in names},
           "like_gauc": {s: {"gauc": r["gauc"], "gauc_ci": r["gauc_ci"]} for s, r in paired_bootstrap(
               v.with_columns(pl.col("like").alias("y")), list(names), "b0_binary").items()},
           "like_groups_val_B": per_group_binary(v.with_columns(pl.col("like").alias("y")), "b0_binary").height,
           "trees": {b: json.loads((RUNS / f"{b}_graded" / "params.json").read_text())["rounds"] for b in BRANCHES}}
    (ART / "graded_label.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
