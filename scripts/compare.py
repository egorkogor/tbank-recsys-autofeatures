"""Paired bootstrap of branches against a base on one stage; writes a markdown table.

  python scripts/compare.py --stage val --part val_B --base b0 b0 r1_decay r1_window ...
  python scripts/compare.py --stage oot --base b0 ...
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.config import ART  # noqa: E402
from tbrec.metrics import paired_bootstrap, summary, verdict  # noqa: E402

RUNS = ART / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("branches", nargs="+")
    ap.add_argument("--stage", default="val")
    ap.add_argument("--part", default="eval", choices=["eval", "val_A", "val_B"])
    ap.add_argument("--base", default="b0")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "day", "y", "val_half")
    if a.part != "eval":
        rows = rows.filter(pl.col("val_half") == a.part[-1])
    df = rows
    names = list(dict.fromkeys([a.base] + a.branches))
    for b in names:
        p = pl.read_parquet(RUNS / b / f"pred_{a.stage}.parquet").rename({"score": b})
        # a model may return no score (e.g. Perseus with an empty history): a constant, i.e. no ranking signal
        p = p.with_columns(pl.col(b).cast(pl.Float64).fill_nan(None).fill_null(pl.col(b).mean()))
        df = df.join(p, on="row", how="inner")
    per_day = {b: [round(summary(df.filter(pl.col("day") == d), b)["gauc"], 4) for d in sorted(df["day"].unique())] for b in names}
    res = paired_bootstrap(df, names, a.base, n=a.n)
    lines = [f"Stage {a.stage}, part {a.part}, rows {df.height}, users {df['uid'].n_unique()}, base {a.base}, bootstrap {a.n} over users", "",
             "| branch | GAUC | Δ GAUC [95% CI] | verdict | NDCG@10 | Δ NDCG@10 [95% CI] | verdict | ROC-AUC | days Δ>0 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for b in names:
        r = res[b]
        roc = summary(df, b)["roc_auc"]
        if b == a.base:
            lines.append(f"| {b} | {r['gauc']:.4f} | — | — | {r['ndcg10']:.4f} | — | — | {roc:.4f} | — |")
            continue
        days_up = sum(x > y for x, y in zip(per_day[b], per_day[a.base]))
        lines.append(f"| {b} | {r['gauc']:.4f} | {r['d_gauc']:+.4f} [{r['d_gauc_ci'][0]:+.4f}, {r['d_gauc_ci'][1]:+.4f}] | {verdict(r['d_gauc'], r['d_gauc_ci'], 'gauc')} "
                     f"| {r['ndcg10']:.4f} | {r['d_ndcg10']:+.4f} [{r['d_ndcg10_ci'][0]:+.4f}, {r['d_ndcg10_ci'][1]:+.4f}] | {verdict(r['d_ndcg10'], r['d_ndcg10_ci'], 'ndcg10')} "
                     f"| {roc:.4f} | {days_up}/{len(per_day[b])} |")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out) if a.out else ART / f"compare_{a.stage}_{a.part}.md"
    out.write_text(text + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({"bootstrap": res, "per_day_gauc": per_day}, indent=1, default=float))


if __name__ == "__main__":
    main()
