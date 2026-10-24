"""Verdicts on the hypotheses fixed in artifacts/freeze.json: paired bootstrap over users for each pair,
decision words (metrics.verdict) and Holm-adjusted p-values over the family (GAUC and NDCG@10 separately).

  python scripts/hypotheses.py --stage oot            # after the single OOT run
  python scripts/hypotheses.py --stage val --part val_B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.config import ART  # noqa: E402
from tbrec.metrics import holm, paired_bootstrap, verdict, verdict_holm  # noqa: E402

RUNS = ART / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="oot")
    ap.add_argument("--part", default="eval", choices=["eval", "val_A", "val_B"])
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    hyp = json.loads((ART / "freeze.json").read_text())["hypotheses"]
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "day", "y", "val_half")
    if a.part != "eval":
        rows = rows.filter(pl.col("val_half") == a.part[-1])
    res = {}
    for h, (x, base) in hyp.items():
        if not all((RUNS / b / f"pred_{a.stage}.parquet").exists() for b in (x, base)):
            res[h] = None
            continue
        df = rows
        for b in (base, x):
            p = pl.read_parquet(RUNS / b / f"pred_{a.stage}.parquet").rename({"score": b})
            df = df.join(p.with_columns(pl.col(b).cast(pl.Float64).fill_nan(None).fill_null(pl.col(b).mean())), on="row")
        r = paired_bootstrap(df, [base, x], base, n=a.n)[x]
        res[h] = {"branch": x, "base": base, **{k: r[k] for k in ("d_gauc", "d_gauc_ci", "p_gauc", "d_ndcg10", "d_ndcg10_ci", "p_ndcg10")}}
    done = {h: v for h, v in res.items() if v}
    for m in ("gauc", "ndcg10"):
        # the family is the frozen list: a hypothesis without results counts with p = 1 (it cannot shrink the family)
        adj = holm({h: (res[h][f"p_{m}"] if res[h] else 1.0) for h in res})
        for h in done:
            done[h][f"p_{m}_holm"] = adj[h]
    lines = [f"Hypotheses, stage {a.stage}, part {a.part}; paired bootstrap {a.n} over users; Holm over {len(done)} hypotheses", "",
             "| hypothesis | pair | Δ GAUC [95% CI] | p (Holm) | verdict (Holm) | Δ NDCG@10 [95% CI] | p (Holm) | verdict (Holm) |", "|---|---|---|---|---|---|---|---|"]
    for h, v in res.items():
        if not v:
            lines.append(f"| {h} | — | not run | | | | | |")
            continue
        lines.append(f"| {h} | {v['branch']} vs {v['base']} | {v['d_gauc']:+.4f} [{v['d_gauc_ci'][0]:+.4f}, {v['d_gauc_ci'][1]:+.4f}] | "
                     f"{v['p_gauc_holm']:.3f} | {verdict_holm(v['d_gauc'], v['d_gauc_ci'], v['p_gauc_holm'], 'gauc')} | "
                     f"{v['d_ndcg10']:+.4f} [{v['d_ndcg10_ci'][0]:+.4f}, {v['d_ndcg10_ci'][1]:+.4f}] | {v['p_ndcg10_holm']:.3f} | "
                     f"{verdict_holm(v['d_ndcg10'], v['d_ndcg10_ci'], v['p_ndcg10_holm'], 'ndcg10')} |")
    text = "\n".join(lines)
    print(text)
    (ART / f"hypotheses_{a.stage}_{a.part}.md").write_text(text + "\n", encoding="utf-8")
    (ART / f"hypotheses_{a.stage}_{a.part}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
