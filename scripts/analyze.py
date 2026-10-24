"""Slices and SHAP for the report.

  python scripts/analyze.py slices --stage val --part val_B b0 r1_decay ...
  python scripts/analyze.py shap b0 r1_decay --stage val

Slices (defined by B0 features, i.e. by history before the day): repeat vs new track for the user,
new artist for the user, cold track (< 20 plays in 30 days by the 9k users), user history length
(terciles of all-time plays). GAUC inside a slice = AUC inside the user-day restricted to slice rows.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from tbrec.branches import resolve  # noqa: E402
from tbrec.config import ART, STAGES  # noqa: E402
from tbrec.metrics import summary  # noqa: E402
from tbrec.model import load, matrix  # noqa: E402

RUNS = ART / "runs"
SLICE_COLS = ["UI__listen__cnt__all", "UA__listen__cnt__all", "I__listen__cnt__w30d", "U__listen__cnt__all"]


def slices(a):
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "day", "y", "val_half")
    if a.part != "eval":
        rows = rows.filter(pl.col("val_half") == a.part[-1])
    df = rows.join(pl.read_parquet(ART / "features" / "b0.parquet", columns=["row"] + SLICE_COLS), on="row")
    for b in a.branches:
        df = df.join(pl.read_parquet(RUNS / b / f"pred_{a.stage}.parquet").rename({"score": b}), on="row")
    q1, q2 = df["U__listen__cnt__all"].quantile(1 / 3), df["U__listen__cnt__all"].quantile(2 / 3)
    defs = {
        "все": pl.lit(True),
        "повтор (трек уже слушал)": pl.col("UI__listen__cnt__all") > 0,
        "новый трек": pl.col("UI__listen__cnt__all") == 0,
        "новый артист": pl.col("UA__listen__cnt__all") == 0,
        "холодный трек (< 20 за 30 дн.)": pl.col("I__listen__cnt__w30d") < 20,
        "короткая история (нижняя треть)": pl.col("U__listen__cnt__all") <= q1,
        "средняя история": (pl.col("U__listen__cnt__all") > q1) & (pl.col("U__listen__cnt__all") <= q2),
        "длинная история (верхняя треть)": pl.col("U__listen__cnt__all") > q2,
    }
    lines = [f"Срезы, этап {a.stage}, часть {a.part}. GAUC внутри пользователь-дня на строках среза.", "",
             "| срез | доля строк | Listen+ | " + " | ".join(a.branches) + " |", "|---" * (3 + len(a.branches)) + "|"]
    for name, cond in defs.items():
        x = df.filter(cond)
        g = [summary(x, b)["gauc"] for b in a.branches]
        lines.append(f"| {name} | {x.height / df.height:.1%} | {x['y'].mean():.3f} | " + " | ".join(f"{v:.4f}" for v in g) + " |")
    text = "\n".join(lines)
    print(text)
    (ART / f"slices_{a.stage}_{a.part}.md").write_text(text + "\n", encoding="utf-8")


def shap(a):
    out = {}
    for b in a.branches:
        model = lgb.Booster(model_file=str(RUNS / b / f"model_{a.stage}.txt"))
        df, cols = load(resolve(b, a.stage))
        ev = df.filter(pl.col("day").is_in(STAGES[a.stage]["eval"])).sample(20000, seed=0)
        contrib = model.predict(matrix(ev, cols), pred_contrib=True)[:, :-1]
        imp = sorted(zip(cols, np.abs(contrib).mean(0)), key=lambda x: -x[1])
        out[b] = imp[:25]
        print(b, [(n, round(float(v), 4)) for n, v in imp[:15]])
    (ART / f"shap_{a.stage}.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["slices", "shap"])
    ap.add_argument("branches", nargs="+")
    ap.add_argument("--stage", default="val")
    ap.add_argument("--part", default="val_B", choices=["eval", "val_A", "val_B"])
    a = ap.parse_args()
    slices(a) if a.what == "slices" else shap(a)


if __name__ == "__main__":
    main()
