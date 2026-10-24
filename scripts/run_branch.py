"""Train and score one branch.

  python scripts/run_branch.py b0 --stage val        # Optuna on val half A, report val A / B
  python scripts/run_branch.py b0 --stage oot        # needs artifacts/freeze.json; runs once

Stage oot reuses the val-stage params and number of trees, retrains on the window shifted by 7 days
and scores the 7 OOT days. It refuses to run twice for a branch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.branches import TRIVIAL, resolve  # noqa: E402
from tbrec.config import ART  # noqa: E402
from tbrec.metrics import summary  # noqa: E402
from tbrec.model import Timer, fit, load, predict, split, tune  # noqa: E402

RUNS = ART / "runs"


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def evaluate(parts: dict, stage: str) -> dict:
    keys = ["val_A", "val_B", "eval"] if stage == "val" else ["eval"]
    return {k: summary(parts[k], "score") for k in keys}


def run_trivial(name: str, stage: str):
    stem, col = TRIVIAL[name]
    sign = -1.0 if col.startswith("-") else 1.0  # "-col": lower is better
    col = col.lstrip("-")
    df, _ = load({stem: [col]})
    fill = float(df[col].max() or 0) + 1 if sign < 0 else -1.0  # missing = worst
    parts = split(df.with_columns((sign * pl.col(col).fill_null(fill)).alias("score")), stage)
    if stage == "val":
        parts["val_A"], parts["val_B"] = (parts["eval"].filter(pl.col("val_half") == h) for h in "AB")
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    parts["eval"].select("row", "score").write_parquet(out / f"pred_{stage}.parquet")
    m = evaluate(parts, stage)
    (out / f"metrics_{stage}.json").write_text(json.dumps(m, indent=1))
    log(f"{name} {stage}: {json.dumps({k: round(v['gauc'], 4) for k, v in m.items()})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("branch")
    ap.add_argument("--stage", choices=["val", "oot"], default="val")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.branch in TRIVIAL:
        return run_trivial(a.branch, a.stage)
    out = RUNS / a.branch
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    if a.stage == "oot":
        freeze = ART / "freeze.json"
        assert freeze.exists(), "OOT is run only after the freeze (artifacts/freeze.json)"
        assert a.force or not (out / "pred_oot.parquet").exists(), "OOT already scored for this branch"
    with timer("load"):
        df, cols = load(resolve(a.branch, a.stage))
        parts = split(df, a.stage)
    log(f"{a.branch} {a.stage}: {len(cols)} features, train {parts['train'].height}, eval {parts['eval'].height}")
    if a.stage == "val":
        # Params: Optuna on the 30% "sel" users of the training window, early stopping on val half A.
        # Trees: refit on the full window with early stopping on val half A; OOT reuses that number.
        with timer("tune"):
            best = tune(parts["train"].filter(pl.col("sel")), parts["val_A"], cols,
                        **({"trials": a.trials} if a.trials else {}), log=log)
        with timer("fit"):
            b = fit(parts["train"], cols, best["params"], va=parts["val_A"])
        best["rounds_subsample"], best["rounds"] = best["rounds"], b.best_iteration
        (out / "params.json").write_text(json.dumps(best, indent=1))
    else:
        best = json.loads((out / "params.json").read_text())
        with timer("fit"):
            b = fit(parts["train"], cols, best["params"], rounds=best["rounds"])
    with timer("predict"):
        parts["eval"] = parts["eval"].with_columns(pl.Series("score", predict(b, parts["eval"], cols)))
    if a.stage == "val":
        parts["val_A"], parts["val_B"] = (parts["eval"].filter(pl.col("val_half") == h) for h in "AB")
    parts["eval"].select("row", "score").write_parquet(out / f"pred_{a.stage}.parquet")
    m = evaluate(parts, a.stage)
    (out / f"metrics_{a.stage}.json").write_text(json.dumps(m, indent=1))
    imp = pl.DataFrame({"feature": cols, "gain": b.feature_importance("gain")}).sort("gain", descending=True)
    imp.write_csv(out / f"importance_{a.stage}.csv")
    b.save_model(str(out / f"model_{a.stage}.txt"))
    timer.t["predict_per_1k_rows"] = timer.t["predict"] / parts["eval"].height * 1000
    timer.dump(out / f"cost_{a.stage}.json")
    log(f"{a.branch} {a.stage}: " + json.dumps({k: {mm: round(v[mm], 4) for mm in ('gauc', 'ndcg10', 'roc_auc')} for k, v in m.items()}))


if __name__ == "__main__":
    main()
