"""LightGBM training, Optuna tuning with an equal budget for every branch, stage datasets."""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import THREADS
from .config import ART, SEED, STAGES
from .metrics import summary

ROWS = ART / "rows.parquet"
FEAT = ART / "features"
BASE_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                   verbose=-1, num_threads=THREADS, seed=SEED, max_bin=255)
OPTUNA_TRIALS = 12  # the same for the final model of every branch (tuned on the 30% "sel" users)
MAX_ROUNDS, EARLY = 3000, 100


def load(sources: dict[str, list[str] | None]) -> tuple[pl.DataFrame, list[str]]:
    """Rows joined with feature files. sources: {file stem: column list or None for all columns}."""
    df = pl.read_parquet(ROWS).select("row", "uid", "day", "y", "val_half", "sel")
    cols: list[str] = []
    for stem, keep in sources.items():
        x = pl.read_parquet(FEAT / f"{stem}.parquet", columns=None if keep is None else ["row"] + keep)
        new = [c for c in x.columns if c != "row" and c not in cols]
        df = df.join(x.select(["row"] + new), on="row", how="left")
        cols += new
    return df, cols


def split(df: pl.DataFrame, stage: str) -> dict[str, pl.DataFrame]:
    s = STAGES[stage]
    tr = df.filter(pl.col("day").is_in(s["train"]))
    ev = df.filter(pl.col("day").is_in(s["eval"]))
    out = {"train": tr, "eval": ev}
    if stage == "val":
        out["val_A"] = ev.filter(pl.col("val_half") == "A")
        out["val_B"] = ev.filter(pl.col("val_half") == "B")
    return out


def matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()


def fit(tr: pl.DataFrame, cols: list[str], params: dict, va: pl.DataFrame | None = None,
        rounds: int | None = None) -> lgb.Booster:
    p = {**BASE_PARAMS, **params}
    dtr = lgb.Dataset(matrix(tr, cols), tr["y"].to_numpy(), feature_name=cols, free_raw_data=True)
    if va is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(matrix(va, cols), va["y"].to_numpy(), reference=dtr)
    return lgb.train(p, dtr, num_boost_round=rounds or MAX_ROUNDS, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(EARLY, verbose=False)])


def predict(b: lgb.Booster, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return b.predict(matrix(df, cols), num_iteration=b.best_iteration or None)


def gauc(df: pl.DataFrame, s: np.ndarray) -> float:
    return summary(df.with_columns(pl.Series("s", s)), "s")["gauc"]


def tune(tr: pl.DataFrame, va: pl.DataFrame, cols: list[str], trials: int = OPTUNA_TRIALS, log=print) -> dict:
    """Optuna on val half A: maximise GAUC; early stopping on the same half. Returns params + rounds."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dtr = lgb.Dataset(matrix(tr, cols), tr["y"].to_numpy(), feature_name=cols, free_raw_data=False)
    Xva, yva = matrix(va, cols), va["y"].to_numpy()
    dva = lgb.Dataset(Xva, yva, reference=dtr, free_raw_data=False)

    def objective(trial):
        p = {**BASE_PARAMS,
             "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
             "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 2000, log=True),
             "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
             "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30, log=True),
             "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True)}
        b = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY, verbose=False)])
        trial.set_user_attr("rounds", b.best_iteration)
        g = gauc(va, b.predict(Xva, num_iteration=b.best_iteration))
        log(f"    trial {trial.number}: gauc {g:.4f} rounds {b.best_iteration}")
        return g

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial({k: BASE_PARAMS[k] for k in ["num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2", "learning_rate"]})
    study.optimize(objective, n_trials=trials)
    return {"params": study.best_params, "rounds": study.best_trial.user_attrs["rounds"],
            "gauc_val_A": study.best_value, "trials": trials}


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.s = time.time()

            def __exit__(self, *a):
                timer.t[name] = timer.t.get(name, 0) + time.time() - self.s
        return _T()

    def dump(self, path):
        path.write_text(json.dumps(self.t, indent=1))
