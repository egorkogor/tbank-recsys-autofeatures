"""Truncation-invariance check, shared by the tests and the agent loop (R5)."""
import numpy as np
import polars as pl

from .config import day_start
from .data import impressions
from .features import snapshot

DAY_TESTED = 290


def small_log(ev: pl.DataFrame, n_users: int = 150) -> pl.DataFrame:
    users = ev.filter(pl.col("day") == DAY_TESTED)["uid"].unique().sort()[:n_users]
    return ev.filter(pl.col("uid").is_in(users.to_list()))


def scramble_future(ev: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    fut = ev["t"] >= cutoff
    return ev.with_columns(
        pl.when(fut).then(pl.Series(rng.integers(0, 101, ev.height)).cast(pl.Int16)).otherwise(pl.col("pr")).alias("pr"),
        pl.when(fut).then(pl.Series(rng.integers(0, 3, ev.height)).cast(pl.UInt8)).otherwise(pl.col("kind")).alias("kind"))


def invariance_errors(ev: pl.DataFrame, names: list[str], user_limit: int | None = None) -> list[str]:
    """Names whose values change when the log is cut at the cutoff or the future is scrambled."""
    cutoff = day_start(DAY_TESTED)
    rows = impressions(ev, [DAY_TESTED]).with_row_index("row")
    kw = dict(with_rhythm=True, user_limit=user_limit)
    full = snapshot(rows, ev, names, cutoff, **kw).sort("row")
    cut = snapshot(rows, ev.filter(pl.col("t") < cutoff), names, cutoff, **kw).sort("row")
    scr = snapshot(rows, scramble_future(ev, cutoff), names, cutoff, **kw).sort("row")
    bad = []
    for c in full.columns:
        if not (full[c].equals(cut[c]) and full[c].equals(scr[c])):
            bad.append(c)
    return bad
