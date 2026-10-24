"""Aggregate-feature grammar, evaluated as of a cutoff (snapshot at the start of the row's day).

A feature is a name  KEY__FILTER__AGG__WEIGHT, e.g. UA__lp__rate__h7d:
    "share of the user's plays of this artist that reached Listen+, events decayed with a 7-day half-life".

KEY     U user, I item, A artist, L album, UI / UA / UL user x item / artist / album
FILTER  listen, lp (played >= 50%), skip (< 10%), org (organic play), algo (recommended play), like, dislike
AGG     cnt   weighted count
        rate  (cnt + a * p0) / (weighted listens + a): smoothed share, p0 = global share before the cutoff
        mpr   weighted mean played ratio (listen filters only)
        rec   log1p(hours since the last such event)            (WEIGHT = all)
        age   log1p(hours since the first such event)           (WEIGHT = all)
        nuniq distinct items (keys U, A, L; WEIGHT = all or a hard window)
WEIGHT  all | w<n><h|d> hard window | h<n><h|d> exponential decay with half-life n

Every feature of a row on day d uses only events with t < day_start(d): `hist` is a prefix slice of the
time-sorted event table, so nothing from the row's own day (including its label) or the future is visible.
The hand-made baseline B0 and the automatic search R1 use this same engine; they differ only in who
chooses the names.
"""
import re
from functools import lru_cache

import numpy as np
import polars as pl

from .config import DAY, OFFSET, day_start

KEYS = {"U": ["uid"], "I": ["item"], "A": ["artist"], "L": ["album"],
        "UI": ["uid", "item"], "UA": ["uid", "artist"], "UL": ["uid", "album"]}
LISTEN = pl.col("kind") == 0
FILTERS = {
    "listen": LISTEN,
    "lp": LISTEN & (pl.col("pr") >= 50),
    "skip": LISTEN & (pl.col("pr") < 10),
    "org": LISTEN & (pl.col("org") == 1),
    "algo": LISTEN & (pl.col("org") == 0),
    "like": pl.col("kind") == 1,
    "dislike": pl.col("kind") == 2,
}
AGGS = ["cnt", "rate", "mpr", "rec", "age", "nuniq"]
ALPHA = 10.0  # pseudo-listens for smoothed shares
DECAY_GRID = ["h6h", "h1d", "h3d", "h7d", "h14d", "h30d", "h90d", "h365d", "all"]
WINDOW_GRID = ["w6h", "w1d", "w3d", "w7d", "w14d", "w30d", "w90d", "w365d", "all"]


@lru_cache(maxsize=None)
def parse(name: str) -> tuple[str, str, str, str]:
    key, filt, agg, wt = name.split("__")
    assert key in KEYS and filt in FILTERS and agg in AGGS, name
    assert wt == "all" or re.fullmatch(r"[wh]\d+[hd]", wt), name
    return key, filt, agg, wt


def valid(key: str, filt: str, agg: str, wt: str) -> bool:
    if agg in ("rec", "age"):
        return wt == "all"
    if agg == "nuniq":
        return key in ("U", "A", "L") and not wt.startswith("h")
    if agg == "mpr":
        return filt in ("listen", "org", "algo")
    if agg == "rate":
        return filt != "listen"
    return True


def grammar(weights: list[str]) -> list[str]:
    """All valid feature names for a weighting grid (the R1 search space)."""
    return [f"{k}__{f}__{a}__{w}" for k in KEYS for f in FILTERS for a in AGGS for w in weights
            if valid(k, f, a, w)]


def _seconds(wt: str) -> int:
    n, unit = int(wt[1:-1]), wt[-1]
    return n * (3600 if unit == "h" else DAY)


def _weight(wt: str, cutoff: int) -> pl.Expr:
    if wt == "all":
        return pl.lit(1.0, pl.Float32)
    age = (cutoff - pl.col("t")).cast(pl.Float32)
    if wt[0] == "w":
        return (age <= _seconds(wt)).cast(pl.Float32)
    return (2.0 ** (-age / _seconds(wt))).cast(pl.Float32)


def history(ev: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """Prefix of the time-sorted event table: all events strictly before the cutoff."""
    return ev.slice(0, int(ev["t"].search_sorted(cutoff, side="left")))


def _key_block(rows: pl.DataFrame, hist: pl.DataFrame, key: str, names: list[str], cutoff: int,
               p0: dict) -> pl.DataFrame:
    cols = KEYS[key]
    keys = rows.select(cols).unique()
    h = hist.join(keys, on=cols, how="semi")
    specs = [parse(n) for n in names]
    filts = {f for _, f, _, _ in specs} | ({"listen"} if any(a == "rate" for _, _, a, _ in specs) else set())
    wts = {w for _, _, _, w in specs}
    h = h.with_columns([FILTERS[f].cast(pl.Float32).alias(f"F_{f}") for f in filts]
                       + [_weight(w, cutoff).alias(f"W_{w}") for w in wts])
    need: dict[str, pl.Expr] = {}
    for _, f, a, w in specs:
        F, W = pl.col(f"F_{f}"), pl.col(f"W_{w}")
        if a in ("cnt", "rate", "mpr"):
            need[f"s:{f}:{w}"] = (F * W).sum()
        if a == "rate":
            need[f"s:listen:{w}"] = (pl.col("F_listen") * W).sum()
        if a == "mpr":
            need[f"p:{f}:{w}"] = (F * W * pl.col("pr").cast(pl.Float32)).sum()
        if a == "rec":
            need[f"mx:{f}"] = pl.col("t").filter(F > 0).max()
        if a == "age":
            need[f"mn:{f}"] = pl.col("t").filter(F > 0).min()
        if a == "nuniq":
            need[f"nu:{f}:{w}"] = pl.col("item").filter((F > 0) & (W > 0)).n_unique()
    agg = h.group_by(cols).agg([e.alias(k) for k, e in need.items()])
    out = keys.join(agg, on=cols, how="left")
    feats = []
    for n, (_, f, a, w) in zip(names, specs):
        if a == "cnt":
            e = pl.col(f"s:{f}:{w}").fill_null(0)
        elif a == "rate":
            e = (pl.col(f"s:{f}:{w}").fill_null(0) + ALPHA * p0[f]) / (pl.col(f"s:listen:{w}").fill_null(0) + ALPHA)
        elif a == "mpr":
            s = pl.col(f"s:{f}:{w}")
            e = pl.when(s > 0).then(pl.col(f"p:{f}:{w}") / s)
        elif a == "rec":
            e = ((cutoff - pl.col(f"mx:{f}")).cast(pl.Float32) / 3600).log1p()
        elif a == "age":
            e = ((cutoff - pl.col(f"mn:{f}")).cast(pl.Float32) / 3600).log1p()
        else:
            e = pl.col(f"nu:{f}:{w}").fill_null(0)
        feats.append(e.cast(pl.Float32).alias(n))
    return out.select(cols + feats)


def _priors(hist: pl.DataFrame) -> dict:
    s = hist.select([FILTERS[f].sum().alias(f) for f in FILTERS]).row(0, named=True)
    return {f: s[f] / max(s["listen"], 1) for f in FILTERS}


def rhythm(rows: pl.DataFrame, hist: pl.DataFrame, cutoff: int) -> pl.DataFrame:
    """Listening rhythm: active days, regularity of gaps, time-of-day mix."""
    h = (hist.filter(LISTEN & (pl.col("t") >= cutoff - 90 * DAY))
         .join(rows.select("uid").unique(), on="uid", how="semi"))
    days = h.select("uid", "day").unique().sort("uid", "day").with_columns(pl.col("day").diff().over("uid").alias("gap"))
    reg = days.group_by("uid").agg(
        (pl.col("day") >= (cutoff + OFFSET) // DAY - 30).sum().cast(pl.Float32).alias("U_rhythm_active_days_30"),
        (pl.col("gap").std() / pl.col("gap").mean()).cast(pl.Float32).alias("U_rhythm_gap_cv_90"))
    part = (h.filter(pl.col("t") >= cutoff - 30 * DAY)
            .with_columns((((pl.col("t") + OFFSET) % DAY) // (6 * 3600)).alias("dp"))
            .group_by("uid").agg([(pl.col("dp") == i).mean().cast(pl.Float32).alias(f"U_rhythm_daypart_{i}") for i in range(4)]))
    return rows.select("uid").unique().join(reg, on="uid", how="left").join(part, on="uid", how="left")


def snapshot(rows: pl.DataFrame, ev: pl.DataFrame, names: list[str], cutoff: int, with_rhythm: bool = False,
             user_limit: int | None = None) -> pl.DataFrame:
    """Features for rows of one day; ev may be the full log - only the prefix before the cutoff is read.

    user_limit: user-keyed blocks (U, UI, UA, UL, rhythm) see only the user's last N events, the way a
    sequence model with max_events_per_sequence = N does; item / artist / album popularity is unchanged.
    """
    hist = history(ev, cutoff)
    p0 = _priors(hist)
    uhist = hist
    if user_limit:
        uhist = hist.join(rows.select("uid").unique(), on="uid", how="semi").group_by("uid", maintain_order=True).tail(user_limit).sort("t")
    by_key: dict[str, list[str]] = {}
    for n in dict.fromkeys(names):
        by_key.setdefault(parse(n)[0], []).append(n)
    out = rows
    for key, ns in by_key.items():
        h = uhist if key.startswith("U") else hist
        out = out.join(_key_block(rows, h, key, ns, cutoff, p0), on=KEYS[key], how="left")
    if with_rhythm:
        out = out.join(rhythm(rows, uhist, cutoff), on="uid", how="left")
    return out


def build(rows: pl.DataFrame, ev: pl.DataFrame, names: list[str], with_rhythm: bool = False, log=print,
          user_limit: int | None = None) -> pl.DataFrame:
    parts = []
    for d in sorted(rows["day"].unique().to_list()):
        parts.append(snapshot(rows.filter(pl.col("day") == d), ev, names, day_start(d), with_rhythm, user_limit))
        log(f"  day {d}: {parts[-1].height} rows")
    return pl.concat(parts, how="vertical_relaxed")


def feature_columns(df: pl.DataFrame) -> list[str]:
    skip = {"uid", "item", "artist", "album", "day", "t", "y", "row"}
    return [c for c in df.columns if c not in skip]


def to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in cols]).to_numpy()
