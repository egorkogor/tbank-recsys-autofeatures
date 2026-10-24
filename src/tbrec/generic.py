"""Schema-driven version of the aggregate grammar: the same KEY__FILTER__AGG__WEIGHT language for any
event log. A dataset is described by a Schema (keys, filters, multi-valued attributes); the grammar,
the search (R1) and the agent loop (R5) then run without code changes.

Single-valued key: columns present in both the event table and the rows (user, item, user x attribute,
attribute x item, ...). Multi-valued key: an item attribute with several values (genres, actors, ...)
given as a mapping table (item, value); events are exploded over the item's values, statistics are
computed per value (optionally per user x value), and a row gets the mean over its item's values.

Same snapshot rule as tbrec.features: only events with t < cutoff are read.
"""
import re
from dataclasses import dataclass, field

import polars as pl

from .config import DAY, day_start
from .features import AGGS, ALPHA, _weight, history

DECAY_GRID_DAYS = ["h1d", "h3d", "h7d", "h14d", "h30d", "h90d", "h180d", "all"]
WINDOW_GRID_DAYS = ["w1d", "w3d", "w7d", "w14d", "w30d", "w90d", "w180d", "all"]


@dataclass
class Schema:
    keys: dict[str, list[str]]
    filters: dict[str, pl.Expr]
    base: str  # denominator of "rate"
    intensity: str = "pr"  # column averaged by "mpr"
    intensity_filters: tuple = ()
    unique_keys: tuple = ()
    multi: dict[str, tuple[str, list[str]]] = field(default_factory=dict)  # name -> (attribute, extra key columns)
    maps: dict[str, pl.DataFrame] = field(default_factory=dict)  # attribute -> DataFrame(item, attribute)

    def parse(self, name: str):
        key, filt, agg, wt = name.split("__")
        assert (key in self.keys or key in self.multi) and filt in self.filters and agg in AGGS, name
        assert wt == "all" or re.fullmatch(r"[wh]\d+[hd]", wt), name
        return key, filt, agg, wt

    def valid(self, key, filt, agg, wt) -> bool:
        if agg in ("rec", "age"):
            return wt == "all"
        if agg == "nuniq":
            return key in self.unique_keys and not wt.startswith("h")
        if agg == "mpr":
            return filt in self.intensity_filters
        if agg == "rate":
            return filt != self.base
        return True

    def grammar(self, weights: list[str]) -> list[str]:
        ks = list(self.keys) + list(self.multi)
        return [f"{k}__{f}__{a}__{w}" for k in ks for f in self.filters for a in AGGS for w in weights
                if self.valid(k, f, a, w)]


def _stats(h: pl.DataFrame, group: list[str], specs, schema: Schema, cutoff: int, p0: dict, names) -> pl.DataFrame:
    """Per-group features (same formulas as tbrec.features) for the events h."""
    filts = {f for _, f, _, _ in specs} | ({schema.base} if any(a == "rate" for _, _, a, _ in specs) else set())
    wts = {w for _, _, _, w in specs}
    h = h.with_columns([schema.filters[f].cast(pl.Float32).alias(f"F_{f}") for f in filts]
                       + [_weight(w, cutoff).alias(f"W_{w}") for w in wts])
    need = {}
    for _, f, a, w in specs:
        F, W = pl.col(f"F_{f}"), pl.col(f"W_{w}")
        if a in ("cnt", "rate", "mpr"):
            need[f"s:{f}:{w}"] = (F * W).sum()
        if a == "rate":
            need[f"s:{schema.base}:{w}"] = (pl.col(f"F_{schema.base}") * W).sum()
        if a == "mpr":
            need[f"p:{f}:{w}"] = (F * W * pl.col(schema.intensity).cast(pl.Float32)).sum()
        if a == "rec":
            need[f"mx:{f}"] = pl.col("t").filter(F > 0).max()
        if a == "age":
            need[f"mn:{f}"] = pl.col("t").filter(F > 0).min()
        if a == "nuniq":
            need[f"nu:{f}:{w}"] = pl.col("item").filter((F > 0) & (W > 0)).n_unique()
    agg = h.group_by(group).agg([e.alias(k) for k, e in need.items()])
    feats = []
    for n, (_, f, a, w) in zip(names, specs):
        if a == "cnt":
            e = pl.col(f"s:{f}:{w}").fill_null(0)
        elif a == "rate":
            e = (pl.col(f"s:{f}:{w}").fill_null(0) + ALPHA * p0[f]) / (pl.col(f"s:{schema.base}:{w}").fill_null(0) + ALPHA)
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
    return agg, feats


def _single(rows, hist, schema, key, names, cutoff, p0):
    cols = schema.keys[key]
    keys = rows.select(cols).unique()
    specs = [schema.parse(n) for n in names]
    agg, feats = _stats(hist.join(keys, on=cols, how="semi"), cols, specs, schema, cutoff, p0, names)
    return keys.join(agg, on=cols, how="left").select(cols + feats), cols


def _multi(rows, hist, schema, key, names, cutoff, p0):
    attr, extra = schema.multi[key]
    m = schema.maps[attr]
    group = extra + [attr]
    rkeys = rows.select(extra + ["item"]).unique()
    gkeys = rkeys.join(m, on="item", how="inner").select(group).unique()
    h = hist.join(m, on="item", how="inner").join(gkeys, on=group, how="semi")
    specs = [schema.parse(n) for n in names]
    agg, feats = _stats(h, group, specs, schema, cutoff, p0, names)
    per_value = gkeys.join(agg, on=group, how="left").select(group + feats)
    out = (rkeys.join(m, on="item", how="inner").join(per_value, on=group, how="left")
           .group_by(extra + ["item"]).agg([pl.col(n).mean() for n in names]))
    return rkeys.join(out, on=extra + ["item"], how="left"), extra + ["item"]


def snapshot(rows: pl.DataFrame, ev: pl.DataFrame, names: list[str], cutoff: int, schema: Schema) -> pl.DataFrame:
    hist = history(ev, cutoff)
    s = hist.select([schema.filters[f].sum().alias(f) for f in schema.filters]).row(0, named=True)
    p0 = {f: s[f] / max(s[schema.base], 1) for f in schema.filters}
    by_key: dict[str, list[str]] = {}
    for n in dict.fromkeys(names):
        by_key.setdefault(schema.parse(n)[0], []).append(n)
    out = rows
    for key, ns in by_key.items():
        block, on = (_multi if key in schema.multi else _single)(rows, hist, schema, key, ns, cutoff, p0)
        out = out.join(block, on=on, how="left")
    return out


def build(rows: pl.DataFrame, ev: pl.DataFrame, names: list[str], schema: Schema, log=print) -> pl.DataFrame:
    parts = []
    for d in sorted(rows["day"].unique().to_list()):
        parts.append(snapshot(rows.filter(pl.col("day") == d), ev, names, day_start(d), schema))
        log(f"  day {d}: {parts[-1].height} rows")
    return pl.concat(parts, how="vertical_relaxed")


def yambda_schema() -> Schema:
    """Yambda in the generic form; must reproduce tbrec.features exactly (tests)."""
    from . import features as F

    return Schema(keys=F.KEYS, filters=F.FILTERS, base="listen", intensity="pr",
                  intensity_filters=("listen", "org", "algo"), unique_keys=("U", "A", "L"))


__all__ = ["Schema", "snapshot", "build", "yambda_schema", "DECAY_GRID_DAYS", "WINDOW_GRID_DAYS", "DAY"]
