"""Request-time context for Yambda: hour, part of day and weekday of the impression itself.

The history stays a snapshot at the start of the day; only the time of the request is added, as a live
service knows it when the user opens the feed ("on the way to work", "late evening"). Yambda time is
relative - the log does not start at midnight and weekday 0 is unknown - but the 24-hour and 7-day cycles
are preserved, so "this user skips a lot at this hour" is learnable even without clock names.

Context keys extend the Yambda schema of the generic engine: the same grammar, new GROUP BY columns.
"""
import polars as pl

from .config import DAY, OFFSET
from .generic import Schema, yambda_schema

CTX_KEYS = {"UDP": ["uid", "dp"], "IDP": ["item", "dp"], "ADP": ["artist", "dp"], "UADP": ["uid", "artist", "dp"],
            "UWD": ["uid", "wd"], "IWD": ["item", "wd"], "UHR": ["uid", "hr"]}
RAW = ["ctx_hr", "ctx_dp", "ctx_wd"]  # the request's own time, as features

# Hand-made context features (what an engineer adds first)
B0_CTX = ["UDP__lp__rate__all", "UDP__skip__rate__all", "UDP__listen__cnt__w30d",
          "IDP__lp__rate__w30d", "ADP__lp__rate__all", "UWD__lp__rate__all",
          "UHR__listen__cnt__w30d", "UHR__lp__rate__all"]


def add(df: pl.DataFrame) -> pl.DataFrame:
    hr = (((pl.col("t") + OFFSET) % DAY) // 3600).cast(pl.Int8)
    return df.with_columns(hr.alias("hr"), (hr // 6).cast(pl.Int8).alias("dp"), (pl.col("day") % 7).cast(pl.Int8).alias("wd"))


def schema() -> Schema:
    s = yambda_schema()
    s.keys = {**s.keys, **CTX_KEYS}
    return s


def grammar(weights: list[str]) -> list[str]:
    s = schema()
    return [n for n in s.grammar(weights) if n.split("__")[0] in CTX_KEYS]


def raw(rows: pl.DataFrame) -> pl.DataFrame:
    return add(rows).select("row", pl.col("hr").cast(pl.Float32).alias("ctx_hr"),
                            pl.col("dp").cast(pl.Float32).alias("ctx_dp"), pl.col("wd").cast(pl.Float32).alias("ctx_wd"))
