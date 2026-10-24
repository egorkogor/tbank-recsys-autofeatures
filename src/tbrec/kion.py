"""KION (MTS online cinema) in the generic grammar.

Rows are (user, candidate) pairs for a target week; candidates exclude titles the user already watched,
so there are no direct user x item links - only indirect ones (genres, actors, directors, countries,
studios, content type) and demographics x item ("how people of this age watch this title").
"""
import polars as pl

from .config import PROC
from .generic import Schema

USER_ATTRS = ["age", "income", "sex", "kids"]
ITEM_ATTRS = ["content_type", "release_decade", "age_rating", "for_kids"]
MULTI_ATTRS = ["genre", "country", "director", "actor", "studio"]

FILTERS = {
    "view": pl.col("kind") == 0,
    "done": pl.col("pr") >= 50,
    "drop": pl.col("pr") < 10,
    "full": pl.col("pr") >= 90,
}


def schema() -> Schema:
    keys = {"U": ["uid"], "I": ["item"],
            "CT": ["content_type"], "UCT": ["uid", "content_type"],
            "DEC": ["release_decade"], "UDEC": ["uid", "release_decade"],
            "AR": ["age_rating"], "UAR": ["uid", "age_rating"],
            "AGE_I": ["age", "item"], "INC_I": ["income", "item"], "SEX_I": ["sex", "item"], "KIDS_I": ["kids", "item"],
            "AGE_CT": ["age", "content_type"]}
    multi = {"G": ("genre", []), "UG": ("genre", ["uid"]), "AGE_G": ("genre", ["age"]),
             "C": ("country", []), "UC": ("country", ["uid"]),
             "D": ("director", []), "UD": ("director", ["uid"]),
             "ACT": ("actor", []), "UACT": ("actor", ["uid"]),
             "ST": ("studio", []), "UST": ("studio", ["uid"])}
    maps = {a: pl.read_parquet(PROC / f"item_{a}.parquet").select("item", a).unique() for a in MULTI_ATTRS}
    return Schema(keys=keys, filters=FILTERS, base="view", intensity="pr", intensity_filters=("view",),
                  unique_keys=("U", "UCT", "UG", "UC", "UD", "UACT", "UST", "CT", "G"), multi=multi, maps=maps)


def enrich(df: pl.DataFrame) -> pl.DataFrame:
    """Attach static user and item attributes (needed by single-valued attribute keys)."""
    users = pl.read_parquet(PROC / "users.parquet").select(["uid"] + USER_ATTRS)
    items = pl.read_parquet(PROC / "items.parquet").select(["item"] + ITEM_ATTRS + ["release_year"])
    df = df.drop([c for c in USER_ATTRS + ITEM_ATTRS + ["release_year", "years_since_release"] if c in df.columns])
    df = (df.join(users, on="uid", how="left").join(items, on="item", how="left")
          .with_columns([pl.col(c).fill_null(0) for c in USER_ATTRS + ITEM_ATTRS]))
    # release date: years between the release year and the row's (or event's) day; START = 2021-03-13
    return df.with_columns((2021.195 + pl.col("day").cast(pl.Float32) / 365.25 - pl.col("release_year")).alias("years_since_release"))


def load_events() -> pl.DataFrame:
    return enrich(pl.read_parquet(PROC / "events.parquet")).sort("t", maintain_order=True)


# Shared by every KION branch: candidate-generator signals and static attributes (not searched).
BASE = ["I__view__age__all", "U__view__age__all"]
STATIC = ["cand_pop_rank", "cand_ials_rank", "cand_ials_score"] + USER_ATTRS + ITEM_ATTRS + ["release_year", "years_since_release"]

# Hand-made KION baseline: what a competitor would write (windows 7 / 30 days and all time).
B0 = [
    # user
    "U__view__cnt__w7d", "U__view__cnt__w30d", "U__view__cnt__all", "U__done__rate__all",
    "U__view__mpr__all", "U__view__rec__all", "U__view__nuniq__w30d",
    # item popularity and quality
    "I__view__cnt__w7d", "I__view__cnt__w14d", "I__view__cnt__w30d", "I__view__cnt__all",
    "I__done__rate__w30d", "I__done__rate__all", "I__view__mpr__w30d", "I__view__rec__all",
    # user x attribute (indirect links)
    "UG__view__cnt__all", "UG__done__rate__all", "UCT__view__cnt__all", "UCT__done__rate__all",
    "UD__view__cnt__all", "UACT__view__cnt__all", "UC__view__cnt__all", "UST__view__cnt__all",
    "UDEC__view__cnt__all", "UAR__view__cnt__all",
    # demographics x item
    "AGE_I__view__cnt__w30d", "INC_I__view__cnt__w30d", "SEX_I__view__cnt__w30d", "KIDS_I__view__cnt__w30d",
    "AGE_I__done__rate__all",
    # attribute popularity
    "G__view__cnt__w7d", "D__view__cnt__all", "ACT__view__cnt__w30d",
]
