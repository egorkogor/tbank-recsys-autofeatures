"""One event table for everything: listens, likes, dislikes, unlikes, undislikes.

Columns: uid, item, artist, album, t (s), day, kind, org, pr (played ratio %, -1 for non-listens).
kind: 0 listen, 1 like, 2 dislike, 3 unlike, 4 undislike.
"""
import polars as pl

from .config import OFFSET, DAY, PROC, RAW

KINDS = {"listens": 0, "likes": 1, "dislikes": 2, "unlikes": 3, "undislikes": 4}
EVENTS = PROC / "events.parquet"
ITEMS = PROC / "items.parquet"


def _item_maps() -> pl.DataFrame:
    # A track can belong to several artists / albums; keep the smallest id as the primary one.
    art = pl.read_parquet(RAW / "artist_item_mapping.parquet").group_by("item_id").agg(pl.col("artist_id").min())
    alb = pl.read_parquet(RAW / "album_item_mapping.parquet").group_by("item_id").agg(pl.col("album_id").min())
    return art.join(alb, on="item_id", how="full", coalesce=True)


def prepare() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    parts = []
    for name, kind in KINDS.items():
        x = pl.scan_parquet(RAW / "flat" / "50m" / f"{name}.parquet")
        cols = [pl.col("uid"), pl.col("item_id"), pl.col("timestamp").cast(pl.Int32).alias("t"),
                pl.lit(kind, pl.UInt8).alias("kind"), pl.col("is_organic").cast(pl.UInt8).alias("org")]
        cols.append(pl.col("played_ratio_pct").cast(pl.Int16).alias("pr") if kind == 0 else pl.lit(-1, pl.Int16).alias("pr"))
        parts.append(x.select(cols))
    maps = _item_maps().lazy()
    ev = (pl.concat(parts).join(maps, on="item_id", how="left")
          .with_columns(((pl.col("t") + OFFSET) // DAY).cast(pl.Int16).alias("day"),
                        pl.col("artist_id").fill_null(0).cast(pl.UInt32).alias("artist"),
                        pl.col("album_id").fill_null(0).cast(pl.UInt32).alias("album"))
          .rename({"item_id": "item"})
          .select("uid", "item", "artist", "album", "t", "day", "kind", "org", "pr")
          .sort("t", "uid", maintain_order=True)
          .collect())
    ev.write_parquet(EVENTS)
    items = (pl.scan_parquet(RAW / "flat" / "50m" / "listens.parquet")
             .group_by("item_id").agg(pl.col("track_length_seconds").median().cast(pl.Float32).alias("track_len"))
             .rename({"item_id": "item"}).collect())
    items.write_parquet(ITEMS)
    print(ev.shape, ev.group_by("kind").len().sort("kind").to_dicts())
    print("no artist", (ev["artist"] == 0).mean(), "no album", (ev["album"] == 0).mean())


def load_events() -> pl.DataFrame:
    return pl.read_parquet(EVENTS)


def load_items() -> pl.DataFrame:
    return pl.read_parquet(ITEMS)


def impressions(ev: pl.DataFrame, days: list[int]) -> pl.DataFrame:
    """Rows of task A: algorithmic plays (is_organic = 0) on the given days, label = Listen+ (>= 50%)."""
    return (ev.filter((pl.col("kind") == 0) & (pl.col("org") == 0) & pl.col("day").is_in(days))
            .select("uid", "item", "artist", "album", "day", "t",
                    (pl.col("pr") >= 50).cast(pl.Int8).alias("y")))
