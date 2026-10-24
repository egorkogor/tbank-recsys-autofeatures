"""Streaming extension (Yambda): today's session state before the impression. NOT part of the main comparison.

The main setup ranks a day's impressions from a snapshot at the start of the day. A live feed re-ranks after
every track, so the user's actions earlier today are known. This module adds them, strictly:
  - only today's listens of the same user with a timestamp strictly earlier than the impression's; events in
    the same 5-second bin as the impression are excluded (Yambda does not document whether a timestamp is the
    start or the end of a play, and plays are sometimes logged in batches);
  - no time gaps between events: if timestamps were play ends, a gap would reveal how long the current track
    was played, i.e. the label.
Features: outcome of the previous listen(s), counts and shares so far today, whether the previous listen was
organic or of the same artist.
"""
import polars as pl

from .config import DAY, OFFSET

NAMES = ["sess_prev_lp", "sess_prev_skip", "sess_prev_pr", "sess_prev_org", "sess_prev_same_artist",
         "sess_last5_lp", "sess_last5_skip", "sess_n_today", "sess_lp_today"]


def features(rows: pl.DataFrame, ev: pl.DataFrame) -> pl.DataFrame:
    """rows: row, uid, artist, day, t. ev: the event table (only listens of the rows' user-days are read)."""
    day_of = ((pl.col("t") + OFFSET) // DAY).cast(pl.Int16)
    keys = rows.select("uid", "day").unique()
    lis = (ev.filter(pl.col("kind") == 0).with_columns(day_of.alias("day")).join(keys, on=["uid", "day"], how="semi")
           .with_columns((pl.col("pr") >= 50).cast(pl.Float32).alias("lp"), (pl.col("pr") < 10).cast(pl.Float32).alias("sk")))
    # one record per (user, day, 5-second bin): everything in a bin is "the same moment"
    g = (lis.group_by("uid", "day", "t").agg(pl.len().alias("n"), pl.col("lp").sum(), pl.col("sk").sum(),
                                            pl.col("pr").cast(pl.Float32).mean().alias("pr"),
                                            pl.col("org").cast(pl.Float32).mean().alias("org"),
                                            pl.col("artist").first().alias("artist"))
         .sort("uid", "day", "t"))
    by = ["uid", "day"]
    g = g.with_columns(
        # state strictly before this bin
        pl.col("n").cum_sum().shift(1).over(by).fill_null(0).alias("cn"),
        pl.col("lp").cum_sum().shift(1).over(by).fill_null(0).alias("clp"),
        (pl.col("lp") / pl.col("n")).shift(1).over(by).alias("sess_prev_lp"),
        (pl.col("sk") / pl.col("n")).shift(1).over(by).alias("sess_prev_skip"),
        pl.col("pr").shift(1).over(by).alias("sess_prev_pr"),
        pl.col("org").shift(1).over(by).alias("sess_prev_org"),
        pl.col("artist").shift(1).over(by).alias("prev_artist"),
        pl.col("lp").shift(1).rolling_sum(5, min_samples=1).over(by).alias("l5lp"),
        pl.col("sk").shift(1).rolling_sum(5, min_samples=1).over(by).alias("l5sk"),
        pl.col("n").shift(1).rolling_sum(5, min_samples=1).over(by).alias("l5n"),
    ).with_columns(
        (pl.col("l5lp") / pl.col("l5n")).alias("sess_last5_lp"), (pl.col("l5sk") / pl.col("l5n")).alias("sess_last5_skip"),
        pl.col("cn").cast(pl.Float32).alias("sess_n_today"),
        pl.when(pl.col("cn") > 0).then(pl.col("clp") / pl.col("cn")).alias("sess_lp_today"))
    out = rows.select("row", "uid", "day", "t", "artist").join(
        g.select(["uid", "day", "t", "prev_artist"] + [c for c in NAMES if c != "sess_prev_same_artist"]),
        on=["uid", "day", "t"], how="left")
    out = out.with_columns((pl.col("prev_artist") == pl.col("artist")).cast(pl.Float32).alias("sess_prev_same_artist"))
    return out.select(["row"] + [pl.col(c).cast(pl.Float32) for c in NAMES])
