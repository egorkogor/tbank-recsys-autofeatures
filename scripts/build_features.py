"""Build a feature table for impression rows of the given days.

A row's features depend only on its day (snapshot at the day's start), not on the stage, so one table
serves both the val and the OOT stage.

  python scripts/build_features.py rows                     # artifacts/rows.parquet (days 273..300)
  python scripts/build_features.py b0                       # BASE + B0 + rhythm + track length
  python scripts/build_features.py grammar --grid decay --days 274,277,280,283,286,287,289,291,293 --users-frac 0.3 --val-half A
  python scripts/build_features.py names --names-file artifacts/r1_selected.json --out r1
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401  (caps CPU before polars starts its thread pool)
import polars as pl  # noqa: E402

from tbrec.config import ART, STAGES  # noqa: E402
from tbrec.data import impressions, load_events, load_items  # noqa: E402
from tbrec.features import DECAY_GRID, WINDOW_GRID, build, grammar  # noqa: E402
from tbrec.specs import B0, BASE  # noqa: E402

ROWS = ART / "rows.parquet"
FEAT = ART / "features"
ALL_DAYS = sorted(set(STAGES["val"]["train"] + STAGES["val"]["eval"] + STAGES["oot"]["train"] + STAGES["oot"]["eval"]))


def user_bucket(uid: int, salt: str) -> float:
    h = hashlib.sha256(f"{salt}|{uid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def make_rows(ev: pl.DataFrame) -> pl.DataFrame:
    r = impressions(ev, ALL_DAYS).sort("day", "uid", "t", "item").with_row_index("row")
    users = r["uid"].unique().to_list()
    # val_half: model selection on half "A", confirmation on half "B" (fixed hash)
    # sel: a fixed 30% of users for the feature search on training days
    b = pl.DataFrame({"uid": users,
                      "val_half": ["A" if user_bucket(u, "half-20261024") < 0.5 else "B" for u in users],
                      "sel": [user_bucket(u, "sel-20261024") < 0.3 for u in users]}).cast({"uid": pl.UInt32})
    return r.join(b, on="uid", how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rows", "b0", "b0_trunc", "grammar", "names"])
    ap.add_argument("--user-limit", type=int, default=512)
    ap.add_argument("--grid", choices=["decay", "window"], default="decay")
    ap.add_argument("--days", default="")
    ap.add_argument("--users-frac", type=float, default=0.0, help="keep only users with sel=True on training days")
    ap.add_argument("--val-half", default="", help="keep only this half of val users on val days")
    ap.add_argument("--names-file", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    FEAT.mkdir(parents=True, exist_ok=True)
    ev = load_events()
    if a.what == "rows":
        r = make_rows(ev)
        r.write_parquet(ROWS)
        print(r.group_by("day").len().sort("day").to_dicts()[:3], r.height)
        return
    rows = pl.read_parquet(ROWS).select("row", "uid", "item", "artist", "album", "day", "t", "y", "val_half", "sel")
    days = [int(d) for d in a.days.split(",")] if a.days else ALL_DAYS
    rows = rows.filter(pl.col("day").is_in(days))
    if a.users_frac:
        train_days = set(STAGES["val"]["train"])
        rows = rows.filter(~pl.col("day").is_in(list(train_days)) | pl.col("sel"))
    if a.val_half:
        val_days = set(STAGES["val"]["eval"])
        rows = rows.filter(~pl.col("day").is_in(list(val_days)) | (pl.col("val_half") == a.val_half))
    limit = None
    if a.what == "b0":
        names, rhythm, out = BASE + B0, True, "b0"
    elif a.what == "b0_trunc":
        names, rhythm, out, limit = BASE + B0, True, f"b0_trunc{a.user_limit}", a.user_limit
    elif a.what == "grammar":
        names, rhythm, out = grammar(DECAY_GRID if a.grid == "decay" else WINDOW_GRID), False, f"grammar_{a.grid}"
    else:
        names, rhythm, out = json.loads(Path(a.names_file).read_text())["names"], False, a.out
    t0 = time.time()
    print(f"{out}: {len(names)} features, {rows.height} rows, {len(days)} days", flush=True)
    x = build(rows.drop("val_half", "sel"), ev, names, with_rhythm=rhythm, user_limit=limit,
              log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True))
    if a.what in ("b0", "b0_trunc"):
        x = x.join(load_items(), on="item", how="left")
    x = x.drop("uid", "item", "artist", "album", "day", "t", "y").sort("row")
    x.write_parquet(FEAT / f"{out}.parquet")
    (FEAT / f"{out}.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "rows": x.height, "features": len(names), "days": len(days)}))
    print("done", x.shape, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
