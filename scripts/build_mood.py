"""Listener-state proxies for Yambda (src/tbrec/mood.py) -> artifacts/features/mood.parquet.

  python scripts/build_mood.py clusters   # 32 k-means clusters of the audio embeddings (content only)
  python scripts/build_mood.py build      # sound x time-of-day cosines + user x cluster (x part of day) rates
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec import context, generic, mood  # noqa: E402
from tbrec.audio import load as load_emb  # noqa: E402
from tbrec.config import ART, day_start  # noqa: E402
from tbrec.data import load_events  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    if sys.argv[1] == "clusters":
        mood.build_clusters()
        print("clusters", f"{time.time() - t0:.0f}s")
        sys.exit()
    idx, E = load_emb()
    ev = mood.add_cluster(context.add(load_events()))
    rows = mood.add_cluster(context.add(pl.read_parquet(ART / "rows.parquet").select("row", "uid", "item", "artist", "album", "day", "t", "y")))
    s = context.schema()
    s.keys = {**s.keys, **mood.MOOD_KEYS}
    parts = []
    for d in sorted(rows["day"].unique().to_list()):
        r = rows.filter(pl.col("day") == d)
        a = mood.snapshot(r, ev, day_start(d), idx, E)
        g = generic.snapshot(r, ev, mood.HAND, day_start(d), s).select(["row"] + mood.HAND)
        parts.append(a.join(g, on="row"))
        print(f"day {d}: {r.height} rows, {time.time() - t0:.0f}s", flush=True)
    out = pl.concat(parts).sort("row")
    out.write_parquet(ART / "features" / "mood.parquet")
    (ART / "features" / "mood.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "features": out.width - 1}))
    print("done", out.shape)
