"""Streaming extension features (src/tbrec/session.py) for all rows -> artifacts/features/session.parquet."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec import session  # noqa: E402
from tbrec.config import ART  # noqa: E402
from tbrec.data import load_events  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "artist", "day", "t")
    ev = load_events()
    parts = [session.features(rows.filter(pl.col("day") == d), ev.filter((pl.col("day") >= d - 1) & (pl.col("day") <= d + 1)))
             for d in sorted(rows["day"].unique().to_list())]
    out = pl.concat(parts).sort("row")
    out.write_parquet(ART / "features" / "session.parquet")
    (ART / "features" / "session.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "features": len(session.NAMES)}))
    print("done", out.shape, f"{time.time() - t0:.0f}s", out.null_count().to_dicts())
