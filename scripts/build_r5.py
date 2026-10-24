"""Accepted agent features (R5) for all rows -> artifacts/features/r5.parquet; branch r1_r5 = R1 + these.

Composite names (ratio / diff) are computed from their grammar components with the Yambda engine.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec.config import ART  # noqa: E402
from tbrec.data import load_events  # noqa: E402
from tbrec.features import build  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r5_eval import components, materialise  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    acc = json.loads((ART / "r5_state.json").read_text())["accepted"]
    assert acc, "no accepted agent features"
    comps = sorted({c for n in acc for c in components(n)})
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "item", "artist", "album", "day", "t", "y")
    x = build(rows, load_events(), comps, log=lambda s: print(s, f"{time.time() - t0:.0f}s", flush=True))
    x = x.with_columns([materialise(x, n).cast(pl.Float32).alias(n) for n in acc if n not in x.columns])
    x.select(["row"] + acc).sort("row").write_parquet(ART / "features" / "r5.parquet")
    (ART / "features" / "r5.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "features": len(acc)}))
    print("done", acc, f"{time.time() - t0:.0f}s")
