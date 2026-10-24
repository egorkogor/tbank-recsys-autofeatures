"""Audio features for all rows (days 273..300) -> artifacts/features/audio.parquet.

  python scripts/build_audio.py prepare   # extract our tracks' embeddings (reads 13.8 GB row group by row group)
  python scripts/build_audio.py build
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tbrec  # noqa: E402,F401
import polars as pl  # noqa: E402

from tbrec import audio  # noqa: E402
from tbrec.config import ART, day_start  # noqa: E402
from tbrec.data import load_events  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    ev = load_events()
    if sys.argv[1] == "prepare":
        audio.prepare(ev["item"].unique().to_numpy())
        print("prepared", f"{time.time() - t0:.0f}s")
        sys.exit()
    idx, E = audio.load()
    rows = pl.read_parquet(ART / "rows.parquet").select("row", "uid", "item", "day")
    parts = []
    for d in sorted(rows["day"].unique().to_list()):
        parts.append(audio.snapshot(rows.filter(pl.col("day") == d), ev, day_start(d), idx, E))
        print(f"day {d}: {parts[-1].height} rows, {time.time() - t0:.0f}s", flush=True)
    out = pl.concat(parts).sort("row")
    out.write_parquet(ART / "features" / "audio.parquet")
    cov = float(out["audio_cos_lp_h30d"].is_not_nan().mean())
    (ART / "features" / "audio.cost.json").write_text(json.dumps({"seconds": time.time() - t0, "coverage": cov, "tracks_with_embedding": int(E.shape[0])}))
    print("done", out.shape, "coverage", round(cov, 3))
