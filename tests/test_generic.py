"""The schema-driven engine reproduces the Yambda engine, and its multi-valued path is consistent."""
import polars as pl
import pytest

from tbrec import features as F
from tbrec import generic as G
from tbrec.config import day_start
from tbrec.data import impressions, load_events
from tbrec.leakcheck import DAY_TESTED, scramble_future, small_log
from tbrec.specs import B0


@pytest.fixture(scope="module")
def log():
    return small_log(load_events())


def _rows(log):
    return impressions(log, [DAY_TESTED]).with_row_index("row")


def test_generic_equals_yambda_engine(log):
    rows, cutoff = _rows(log), day_start(DAY_TESTED)
    names = B0 + F.grammar(["h7d", "w30d", "all"])[:200]
    a = F.snapshot(rows, log, names, cutoff).sort("row")
    b = G.snapshot(rows, log, names, cutoff, G.yambda_schema()).sort("row").select(a.columns)
    assert a.equals(b)


def test_multi_valued_path_matches_single_key(log):
    # artist as a "multi-valued" attribute with exactly one value per track must give the A / UA features
    rows, cutoff = _rows(log), day_start(DAY_TESTED)
    s = G.yambda_schema()
    s.multi = {"MA": ("artist", []), "MUA": ("artist", ["uid"])}
    s.maps = {"artist": log.select("item", "artist").unique()}
    s.unique_keys = s.unique_keys + ("MA",)
    names = [n for n in F.grammar(["h7d", "all"]) if n.split("__")[0] in ("A", "UA")]
    multi = [n.replace("UA__", "MUA__", 1) if n.startswith("UA__") else n.replace("A__", "MA__", 1) for n in names]
    a = G.snapshot(rows, log, names, cutoff, s).sort("row")
    b = G.snapshot(rows, log, multi, cutoff, s).sort("row")
    for n, m in zip(names, multi):
        assert a[n].cast(pl.Float64).fill_null(-1).round(4).equals(b[m].cast(pl.Float64).fill_null(-1).round(4)), n


def test_generic_invariant(log):
    rows, cutoff = _rows(log), day_start(DAY_TESTED)
    s = G.yambda_schema()
    names = s.grammar(G.DECAY_GRID_DAYS)
    full = G.snapshot(rows, log, names, cutoff, s).sort("row")
    assert full.equals(G.snapshot(rows, log.filter(pl.col("t") < cutoff), names, cutoff, s).sort("row"))
    assert full.equals(G.snapshot(rows, scramble_future(log, cutoff), names, cutoff, s).sort("row"))


def test_context_invariant(log):
    # request-time context: features use history before the cutoff plus the row's own time only
    from tbrec import context

    rows, cutoff = context.add(_rows(log)), day_start(DAY_TESTED)
    s, names = context.schema(), context.B0_CTX + context.grammar(["h7d", "all"])[:150]
    full = G.snapshot(rows, context.add(log), names, cutoff, s).sort("row")
    cut = G.snapshot(rows, context.add(log.filter(pl.col("t") < cutoff)), names, cutoff, s).sort("row")
    scr = G.snapshot(rows, context.add(scramble_future(log, cutoff)), names, cutoff, s).sort("row")
    assert full.equals(cut) and full.equals(scr)


def test_mood_invariant(log):
    from tbrec import audio, mood

    if not (audio.EMB.exists() and mood.CLUSTERS.exists()):
        pytest.skip("embeddings / clusters not prepared")
    idx, E = audio.load()
    rows, cutoff = _rows(log), day_start(DAY_TESTED)
    full = mood.snapshot(rows, log, cutoff, idx, E).sort("row")
    cut = mood.snapshot(rows, log.filter(pl.col("t") < cutoff), cutoff, idx, E).sort("row")
    scr = mood.snapshot(rows, scramble_future(log, cutoff), cutoff, idx, E).sort("row")
    assert full.equals(cut) and full.equals(scr)


def test_session_features_see_only_the_past(log):
    # streaming extension: features of an impression must not move when everything at or after its own
    # 5-second bin is scrambled (including the impression's own outcome)
    import numpy as np
    from tbrec import session

    rows = _rows(log).select("row", "uid", "artist", "day", "t")
    a = session.features(rows, log).sort("row")
    rng = np.random.default_rng(1)
    for r in rows.sample(40, seed=3).iter_rows(named=True):
        fut = (log["uid"] == r["uid"]) & (log["t"] >= r["t"])
        scr = log.with_columns(pl.when(fut).then(pl.Series(rng.integers(0, 101, log.height)).cast(pl.Int16)).otherwise(pl.col("pr")).alias("pr"))
        b = session.features(rows.filter(pl.col("row") == r["row"]), scr)
        assert a.filter(pl.col("row") == r["row"]).equals(b), r
