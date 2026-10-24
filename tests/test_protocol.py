"""Protocol tests added after the blind review (24.10): ties in metrics, the OOT gate, candidate generators of
the transfer domains, and what the first-stage models were allowed to read."""
import json

import numpy as np
import polars as pl
import pytest

from tbrec import freeze as fz
from tbrec.config import ROOT, STAGES
from tbrec.metrics import per_group, summary


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"row": np.arange(n, dtype=np.uint32), "uid": rng.integers(0, 80, n), "day": rng.integers(0, 3, n),
                         "y": rng.integers(0, 2, n)})


def test_constant_score_is_a_coin():
    df = _toy().with_columns(pl.lit(1.0).alias("s"))
    assert abs(summary(df, "s")["gauc"] - 0.5) < 1e-12


def test_ndcg_does_not_depend_on_row_order_under_ties():
    df = _toy().with_columns((pl.col("row") % 3).cast(pl.Float64).alias("s"))  # many ties
    a = per_group(df, "s").sort("uid", "day")
    b = per_group(df.sample(fraction=1.0, shuffle=True, seed=5), "s").sort("uid", "day")
    assert np.allclose(a["ndcg"].to_numpy(), b["ndcg"].to_numpy())


def test_oot_gate(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    (art / "runs" / "b0").mkdir(parents=True)
    (art / "features").mkdir()
    for f in ["rows.parquet", "objective.json", "runs/b0/params.json", "features/b0.parquet"]:
        (art / f).write_text("x")
    monkeypatch.setattr(fz, "ART", art)
    monkeypatch.setattr(fz, "FREEZE", art / "freeze.json")
    monkeypatch.setattr(fz, "ROOT", tmp_path)
    monkeypatch.setattr(fz, "git", lambda *a: "abc" if a[0] == "rev-parse" else "")
    monkeypatch.setattr(fz, "branch_files", lambda b: [art / "runs/b0/params.json", art / "features/b0.parquet"])
    with pytest.raises(AssertionError, match="only after the freeze"):
        fz.check("b0")
    m = fz.manifest(["b0"])
    fz.FREEZE.write_text(json.dumps({**m, "branches": ["b0"]}))
    fz.check("b0")  # passes
    with pytest.raises(AssertionError, match="not in the frozen list"):
        fz.check("r1_decay")
    (art / "runs/b0/params.json").write_text("changed")
    with pytest.raises(AssertionError, match="changed after the freeze"):
        fz.check("b0")
    (art / "runs/b0/params.json").write_text("x")
    (art / "runs/b0/pred_oot.parquet").write_text("x")
    with pytest.raises(AssertionError, match="already scored"):
        fz.check("b0")


def test_x5_candidates_ignore_the_future():
    # candidates and their features for week c come only from events before c (labels may differ)
    from tbrec import x5

    path = ROOT / "data" / "processed" / "x5" / "events.parquet"
    if not path.exists():
        pytest.skip("X5 not prepared")
    ev = pl.read_parquet(path).filter(pl.col("uid") < 400)
    c = int(ev["day"].max()) - 13
    n_items = int(ev["item"].max()) + 1
    fut = ev["day"] >= c
    scr = ev.with_columns(pl.when(fut).then(pl.Series(np.random.default_rng(0).integers(0, n_items, ev.height)).cast(pl.UInt32))
                          .otherwise(pl.col("item")).alias("item"))
    cols = ["uid", "item", "cand_hist_rank", "cand_pop_rank", "cand_ials_rank"]
    users = x5._candidates(ev, c, n_items).select("uid").unique()
    a = x5._candidates(ev, c, n_items).join(users, on="uid", how="semi").select(cols).sort("uid", "item")
    b = x5._candidates(scr, c, n_items).join(users, on="uid", how="semi").select(cols).sort("uid", "item")
    assert a.equals(b)


def test_first_stage_models_read_only_the_allowed_past():
    ials = ROOT / "artifacts" / "features" / "ials_val.cost.json"
    if ials.exists():
        assert json.loads(ials.read_text())["trained_before_day"] == STAGES["val"]["first_stage_before"]
    audit = ROOT / "external" / "perseus_stage2" / "label_access.jsonl"
    if audit.exists():
        for line in audit.read_text().splitlines():
            r = json.loads(line)
            assert r["max_day"] <= max(STAGES["val"]["eval"]), "Perseus read OOT labels"
            if r["start"] >= min(STAGES["val"]["eval"]):
                assert r["val_A_only"], "Perseus read val half B labels"
