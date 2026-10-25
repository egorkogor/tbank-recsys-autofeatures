"""Small, disk-isolated regression cases for the post-result label checks.

No production rows, model training, scoring, or OOT reads are used.
"""
import polars as pl
import pytest
from scripts import label_checks as checks


def test_repeat_pairs_require_strictly_later_time_and_same_day():
    rows = pl.DataFrame({
        "uid": [1] * 8, "item": [7] * 8,
        "day": [292, 292, 292, 293, 293, 293, 294, 294],
        "t": [10, 10, 20, 30, 30, 40, 50, 60],
        "y": [1, 1, 0, 0, 0, 1, 1, 1],
    })
    result = checks.repeat_agreement(rows)
    # Equal-time duplicates, cross-day links, and OOT pairs must not count.
    assert result["pairs"] == 2
    assert result["same_outcome"] == 0
    assert result["chance_same_outcome"] == 0.5


@pytest.fixture
def session_case(tmp_path, monkeypatch):
    proc, art = tmp_path / "processed", tmp_path / "artifacts"
    proc.mkdir()
    for branch in ("b0", "b0_sess"):
        (art / "runs" / branch).mkdir(parents=True)
    events = pl.DataFrame({
        "uid": [1] * 9, "kind": [0] * 9,
        "day": [287] * 7 + [288, 288],
        "t": [10, 10, 10, 20, 100, 100, 110, 86410, 86420],
        "item": [7, 7, 8, 9, 7, 8, 7, 7, 7],
    })
    events.write_parquet(proc / "events.parquet")
    rows = events.unique().sort("day", "t", "item").with_row_index("row").drop("kind")
    rows = rows.with_columns((pl.col("row") % 2).alias("y"), pl.lit("B").alias("val_half"))
    for branch in ("b0", "b0_sess"):
        rows.select("row", pl.col("row").cast(pl.Float64).alias("score")).write_parquet(
            art / "runs" / branch / "pred_val.parquet")
    monkeypatch.setattr(checks, "PROC", proc)
    monkeypatch.setattr(checks, "ART", art)
    calls = []
    def bootstrap(frame, names, base):
        calls.append(frame.clone())
        return {"b0": {"gauc": 0.5}, "b0_sess": {
            "gauc": 0.6, "d_gauc": 0.1, "d_gauc_ci": [0.05, 0.15]}}
    monkeypatch.setattr(checks, "paired_bootstrap", bootstrap)
    return rows, calls, art


def test_session_filters_same_time_duplicates_day_boundary_and_earlier_track(session_case):
    rows, calls, _ = session_case
    result = checks.session_check(rows)
    all_rows, no_today, no_previous = calls
    assert result["rows"] == rows.height == 8  # exact event duplicate did not multiply joins
    flags = {(r["day"], r["t"], r["item"]): (r["seen_today"], r["prev_same"])
             for r in all_rows.to_dicts()}
    assert flags[(287, 10, 7)] == (False, False)
    assert flags[(287, 10, 8)] == (False, False)  # other track in the same time bucket
    assert flags[(287, 100, 7)] == (True, False)  # earlier today, but previous bucket is track 9
    assert flags[(287, 110, 7)] == (True, True)
    assert flags[(288, 86410, 7)] == (False, False)  # yesterday is not today
    assert flags[(288, 86420, 7)] == (True, True)
    assert no_today.height == 4
    assert no_previous.height == 6
    assert not no_today["seen_today"].any()
    assert not no_previous["prev_same"].any()


@pytest.mark.parametrize("fault", ["duplicate", "missing"])
def test_session_rejects_duplicate_row_or_missing_prediction(session_case, fault):
    rows, _, art = session_case
    path = art / "runs/b0/pred_val.parquet"
    pred = pl.read_parquet(path)
    pred = pl.concat([pred, pred.head(1)]) if fault == "duplicate" else pred.slice(1)
    pred.write_parquet(path)
    with pytest.raises(AssertionError):
        checks.session_check(rows)
