"""Leakage tests.

1. Truncation invariance: features of day-d rows computed on the full log equal those computed on the
   log cut at day_start(d), and do not move when every event from day d on (labels included) is
   scrambled. Runs for the B0 set, B0 on truncated history, the whole R1 grammar and the agent's
   accepted features (artifacts/r5_accepted.json, if present).
2. A row's own label does not enter its features.
3. Stage windows are ordered: first-stage models < booster training rows < evaluation days.
"""
import json

import polars as pl
import pytest

from tbrec.config import ART, STAGES, day_start
from tbrec.data import impressions, load_events
from tbrec.features import DECAY_GRID, WINDOW_GRID, grammar, snapshot
from tbrec.leakcheck import DAY_TESTED, invariance_errors, small_log
from tbrec.specs import B0, BASE


@pytest.fixture(scope="module")
def log():
    return small_log(load_events())


def test_b0_invariant(log):
    assert invariance_errors(log, BASE + B0) == []


def test_b0_truncated_history_invariant(log):
    assert invariance_errors(log, BASE + B0, user_limit=512) == []


def test_grammar_invariant(log):
    assert invariance_errors(log, grammar(DECAY_GRID) + grammar(WINDOW_GRID)) == []


def test_agent_features_invariant(log):
    path = ART / "r5_accepted.json"
    if not path.exists():
        pytest.skip("no agent features yet")
    names = json.loads(path.read_text())["grammar_names"]
    assert invariance_errors(log, names) == []


def test_label_not_in_row_features(log):
    cutoff = day_start(DAY_TESTED)
    rows = impressions(log, [DAY_TESTED]).with_row_index("row")
    a = snapshot(rows, log, B0, cutoff).drop("y").sort("row")
    b = snapshot(rows.with_columns((1 - pl.col("y")).alias("y")), log, B0, cutoff).drop("y").sort("row")
    assert a.equals(b)


def test_stage_windows_are_ordered():
    for s in STAGES.values():
        assert s["first_stage_before"] <= min(s["train"]) and max(s["train"]) < min(s["eval"])
    assert max(STAGES["val"]["eval"]) < min(STAGES["oot"]["eval"])
