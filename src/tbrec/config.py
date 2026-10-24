"""Paths, time axis and the train / val / OOT windows (day indices)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "yambda"
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

DAY = 86_400
# Yambda-50M timestamps are seconds in [0, 26_000_000], rounded to 5 s.
# Days are counted back from the end of the log, so day 300 is exactly the last 24 h
# (the test day of the Yambda paper's Global Temporal Split).
T_END = 26_000_005
N_DAYS = 301
OFFSET = N_DAYS * DAY - T_END  # 6395 s; day d covers [d*DAY - OFFSET, (d+1)*DAY - OFFSET)


def day_start(d: int) -> int:
    """Cutoff of day d: features of a row on day d see only events with t < day_start(d)."""
    return d * DAY - OFFSET


# Stage "val": model selection. Stage "oot": the single final run, window shifted by 7 days.
# first_stage_before: models whose scores feed the booster (iALS, Perseus, ...) train only on
# events of days < this value, i.e. before the first training row of the booster.
STAGES = {
    "val": dict(train=list(range(273, 287)), eval=list(range(287, 294)), first_stage_before=273),
    "oot": dict(train=list(range(280, 294)), eval=list(range(294, 301)), first_stage_before=280),
}

SEED = 20261024
